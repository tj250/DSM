import time, json, os
import shutil
import requests, uuid
from langgraph.types import interrupt
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import config
from data_access.dist_computing_nodes import get_computing_nodes
from agents.data_structure.build_model_state import BuildModelState
from agents.data_structure.task_state import TaskState
from data_access.task import TaskDataAccess
from data_access.build_model import BuildModelDataAccess
from .utils.views import BMUIChoice, BMActiveState, print_agent_output, TaskStage
from .base_agent import BaseAgent
from agents.utils.visualization import save_graph_to_png
from eda.analysis import execute_analysis
from data_access.structure_data_dealer import StructureDataDealer
from heuristic_rules.mlr import hypothesis_testing as mlr_hypothesis_testing
from heuristic_rules.elastic_net import hypothesis_testing as elastic_net_hypothesis_testing
from DSMAlgorithms import AlgorithmsType
from DSMAlgorithms import DataDistributionType

'''
建模智能体
'''


class BuildModelAgent(BaseAgent):
    """
    初始化子图结构
    """

    def init_workflow(self, checkpointer):
        subgraph_builder = StateGraph(BuildModelState)
        subgraph_builder.add_node("启发式过滤", self.heuristics_filter)
        subgraph_builder.add_node("参数搜索", self.params_search)
        subgraph_builder.add_node("用户确认参数搜索结果", self.user_confirm_param_search_result)
        subgraph_builder.add_node("模型堆叠", self.stacking)

        subgraph_builder.add_edge(START, "启发式过滤")
        subgraph_builder.add_edge("启发式过滤", "参数搜索")
        subgraph_builder.add_edge("参数搜索", "用户确认参数搜索结果")
        # 用户最终复核建模的结果，要么结束流程，进入下一个环节，要么进一步堆叠建模结果
        subgraph_builder.add_conditional_edges(
            "用户确认参数搜索结果",
            lambda build_model_state: build_model_state["choice"],
            path_map={
                "优化建模结果": "模型堆叠",  # 跳转到模型堆叠节点
                "用户已确认参数搜索结果": END  # 跳转到用户确认最终预测节点
            }
        )
        subgraph_builder.add_edge("模型堆叠", END)
        self.subgraph = subgraph_builder.compile(checkpointer=checkpointer)

        # 可视化图结构
        save_graph_to_png(self.subgraph.get_graph(), 'build_model')

    '''
    图节点函数：启发式过滤
    结合ESDA，对12类算法及扩充算法进行初步筛选，以过滤掉不合适的算法
    '''

    async def heuristics_filter(self, state: BuildModelState):
        print_agent_output(f"执行建模，第一阶段启动-启发式算法过滤... 调用到heuristics_filter", agent="BUILDMODEL")
        task = state.get('task')
        build_model_id = BuildModelDataAccess.query_build_model_id(task.task_id)  # 检测任务是否已经有建模记录
        if build_model_id is not None:
            return {} # 表示曾经已经经过启发式过滤，直接跳过该节点
        params_for_build_model = state.get("params_for_build_model")
        esda_rsult = state.get('esda_result')
        # 统一处理
        gpd = StructureDataDealer.data_cleaning(task.sample_file, params_for_build_model.continuous_variables, False)
        y = gpd.pop(params_for_build_model.prediction_variable)
        if esda_rsult.with_geometry:
            X = gpd.drop(config.DF_GEOM_COL, axis=1)
        else:
            X = gpd
        filtered_algorithms = []
        if config.DEBUG:  # config.DEBUG:  # 仅用于快速调试
            filtered_algorithms.append(AlgorithmsType.GLM.value)
            # filtered_algorithms.append(AlgorithmsType.SVR.value)
            # filtered_algorithms.append(AlgorithmsType.PLSR.value)
            # filtered_algorithms.append(AlgorithmsType.MLP.value)
        else:
            # 针对不同原理的算法，逐个筛选每个算法是否适合参与下一阶段的参数搜索过程
            # 2、偏最小二乘回归：适合处理多重共线性、高维小样本数据，对响应变量分布无要求。
            if esda_rsult.is_few_samples and esda_rsult.is_high_dimension:  # plsr适合处理小样本高维数据
                filtered_algorithms.append(AlgorithmsType.PLSR.value)
            # 3、广义线性回归：需要确定响应变量的分布类型，错误的分布类型设定会导致偏差，连续非正态数据可使用伽马分布，正态分布则使用MLR（普通多元线性回归）
            if esda_rsult.data_distribution != DataDistributionType.Unknown:  # 如果数据分布明确，则可以采用GLM进行建模
                filtered_algorithms.append(AlgorithmsType.GLM.value)
            # 4、弹性网络：适合处理多重共线性、高维数据，但需要对响应变量需进行正态分布验证，必要时进行变换，符合正态分布后使用，还需执行特征标准化，异常值剔除
            if elastic_net_hypothesis_testing(X, y, params_for_build_model.continuous_variables):
                filtered_algorithms.append(AlgorithmsType.EN.value)
            # 5、多尺度地理加权回归：对响应变量分布无要求（局部回归降低了依赖），适用于空间异质性显著区域,MGWR对数百个要素的数据集较为有效
            if len(y) > 200 and esda_rsult.with_geometry and esda_rsult.have_heterogeneity:  # 如果因变量存在空间异质性,并且样点数量较为充足
                filtered_algorithms.append(AlgorithmsType.MGWR.value)
            # 6、回归克里金：若采用线性回归拟合趋势项，要求‌残差近似正态分布，是否需避免多重共线性视采用的线性回归方法而定，
            # 残差需满足‌空间平稳性‌（均值、方差恒定）和‌正态性假设‌，以构建合理的协方差模型‌。
            # 响应变量需要具有显著的空间自相关性。样本量需30个以上。当存在大尺度（如海拔）和小尺度（如植被覆盖）共同作用时，回归部分捕捉趋势（趋势项），
            # 克里金修正局部波动‌(残差项)
            if esda_rsult.with_geometry and not esda_rsult.is_few_samples:
                filtered_algorithms.append(AlgorithmsType.RK.value)  # 由于回归克里金模型可以选择线性模型或随机森林等机器学习模型，完全依赖后验即可
            # 7、协同克里金：响应变量（主变量）及协变量（辅助变量）需要具有空间自相关性。协变量与主变量之间还应存在统计相关性（线性相关，指数相关等）。
            if esda_rsult.with_geometry:
                filtered_algorithms.append(AlgorithmsType.CK.value)
            # 8、支持向量回归，已分解为了四个自定义的不同的核的模型
            if esda_rsult.is_few_samples and esda_rsult.is_high_dimension:  # svr适合处理小样本高维数据
                filtered_algorithms.append(AlgorithmsType.SVR.value)
            # 9、K近邻回归小样本时模型不稳定
            if not esda_rsult.is_few_samples:
                filtered_algorithms.append(AlgorithmsType.KNR.value)
            # 10、随机森林回归，必选
            filtered_algorithms.append(AlgorithmsType.RFR.value)
            # 11、极限梯度提升树回归，必选
            filtered_algorithms.append(AlgorithmsType.XGBR.value)
            # 12、多层感知机要求的样本数量不能太少
            if len(y) > 200 and len(y) * (1 - 1 / config.KFOLD) > len(X.columns):  # 注意分割后的样本数必须大于特征数
                filtered_algorithms.append(AlgorithmsType.MLP.value)

            # 以下处理所有扩充的模型
            filtered_custom_models = []  # 过滤后的自定义算法
            custom_models = BuildModelDataAccess.get_extend_models()
            for custom_model in custom_models:
                # 如果样本数量过少，并且算法不支持小样本，则跳过
                if esda_rsult.is_few_samples and not custom_model.can_deal_small_samples:
                    continue
                # 如果是高维数据，并且算法不支持高维数据的处理，则跳过
                if esda_rsult.is_high_dimension and not custom_model.can_deal_high_dims:
                    continue
                # 如果数据分布未知，并且算法无处处理未知分布数据，则跳过
                if esda_rsult.data_distribution == DataDistributionType.Unknown and not custom_model.can_deal_unknown_distribution:
                    continue
                # 如果样本数据不带坐标，但是模型要求带有坐标，则跳过
                if not esda_rsult.with_geometry and custom_model.X_with_geometry:
                    continue
                # 如果样本数据具有空间异质性，但是模型无法处理带有空间异质性的数据，则跳过
                if esda_rsult.have_heterogeneity and not custom_model.can_deal_heterogeneity:
                    continue
                filtered_custom_models.append(custom_model)

        # 启发式过滤的最后，需要将过滤后的算法列表写入数据库，以开启一次新的建模批次
        algorithms_info = {}
        for algorithm in filtered_algorithms:
            algorithms_info[algorithm] = str(uuid.uuid1())  # 对每一种算法，生成一个唯一的algorithms_id
        if not config.DEBUG:
            for model in filtered_custom_models:  # 对于自定义模型
                algorithms_info[model.custom_model_name] = str(uuid.uuid1())  # 对每一种自定义算法，生成一个唯一的algorithms_id
        # 向分布式集群发送建模请求前，将调整后数据源文件和类别型变量文件信息写入数据库，以便分布式集群在异常后可以基于这些信息恢复算法的执行
        successful, build_model_id = BuildModelDataAccess.record_task_batch(task, algorithms_info,
                                                                            params_for_build_model.prediction_variable,
                                                                            params_for_build_model.categorical_vars_detail,
                                                                            esda_rsult.data_distribution,
                                                                            esda_rsult.left_cols_after_rfe,
                                                                            esda_rsult.left_cols_after_fia,
                                                                            esda_rsult.left_cols_after_rfe_fia)
        if not successful:
            raise ValueError('未能在数据库中记录启发式过略的结果！')
        task.last_build_model_id = build_model_id  # 需要更新任务状态信息
        return {"task": task,
                "algorithms_info": algorithms_info,
                "build_model_state": BMActiveState.HeuristicsFiltered}

    '''
    图节点函数，执行参数搜索（将参数搜索的要求发送到分布式部署的算法）
    '''

    async def params_search(self, state: BuildModelState):
        print_agent_output(f"执行建模，第二阶段启动-多模型分布式参数搜索... 调用到params_search", agent="BUILDMODEL")
        while True:
            task = state.get('task')
            if state.get("build_model_state") == BMActiveState.HeuristicsFiltered:  # 之前persistant中记录的状态为启发式规则过滤结束
                exist_unfinished_searching = BuildModelDataAccess.exists_task_batch(task.last_build_model_id)
                if exist_unfinished_searching:
                    state["build_model_state"] = BMActiveState.ParamsSearching
                    continue  # 跳过发送请求到分布式集群，进入等待单算法建模结束状态
                algorithms_info = state.get("algorithms_info")
                params_for_build_model = state.get("params_for_build_model")
                sample_file_name = os.path.basename(task.sample_file)
                payload = {'prediction_variable': params_for_build_model.prediction_variable,
                           'categorical_vars': params_for_build_model.categorical_vars_detail,
                           'data_distribution_type': state.get('esda_result').data_distribution.value,
                           'left_cols1': ','.join(state.get('esda_result').left_cols_after_rfe),
                           'left_cols2': ','.join(state.get('esda_result').left_cols_after_fia),
                           'left_cols3': ','.join(state.get('esda_result').left_cols_after_rfe_fia)}
                if config.DOCKER_MODE:  # 如果算法服务是包装为Docker运行
                    computing_nodes_url, mapping_paths = get_computing_nodes()
                else:
                    computing_nodes_url = ['http://localhost:8389']
                    mapping_paths = ['E:\\data\\DSM_Test']
                docker_instance_count = len(computing_nodes_url)  # 算法容器实例数量
                now_instance = 1
                for algorithm_name, algorithm_id in algorithms_info.items():
                    target_data_path = mapping_paths[now_instance % docker_instance_count]
                    request_url = computing_nodes_url[now_instance % docker_instance_count]
                    request_url += "/call_model"
                    now_instance += 1
                    payload['algorithm'] = algorithm_name
                    payload['algorithm_id'] = algorithm_id
                    payload['file_name'] = sample_file_name
                    payload['covariates_directory'] = os.path.basename(task.covariates_path)  # 协变量所在的目录名称

                    if config.DOCKER_MODE:
                        # 将建模相关环境协变量复制到宿主机的映射目录下，供docker中的应用访问
                        target_covariates_path = os.path.join(target_data_path, os.path.basename(task.covariates_path))
                        if not os.path.exists(target_covariates_path):  # 目标目录下不存在要复制的样点数据文件
                            shutil.copytree(task.covariates_path, target_covariates_path)
                    else:  # 本地模式
                        pass
                    try:
                        response = requests.post(
                            request_url,
                            data=json.dumps(payload),
                            headers={'Content-Type': 'application/json'}
                        )
                        response.raise_for_status()  # 检查HTTP错误
                        if response.status_code != 200:  # http未返回成功状态码
                            raise ValueError('分布式计算出现错误！')
                    except Exception as e:
                        print(f'error:{e}')
                        raise
                state["build_model_state"] = BMActiveState.ParamsSearching  # 已经将建模请求发送至分布式计算集群
            elif state.get("build_model_state") == BMActiveState.ParamsSearching:  # 之前persistant中记录的状态为执行中
                # 轮询之前开启的建模执行状态
                self.show_ui_progress("轮询回归建模任务是否全部结束...")
                finished = BuildModelDataAccess.task_batch_is_finished(task.last_build_model_id)
                if finished:
                    build_model_results = BuildModelDataAccess.get_model_metrics(task.last_build_model_id)
                    return {"build_model_results": build_model_results,
                            "build_model_state": BMActiveState.ParamsSearched}  # 分布式计算集群已完成所有建模
                else:
                    time.sleep(10)  # 等待10秒开始下一轮循环

    '''
    图节点函数：请求用户对参数搜索的结果做出确认
    '''

    def user_confirm_param_search_result(self, state: BuildModelState):
        print_agent_output(f"请求用户确认建模结果... user_confirm_param_search_result", agent="BUILDMODEL")
        self.interrput_event.set()  # 通知等待线程可以执行
        choice = interrupt('')
        if choice == BMUIChoice.EnterNextStep:
            self.quit_chat = True
            return {"choice": "用户已确认参数搜索结果"}
        else:
            return {"choice": "优化建模结果"}

    '''
    图节点函数，堆叠异构模型
    '''

    async def stacking(self, state: BuildModelState):
        print_agent_output(f"执行建模，第三阶段启动-模型堆叠... 调用到stacking", agent="BUILDMODEL")
        while True:
            task = state.get('task')
            if state.get("build_model_state") == BMActiveState.ParamsSearched:  # 之前persistant中记录的状态为启发式规则过滤结束
                exist_unfinished_stacking = BuildModelDataAccess.exists_stacking(task.last_build_model_id)
                if exist_unfinished_stacking:
                    state["build_model_state"] = BMActiveState.Stacking
                    continue  # 跳过发送请求到分布式集群，进入等待堆叠建模结束状态
                params_for_build_model = state.get("params_for_build_model")
                sample_file_name = os.path.basename(task.sample_file)
                model_metrics = BuildModelDataAccess.get_model_metrics(task.last_build_model_id)  # 获取排序的建模结果信息
                stacking_algorithms = {}
                for i in range(len(model_metrics)):
                    is_suitable_tacking = BuildModelDataAccess.check_model_suitable_for_stacking(
                        model_metrics[i].algorithms_name)
                    if not is_suitable_tacking:
                        continue
                    for j in range(i + 1, len(model_metrics)):
                        is_suitable_tacking = BuildModelDataAccess.check_model_suitable_for_stacking(
                            model_metrics[j].algorithms_name)
                        if not is_suitable_tacking:
                            continue
                        stacking_algorithms_name = '|'.join([model_metrics[i].algorithms_name,
                                                             model_metrics[j].algorithms_name])
                        stacking_algorithms[stacking_algorithms_name] = str(uuid.uuid1())  # 对每一种组合算法生成一个唯一的id
                successful = BuildModelDataAccess.record_task_stacking(task, stacking_algorithms)
                payload = {'prediction_variable': params_for_build_model.prediction_variable,
                           'categorical_vars': params_for_build_model.categorical_vars_detail,
                           'data_distribution_type': state.get('esda_result').data_distribution.value,
                           'left_cols1': ','.join(state.get('esda_result').left_cols_after_rfe),
                           'left_cols2': ','.join(state.get('esda_result').left_cols_after_fia),
                           'left_cols3': ','.join(state.get('esda_result').left_cols_after_rfe_fia)}
                if config.DOCKER_MODE:  # 如果算法服务是包装为Docker运行
                    computing_nodes_url, mapping_paths = get_computing_nodes()
                else:
                    computing_nodes_url = ['http://localhost:8389']
                    mapping_paths = ['E:\\data\\DSM_Test']
                docker_instance_count = len(computing_nodes_url)  # 算法容器实例数量
                now_instance = 1
                for algorithms, algorithms_id in stacking_algorithms.items():
                    target_data_path = mapping_paths[now_instance % docker_instance_count]
                    request_url = computing_nodes_url[now_instance % docker_instance_count]
                    request_url += "/stacking"
                    now_instance += 1
                    payload['stacking_algorithms'] = algorithms
                    payload['stacking_algorithms_id'] = algorithms_id
                    payload['file_name'] = sample_file_name
                    payload['covariates_directory'] = os.path.basename(task.covariates_path)  # 协变量所在的目录名称

                    if config.DOCKER_MODE:
                        # 将建模相关环境协变量复制到宿主机的映射目录下，供docker中的应用访问
                        shutil.copytree(task.covariates_path,
                                        os.path.join(target_data_path, os.path.basename(task.covariates_path)))
                    else:  # 本地模式
                        pass
                    try:
                        response = requests.post(
                            request_url,
                            data=json.dumps(payload),
                            headers={'Content-Type': 'application/json'}
                        )
                        response.raise_for_status()  # 检查HTTP错误
                        if response.status_code != 200:  # http未返回成功状态码
                            raise ValueError('分布式计算出现错误！')
                    except Exception as e:
                        print(f'error:{e}')
                        raise
                state["build_model_state"] = BMActiveState.Stacking  # 已经将堆叠请求发送至分布式计算集群
            elif state.get("build_model_state") == BMActiveState.Stacking:  # 之前persistant中记录的状态为执行中
                # 轮询之前开启的建模执行状态
                self.show_ui_progress("轮询堆叠任务是否全部结束...")
                finished = BuildModelDataAccess.stacking_is_finished(task.last_build_model_id)
                if finished:
                    build_model_results = BuildModelDataAccess.get_model_metrics(task.last_build_model_id, True)
                    self.quit_chat = True  # 重要，指示子图执行完毕
                    return {"build_model_results": build_model_results,
                            "build_model_state": BMActiveState.Finished}  # 分布式计算集群已完成所有建模
                else:
                    time.sleep(10)  # 等待10秒开始下一轮循环


    '''
    异步的agent主控函数：定义一个完成建模的子图，完成对数据挖掘模型的建模。
    采用Evaluator-Optimizer设计模式，根据建模的数据信息，调用外部工具完成建模，并对建模结果进行评估，最后交由用户确认后结束建模过程
    '''

    async def async_run(self, global_state: TaskState) -> TaskState:
        print_agent_output(f"开始建模...", agent="BUILDMODEL")
        async with AsyncSqliteSaver.from_conn_string('./build_model.db') as checkpointer:
            # 初始化子图
            self.init_workflow(checkpointer)

            # 启动子图的执行，进入建模节点
            thread_config = {"configurable": {"thread_id": global_state["task"].task_id}}

            if global_state["task"].stage != TaskStage.BuildModel:  # 如果当前任务不处于建模的状态，直接跳出
                state = await self.subgraph.aget_state(thread_config)
                return {}

            # 获取子图的历史状态快照
            history_states = [state async for state in self.subgraph.aget_state_history(thread_config)]
            if len(history_states) > 0:  # 存在历史的check points,处于恢复模式（从长任务中恢复历史上的执行状态）
                if len(history_states[0].next) == 0:
                    print("checkpoint异常，可能是从调试中恢复！！！")
                    self.interrput_event.set()
                    self.quit_chat = True
                else:
                    await self.subgraph.ainvoke(None, history_states[0].config)
            else:  # 否则，初次执行
                await self.subgraph.ainvoke(input={"task": global_state["task"],
                                                   "params_for_build_model": global_state["params_for_build_model"],
                                                   "esda_result": global_state["esda_result"]
                                                   }, config=thread_config)
            # 一直循环，直到得到结果
            while True:
                # 1、阻塞，直到agent的interrupt被调用
                self.interrput_event.wait()
                # 延迟半秒，确保如果经历evaluate_task-》accept_input时，图的执行线程可以再一次进入accept_input节点
                time.sleep(0.5)
                # 判断是否需要继续用户反馈
                if self.quit_chat:  # 无需再接收用户反馈（意味着子图已经执行完毕）
                    break
                # 重置信号,以便进入下一轮的阻塞
                self.interrput_event.clear()

                # 2、执行业务逻辑--由用户给出反馈
                suggestion = ""
                # Get the graph state to get interrupt information.
                state = await self.subgraph.aget_state(thread_config)  # 获取最后一次持久化保存的checkpoint状态

                # 根据获取的最后的持久化记录进行相应的处理
                input_finished_event = self.agent_signals.create_input_finished_event()  # 创建一个python的事件，用于阻塞线程，直到获取用户的反馈
                input_finished_event.clear()  # 重置信号，以便用户给出反馈

                # if state.next[0] == "用户确认参数搜索结果":  # 请求用户对对参数搜索结果进行确认
                confirm_info_list = state.values["build_model_results"]
                self.agent_signals.user_confirm_build_model_result.emit(self.agent_signals.task_id,
                                                                        input_finished_event,
                                                                        confirm_info_list)  # 请求用户确认建模结果

                input_finished_event.wait()  # 阻塞，直到输入结束事件被触发(获取到了用户的反馈)

                # 3、恢复interrupt的执行
                await self.subgraph.ainvoke(Command(resume=input_finished_event.data), config=thread_config)

            # 将任务阶段信息写入数据库，指示进入建模阶段build_model_results
            last_state = await self.subgraph.aget_state(thread_config)  # 获取建模的最后状态参数
            # latest_state_values = last_state.values
            # 将建模结果信息写入数据库
            # BuildModelDataAccess.record_task_build_model_result(self.agent_signals.task_id)
            TaskDataAccess.update_task_stage(self.agent_signals.task_id, TaskStage.Evaluation)
            # 更新全局的State状态
            global_state["task"].stage = TaskStage.Evaluation  # 指示进入下一环节
            self.show_ui_progress("即将进入预测制图阶段...")
            return {"task": global_state["task"]}
