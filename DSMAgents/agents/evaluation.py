import time
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from data_access.mapping import MappingDataAccess
from .utils.views import TaskStage, print_agent_output, UserConfirmUIChoice
from agents.data_structure.evaluation_state import EvaluationState
from agents.data_structure.task_state import TaskState
from data_access.task import TaskDataAccess
from data_access.build_model import BuildModelDataAccess
from .base_agent import BaseAgent
from agents.utils.visualization import save_graph_to_png
from data_access.mapping import mapping, create_uncertainty_map, compute_indepent_test_metrics
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

'''
评估智能体，通过人机交互方式，实现预测信息。
需要处理的业务：
1、制图
2、准确性评估
3、不确定性评估
'''


class EvaluationAgent(BaseAgent):
    """
    初始化子图结构
    """

    def init_workflow(self, checkpointer):
        subgraph_builder = StateGraph(EvaluationState)
        subgraph_builder.add_node("选择制图模型", self.choice_mapping_model)
        subgraph_builder.add_node("制图", self.mapping)
        subgraph_builder.add_node("制图结果确认", self.user_confirm_mapping_result)
        subgraph_builder.add_node("评估参数确认", self.user_confirm_evaluation_params)
        subgraph_builder.add_node("评估", self.evaluating)
        subgraph_builder.add_node("评估结果确认", self.user_confirm_evaluation_result)

        subgraph_builder.add_edge(START, "选择制图模型")
        subgraph_builder.add_edge("选择制图模型", "制图")
        subgraph_builder.add_edge("制图", "制图结果确认")
        subgraph_builder.add_conditional_edges(
            "制图结果确认",
            lambda prediction_state: prediction_state["choice"],
            path_map={
                "重新制图": "选择制图模型",
                "确认": "评估参数确认"
            }
        )
        subgraph_builder.add_edge("评估参数确认", "评估")
        subgraph_builder.add_edge("评估", "评估结果确认")
        subgraph_builder.add_conditional_edges(
            "评估结果确认",
            lambda prediction_state: prediction_state["choice"],
            path_map={
                "重新制图": "选择制图模型",
                "重新评估": "评估参数确认",
                "确认": END
            }
        )

        self.subgraph = subgraph_builder.compile(checkpointer=checkpointer)  # 父图会自动做好持久化，子图无需指定checkpointer

        # 可视化图结构
        save_graph_to_png(self.subgraph.get_graph(), 'evaluation')

    '''
    图节点函数：请求用户在预测制图前对必要的信息做出确认,确认如下三项信息：
    1、需要执行的算法（可以有多个）
    2、协变量栅格文件存储路径
    3、制图区域数据文件
    '''

    def choice_mapping_model(self, state: EvaluationState):
        print_agent_output(
            f"请求用户确认预测所需信息... 调用到user_confirm_before_prediction",
            agent="EVALUATION")
        self.interrput_event.set()  # 通知等待线程可以执行
        selected_algorithms, covariates_path, mapping_area_file = interrupt('')
        return {"selected_algorithms": selected_algorithms,
                "covariates_path": covariates_path,
                "mapping_area_file": mapping_area_file}

    '''
    图节点函数：生成属性图,并进行准确性评估和不确定性评估
    这是一个长时间任务，节点方法内部会将选定的若干算法依次顺序执行。
    '''

    async def mapping(self, state: EvaluationState):
        print_agent_output(f"制图... 调用到mapping", agent="EVALUATION")
        selected_algorithms = state.get("selected_algorithms")
        build_model_results = BuildModelDataAccess.get_model_metrics(state.get("task").last_build_model_id, True)
        mapping_results = []
        for algorithms_id, _ in selected_algorithms.items():  # 遍历每一个需建模的算法
            mapping_metrics = mapping(state.get("task").task_id,
                                      algorithms_id,
                                      state.get("mapping_area_file"),
                                      state.get("covariates_path"),
                                      state.get("params_for_build_model"))
            for metric in build_model_results:
                if metric.algorithms_id == algorithms_id:
                    mapping_metrics.CV_best_score = metric.CV_best_score
                    mapping_metrics.R2 = metric.R2
                    mapping_metrics.RMSE = metric.RMSE
                    break
            mapping_results.append(mapping_metrics)
        return {"mapping_results": mapping_results}  # 返回建模结果的指标

    '''
    图节点函数：对预测的结果进行展示，并请求用户做出确认，是否重新预测还是结束预测
    '''

    def user_confirm_mapping_result(self, state: EvaluationState):
        print_agent_output(
            f"请求用户确认制图结果... user_confirm_mapping_result",
            agent="EVALUATION")
        self.interrput_event.set()  # 通知等待线程可以执行
        choice = interrupt('')
        if choice == UserConfirmUIChoice.MappingAgain:
            return {"choice": "重新制图"}
        else:
            return {"choice": "确认"}

    '''
    图节点函数：请求用户确定评估参数的设置
    '''

    def user_confirm_evaluation_params(self, state: EvaluationState):
        print_agent_output(
            f"开始请求用户确定评估参数的设置... user_confirm_evaluation_params",
            agent="EVALUATION")
        self.interrput_event.set()  # 通知等待线程可以执行
        user_choice = interrupt('')
        return {"uncertainty_metrics_type": user_choice['uncertainty_metrics_type'],
                "indepent_file": user_choice['indepent_file'] if 'indepent_file' in user_choice else ''}

    '''
    图节点函数：进行准确性评估和不确定性评估
    这是一个长时间任务，节点方法内部会将选定的若干算法依次顺序执行。
    '''

    async def evaluating(self, state: EvaluationState):
        print_agent_output(f"评估... 调用到evaluating", agent="EVALUATION")
        # 不确定性分布图的计算过程：
        # 0、确定一次训练数据
        # 1、使用生成土壤属性图的模型，重新用训练数据fit
        # 2、fit后，预测制图
        # 3、上述步骤执行N次，每个像素位置则有N个值
        # 4、使用分位数计算PICP
        # uncertainty_analysis = Uncertainty()
        selected_algorithms = state.get("selected_algorithms")
        evaluating_results = []
        for algorithms_id, _ in selected_algorithms.items():  # 遍历每一个需建模的算法
            # 计算独立数据集上的指标
            rmse, r2 = compute_indepent_test_metrics(algorithms_id,
                                                     r"D:\PycharmProjects\DSM训练数据\土壤有机碳_validation.csv" if state.get("indepent_file") == '' else state.get("indepent_file"),
                                                     state.get("covariates_path"),
                                                     state.get("params_for_build_model"),
                                                     state.get("esda_result")
                                                     )
            evaluating_metrics = create_uncertainty_map(state.get("task").task_id,
                                                        algorithms_id,
                                                        state.get("task").sample_file,
                                                        state.get("mapping_area_file"),
                                                        state.get("covariates_path"),
                                                        state.get("uncertainty_metrics_type"),
                                                        state.get("params_for_build_model"),
                                                        state.get("esda_result"))
            evaluating_metrics.R2 = r2
            evaluating_metrics.RMSE = rmse
            evaluating_results.append(evaluating_metrics)
        return {"evaluating_results": evaluating_results}  # 返回建模结果的指标

    '''
    图节点函数：对预测的结果进行展示，并请求用户做出确认，是否重新预测还是结束预测
    '''

    def user_confirm_evaluation_result(self, state: EvaluationState):
        print_agent_output(
            f"请求用户确认评估结果... user_confirm_evaluation_result",
            agent="EVALUATION")
        self.interrput_event.set()  # 通知等待线程可以执行
        choice = interrupt('')
        if choice == UserConfirmUIChoice.MappingAgain:
            return {"choice": "重新制图"}
        elif choice == UserConfirmUIChoice.EvaluatingAgain:
            return {"choice": "重新评估"}
        else:
            self.quit_chat = True
            return {"choice": "确认"}

    '''
    异步的agent主控函数：定义一个预测的子图。
    '''

    async def async_run(self, global_state: TaskState) -> TaskState:
        print_agent_output(f"开始预测...", agent="EVALUATION")
        async with AsyncSqliteSaver.from_conn_string('./prediction.db') as checkpointer:
            # 初始化子图
            self.init_workflow(checkpointer)

            # 启动子图的执行，进入accept_input节点
            thread_config = {"configurable": {"thread_id": global_state["task"].task_id}}

            if global_state["task"].stage != TaskStage.Evaluation:  # 如果当前任务不处于预测的状态，直接跳出
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
                                                   "esda_result": global_state["esda_result"]},
                                            config=thread_config)
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
                if state.values.get("chat_messages") is not None and len(state.values['chat_messages']) > 0 and \
                        type(state.values['chat_messages'][-1]) is HumanMessage:  # 已经有人类反馈，在恢复模式下
                    await self.subgraph.ainvoke(Command(resume=state.values['task'].sample_file),
                                                config=thread_config)
                else:
                    # 请求用户输入
                    if state.values.get("chat_messages") is not None and len(state.values['chat_messages']) > 0 and \
                            type(state.values['chat_messages'][-1]) is AIMessage:  # Agent给出的消息尚未确定数据准备就绪，则需要继续向人类反馈
                        # 附带了LLM反馈
                        suggestion = state.values['suggestion']

                    # 根据获取的最后的持久化记录进行相应的处理
                    input_finished_event = self.agent_signals.create_input_finished_event()  # 创建一个python的事件，用于阻塞线程，直到获取用户的反馈
                    input_finished_event.clear()  # 重置信号，以便用户给出反馈

                    # 准备用户确认的数据，然后通知到UI层
                    if state.next[0] == "选择制图模型":
                        build_model_results = BuildModelDataAccess.get_model_metrics(
                            global_state["task"].last_build_model_id,
                            True)
                        self.agent_signals.user_choice_mapping_model.emit(self.agent_signals.task_id,
                                                                          input_finished_event,
                                                                          global_state["task"],
                                                                          build_model_results)  # 请求用户指定预测所需信息
                    elif state.next[0] == "制图结果确认":
                        self.agent_signals.user_confirm_mapping_result.emit(self.agent_signals.task_id,
                                                                            input_finished_event,
                                                                            global_state["task"],
                                                                            state.values[
                                                                                "mapping_results"])  # 请求用户确定制图结果
                    elif state.next[0] == "评估参数确认":
                        self.agent_signals.user_confirm_evaluating_params.emit(self.agent_signals.task_id,
                                                                               input_finished_event)  # 请求用户指定预测所需信息
                    elif state.next[0] == "评估结果确认":
                        self.agent_signals.user_confirm_evaluating_result.emit(self.agent_signals.task_id,
                                                                               input_finished_event,
                                                                               state.values[
                                                                                   "uncertainty_metrics_type"],
                                                                               state.values[
                                                                                   "evaluating_results"])  # 请求用户确定制图结果

                    # 阻塞，直到输入结束事件被触发(获取到了UI层的用户的反馈)
                    input_finished_event.wait()

                    # 3、恢复interrupt的执行
                    await self.subgraph.ainvoke(Command(resume=input_finished_event.data), config=thread_config)

            # 将最后一次进行预测制图的结果写入数据库
            last_state = await self.subgraph.aget_state(thread_config)  # 获取建模的最后状态参数
            MappingDataAccess.record_mapping_metrics(last_state.values["mapping_results"])

            # 将任务阶段信息写入数据库，指示进入建模阶段
            TaskDataAccess.update_task_stage(self.agent_signals.task_id, TaskStage.Finished)
            # 更新全局的State状态
            global_state["task"].stage = TaskStage.Finished  # 指示任务结束
            self.show_ui_progress("结束本次DSM任务")
            return {"task": global_state["task"]}
