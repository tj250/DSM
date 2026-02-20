import time
from langchain_core.messages import (
    AIMessage,
    HumanMessage
)
from langgraph.graph import START, StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agents.llm.llm_functions import call_llm_with_pydantic_output
from data_access.task import TaskDataAccess
from agents.data_structure.task_state import TaskState
from agents.data_structure.define_task_state import DefineTaskState
from agents.data_structure.define_task_pydantic import TaskAnalyzingResult, EvaluatePropertyResult, EvaluateEnvironmentBGResult
from .utils.views import print_agent_output, TaskStage, TaskAnalysisConfirmUIChoice
from .base_agent import BaseAgent
from agents.utils.visualization import save_graph_to_png

'''
理解用户挖掘意图的智能体。
，一步一步诱导用户给出挖掘任务。需要确定的方面包括
1、挖掘的任务类型：分类/回归/异常检测等等
2、挖掘数据的构成
3、是否需要建模，或者是端对端挖掘
4、面向（专业型用户）是否需要进一步提供模型，无论是建模，或者是端对端挖掘
'''


class DefineTaskAgent(BaseAgent):

    '''
    初始化子图结构
    '''

    def init_workflow(self, checkpointer):
        subgraph_builder = StateGraph(DefineTaskState)
        subgraph_builder.add_node("描述任务", self.acquire_task)
        subgraph_builder.add_node("理解任务", self.analyze_task)
        subgraph_builder.add_node("评估任务", self.evaluate_task)
        subgraph_builder.add_node("用户确认", self.user_confirm_task)

        subgraph_builder.add_edge(START, "描述任务")
        subgraph_builder.add_edge("描述任务", "理解任务")
        subgraph_builder.add_edge("理解任务", "评估任务")
        subgraph_builder.add_conditional_edges(
            "评估任务",
            lambda define_task_state: "任务澄清" if define_task_state.get(
                                                        'soil_property') is None else "分析结果确认",
            {"任务澄清": "描述任务", "分析结果确认": "用户确认"}
        )
        subgraph_builder.add_conditional_edges(
            "用户确认",
            lambda define_task_state: "任务澄清" if define_task_state.get(
            'user_choice') == TaskAnalysisConfirmUIChoice.AnalyzingAgain else "分析结束",
            {"任务澄清": "描述任务", "分析结束": END}
        )
        self.subgraph = subgraph_builder.compile(checkpointer=checkpointer)  # 父图会自动做好持久化，子图无需指定checkpointer
        # 可视化图结构
        save_graph_to_png(self.subgraph.get_graph(), 'accept_task')

    '''
    请求用户给出任务描述或进一步澄清
    '''

    def acquire_task(self, state: DefineTaskState):
        print_agent_output(f"开始请求用户给出任务描述或进一步澄清... acquire_task", agent="ACCEPTTASK")
        self.interrput_event.set()  # 通知等待线程可以执行
        feedback = interrupt('')
        if state.get("chat_messages") is None:  # 如果是首次输入，则根据提示模板生成任务理解提示
            task_prompt = """你需要针对如下有关土壤属性制图任务的描述进行评估，以确认其是否为一个有效的土壤属性制图任务。任务描述为：
            {}          
            如果你认为给定的描述是一个有效的土壤属性制图任务，则分析成功，此时需要确定要对哪种土壤属性进行制图，例如土壤pH或者土壤有机碳。
            同时，你还需要对该描述进行较为简洁的优化改写，用于反馈给用户确认。

            如果你认为给定的描述不是有效的土壤属性制图任务，则代表分析失败，需给出有关改进任务描述的建议。
            """
            messages = [HumanMessage(f"{task_prompt}".format(feedback))]  # 第一个用户消息
        else:  # 已经是后续对话轮次
            task_prompt = """用户补充了如下信息：
            {}
            请根据补充的上述信息再次对土壤属性制图任务的描述进行评估，并输出评估结果。
            """
            new_prompt = f"{task_prompt}".format(feedback)
            messages = state["chat_messages"]
            if state["chat_messages"][-1].content != new_prompt:
                messages = state["chat_messages"] + [HumanMessage(new_prompt)]  # 新增的用户消息
        # We return a list, because this will get added to the existing list
        return {"task_desc": feedback, "chat_messages": messages}

    '''
    基于用户给出的描述，分析出待预测的目标土壤属性以及建模区域的地理环境背景信息(用于后续的环境协变量选择)
    '''
    def analyze_task(self, state: DefineTaskState):
        print_agent_output(f"开始请求LLM分析用户给出的任务... analyze_task", agent="ACCEPTTASK")
        # 过滤消息，以免超过模型的最大token数量
        if len(state["chat_messages"]) > 6:
            reduced_messages = state["chat_messages"][:2] + state["chat_messages"][-3:]  # 前两条+最后三条
        else:
            reduced_messages = state["chat_messages"]
        # 调用模型
        self.show_ui_progress("正在从用户给出的描述中分析任务信息...")
        response = call_llm_with_pydantic_output(reduced_messages, TaskAnalyzingResult)
        # 解析结果
        if response is not None: # llm有响应
            if response['parsing_error'] == None:
                if response['parsed'].isvalid:  # 已确定土壤属性制图任务类型
                    last_response = {"chat_messages": state["chat_messages"] + [response['raw']],
                                     "region_geo_environment":response['parsed'].region_geo_environment,
                                     "soil_property": response['parsed'].soil_property,
                                     "task_summary": response['parsed'].summary}
                    # self.interrput_event.set()  # 通知等待线程可以执行
                    return last_response
                else:  # 不确定土壤属性制图任务类型，需要继续请求人类澄清
                    return {"chat_messages": state["chat_messages"] + [response['raw']],
                            # 向消息列表中追加AIMessage，仅将LLM给出的修改任务描述的文本加入
                            "soil_property": None,
                            "suggestion": response['parsed'].suggestion}
            else:  # LLM的输出不正确
                return {"chat_messages": state["chat_messages"] + [response['raw']],
                        "soil_property": None,
                        "suggestion": "无法确定你给出的描述是否属于土壤属性制图任务，需要你给出更清楚的描述。"}
        else: # 无法调用到llm
            print_agent_output(f"LLM异常，请检查网络连接或LLM服务是否可用",
                               agent="ACCEPTTASK")
            return {}


        return {}

    '''
    评估LLM给出的分析结果。评估通过的标志是设置property为某一具体的土壤属性值
    '''

    def evaluate_task(self, state: DefineTaskState):
        print_agent_output(f"开始评估LLM生成的任务分析结果... evaluate_task", agent="ACCEPTTASK")
        if state.get("soil_property") is None:  # 分析自身就失败了，直接请求用户澄清
            # self.interrput_event.set()  # 通知等待线程可以执行
            return {"soil_property": None}
        # 调用模型
        self.show_ui_progress("正在评估用户给出的描述是否为有效的土壤属性制图任务...")
        message = HumanMessage(f"本次土壤属性制图任务要预测的土壤属性为：{state['soil_property']}")
        response = call_llm_with_pydantic_output([message], EvaluatePropertyResult)      # 评估土壤属性是否分析正确
        # 解析结果
        if response is not None: # llm有响应
            if response['parsing_error'] == None:
                if response['parsed'].isvalid:  # 已确定土壤属性制图任务的预测属性有效
                    # 进一步分析研究区域环境背景信息描述的有效性
                    message = HumanMessage(f"请确认如下描述信息是否为针对某个研究区域的地理环境背景信息的描述：{state['region_geo_environment']}")
                    response2 = call_llm_with_pydantic_output([message], EvaluateEnvironmentBGResult)  # 评估土壤属性是否分析正确
                    if response2 is not None:  # llm有响应
                        if response2['parsing_error'] == None:
                            if response2['parsed'].isvalid:  # 已确定土壤属性制图任务的预测属性有效
                                self.interrput_event.set()  # 通知等待线程可以执行
                                return {}
                            else:  # 地理环境背景信息描述无效，需要继续请求人类澄清
                                return {"soil_property": None,
                                        "suggestion": response2['parsed'].suggestion}  # 附带LLM的反馈
                        else:  # LLM的输出不正确
                            return {"soil_property": None,
                                    "suggestion": "LLM无法正确解析出土壤属性制图研究区域的地理环境背景信息，需要你给出更清楚的描述或更换模型。"}
                    else:  # 无法调用到llm
                        print_agent_output(f"LLM异常，请检查网络连接或LLM服务是否可用",
                                           agent="ACCEPTTASK")
                        return {}
                else:  # 无法评估分析结果（待预测土壤属性），需要继续请求人类澄清
                    return {"soil_property": None,
                            "suggestion": response['parsed'].suggestion}
            else:  # LLM的输出不正确
                return {"soil_property": None,
                        "suggestion": "LLM无法正确解析出有效的待预测土壤属性，需要你给出更清楚的描述或更换模型。"}
        else: # 无法调用到llm
            print_agent_output(f"LLM异常，请检查网络连接或LLM服务是否可用",
                               agent="ACCEPTTASK")
            return {}

    '''
    图节点函数：请求用户对任务分析结果做出最终确认
    '''

    def user_confirm_task(self, state: DefineTaskState):
        print_agent_output(f"开始请求用户对任务分析结果做出最终确认... 调用到了user_confirm_task", agent="ACCEPTTASK")
        self.interrput_event.set()  # 通知等待线程可以执行
        choice = interrupt('')
        if choice == TaskAnalysisConfirmUIChoice.EnterNextStep:
            self.quit_chat = True  # 指示结束子图
        return {"user_choice": choice}


    '''
    定义一个接收任务的子图，完成对土壤属性制图任务的意图分析
    采用Evaluator-Optimizer设计模式，在用户给出任务信息时，进行评估，经过多轮迭代，最终获取用户的土壤属性制图任务意图
    '''

    async def async_run(self, global_state: TaskState):
        print_agent_output(f"开始确定土壤属性制图任务...", agent="ACCEPTTASK")
        # 初始化子图结构
        async with AsyncSqliteSaver.from_conn_string('./define_task.db') as checkpointer:
            # 初始化子图
            self.init_workflow(checkpointer)

            thread_config = {"configurable": {"thread_id": global_state["task"].task_id}}

            # 检测当前节点是否已经执行完毕，是则跳过，同时获取历史状态，用于全局状态的恢复
            if global_state["task"].stage != TaskStage.AcceptTask:  # 如果当前任务不处于确定任务总体信息的状态，则短路跳出
                state = await self.subgraph.aget_state(thread_config)
                global_state["task"].soil_property = state.values["soil_property"]
                global_state["task"].summary =  state.values["task_summary"]
                return {"task": global_state["task"]}

            # 启动子图的执行，进入accept_input节点
            # 获取子图的历史状态快照
            history_states = [state async for state in self.subgraph.aget_state_history(thread_config)]  # 获取所有的check points
            if len(history_states) > 0:  # 存在历史的check points,处于恢复模式（从长任务中恢复历史上的执行状态）
                if len(history_states[0].next) == 0:
                    print("checkpoint异常，可能是从调试中恢复！！！")
                    self.interrput_event.set()
                    self.quit_chat = True
                else:
                    await self.subgraph.ainvoke(None, history_states[0].config)
            else:# 否则，初次执行
                await self.subgraph.ainvoke(input={}, config=thread_config)

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
                state = await self.subgraph.aget_state(thread_config)
                if state.values.get("chat_messages") is not None and len(state.values['chat_messages']) > 0 and \
                        type(state.values['chat_messages'][-1]) is HumanMessage:  # 已经有人类反馈，在恢复模式下
                    await self.subgraph.ainvoke(Command(resume=state.values['task_desc']), config=thread_config)
                else:
                    # 请求用户输入
                    if state.values.get("chat_messages") is not None and len(state.values['chat_messages']) > 0 and \
                            type(state.values['chat_messages'][-1]) is AIMessage and \
                            state.values['soil_property'] is None:  # LLM给出的最后一条消息尚未确定任务类型，则需要继续向人类反馈
                        # 附带了LLM反馈
                        suggestion = state.values['suggestion']

                    input_finished_event = self.agent_signals.create_input_finished_event()  # 创建一个python的事件，用于阻塞线程，直到获取用户的反馈
                    input_finished_event.clear()  # 重置信号，以便用户给出反馈
                    if len(state.values) == 0:  # 首次来到这里，即：处于获取数据源的节点
                        self.agent_signals.wait_input_task_desc.emit(self.agent_signals.task_id, input_finished_event,
                                                                     global_state["task"].description,
                                                                     suggestion)  # 请求用户指定数据源
                    else:  # 已经是非首次获取数据源的情况
                        if state.next is not None:  # 已经有明确的下一步
                            if state.next[0] == "描述任务":  # 请求用户对任务进行描述或澄清
                                self.agent_signals.wait_input_task_desc.emit(self.agent_signals.task_id,
                                                                             input_finished_event,
                                                                             state.values['task_desc'],
                                                                             suggestion)
                            elif state.next[0] == "用户确认":  # 请求用户确认任务分析的最终结果
                                confirm_info_list = {
                                    "待制图的土壤属性":state.values["soil_property"],
                                    "任务概述":state.values["task_summary"]}
                                self.agent_signals.user_confirm_task_analysis_result.emit(self.agent_signals.task_id,
                                                                                          input_finished_event,
                                                                                          confirm_info_list)  # 请求用户确认分析结果
                    input_finished_event.wait()  # 阻塞，直到输入结束事件被触发(获取到了用户的反馈)

                    # 3、恢复interrupt的执行
                    await self.subgraph.ainvoke(Command(resume=input_finished_event.data), config=thread_config)

            # transform response back to the parent state
            last_state = await self.subgraph.aget_state(thread_config)
            latest_state_values = last_state.values
            self.agent_signals.task_analysis_finished.emit(self.agent_signals.task_id, latest_state_values["soil_property"],
                                                           latest_state_values["task_summary"])
            # 将土壤属性制图任务已确定信息写入数据库，包括指示任务进入DataExplore阶段
            TaskDataAccess.confirm_task_basic_info(self.agent_signals.task_id, latest_state_values["soil_property"],
                                                   latest_state_values["task_summary"])
            # 更新State状态
            global_state["task"].soil_property = latest_state_values["soil_property"]
            global_state["task"].summary = latest_state_values["task_summary"] # LLM总结出来的反馈
            global_state["task"].description = latest_state_values["task_desc"] # 用户输入的任务描述
            global_state["task"].stage = TaskStage.DataExplore
            self.show_ui_progress("即将进入数据探索阶段...")
            return {"task": global_state["task"]}
