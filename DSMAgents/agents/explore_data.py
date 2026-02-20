import os.path
import time
import numpy as np
import pandas as pd
from decimal import Decimal
from collections import Counter
from natsort import natsorted

import config
from data_access.structure_data_dealer import StructureDataDealer
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from .utils.views import DEUIChoice, DEUIType, TaskStage, print_agent_output
from .base_agent import BaseAgent
from eda.analysis import parse_categorical_variables
from eda.analysis import execute_analysis
from agents.llm.llm_functions import call_llm_with_pydantic_output, get_llm
from agents.data_structure.explore_data_pydantic import PredictionVariable, CategoricalVariables
from agents.data_structure.explore_data_state import RegressionParams, ExploreDataState
from agents.data_structure.task_state import TaskState
from data_access.task import TaskDataAccess
from eda.regression import analyze_features_importance
from agents.utils.visualization import save_graph_to_png
from data_access.build_model import BuildModelDataAccess

'''
数据探索智能体，通过人机交互方式，分析用于建模的数据信息。
需要处理的业务：
1、根据任务类型来校验数据来源是否正确
2、探索数据，以确定所提供的结构化数据中的解释变量，类别型变量以及预测变量
'''


class ExploreDataAgent(BaseAgent):
    '''
    根据提供的数据文件，解析出数据文件的schema，包括两个部分:一个是浮点型列的集合，另一个是整数型列的集合
    '''

    def parse_data_schema(self, data_source: str) -> tuple[list[str], list[str]]:
        schema = StructureDataDealer.get_structed_data_schema(data_source, 1)  # 获取结构化数据的schema
        float_columns = [col for col, dtype in schema.items() if np.issubdtype(dtype, np.floating) or dtype is Decimal]
        integer_columns = [col for col, dtype in schema.items() if np.issubdtype(dtype, np.integer)]
        return float_columns, integer_columns

    '''
    初始化子图结构
    '''

    def init_workflow(self, checkpointer):
        subgraph_builder = StateGraph(ExploreDataState)
        subgraph_builder.add_node("用户设置数据源", self.user_set_datasource)
        subgraph_builder.add_node("变量分析", self.analyze_variables)
        subgraph_builder.add_node("选择确认响应变量的方式", self.user_choice_prediction_variable)
        subgraph_builder.add_node("选择确认解释变量的方式", self.user_choice_interpretation_variables)
        subgraph_builder.add_node("选择确认类别型变量的方式", self.user_choice_categorical_variables)
        subgraph_builder.add_node("用户指定响应变量", self.user_set_prediction_variable)
        subgraph_builder.add_node("用户指定解释变量", self.user_set_interpretation_variable)
        subgraph_builder.add_node("用户指定类别型变量", self.user_set_categorical_variables)
        subgraph_builder.add_node("用户确认变量分析结果", self.user_confirm_result)
        subgraph_builder.add_node("探索性空间数据分析", self.esda)

        subgraph_builder.add_edge(START, "用户设置数据源")
        subgraph_builder.add_edge("用户设置数据源", "变量分析")
        subgraph_builder.add_conditional_edges(
            "变量分析",  # 源节点
            lambda data_explorer_state: data_explorer_state["next_node"],
            path_map={
                "需更换数据源": "用户设置数据源",  # 跳转到工具调用节点
                "需指定响应变量": "选择确认响应变量的方式",  # 跳转到用户选择节点
                "需更正解释变量": "选择确认解释变量的方式",  # 跳转到用户选择节点
                "需更正类别型变量": "选择确认类别型变量的方式",  # 跳转到用户选择节点
                "确认变量分析结果": "用户确认变量分析结果"  # 跳转到用户最终确认节点
            }
        )
        # 用户做出选择后的路由，要么更换数据源，要么直接指定响应变量
        subgraph_builder.add_conditional_edges(
            "选择确认响应变量的方式",
            lambda data_explorer_state: "直接指定" if data_explorer_state[
                                                                  'user_choice'] == DEUIChoice.CorrectPredictionVariable else "更换数据源",
            {"更换数据源": "用户设置数据源", "直接指定": "用户指定响应变量"}
        )
        subgraph_builder.add_conditional_edges(
            "选择确认解释变量的方式",
            lambda data_explorer_state: "直接指定" if data_explorer_state[
                                                                  'user_choice'] == DEUIChoice.CorrectPredictionVariable else "更换数据源",
            {"更换数据源": "用户设置数据源", "直接指定": "用户指定解释变量"}
        )
        subgraph_builder.add_conditional_edges(
            "选择确认类别型变量的方式",
            lambda data_explorer_state: "直接指定" if data_explorer_state[
                                                                'user_choice'] == DEUIChoice.CorrectCategoticalVariables else "更换数据源",
            {"更换数据源": "用户设置数据源", "直接指定": "用户指定类别型变量"}
        )
        subgraph_builder.add_edge("用户指定响应变量", "用户确认变量分析结果")
        subgraph_builder.add_edge("用户指定解释变量", "用户确认变量分析结果")
        subgraph_builder.add_edge("用户指定类别型变量", "用户确认变量分析结果")
        # 用户最终复核数据集分析的结果，要么进入下一个环节（开始探索性数据分析），要么执行重新指定数据源等其他类型操作
        subgraph_builder.add_conditional_edges(
            "用户确认变量分析结果",
            lambda data_explorer_state: data_explorer_state["user_choice"].value,
            path_map={
                DEUIChoice.ChangeDataSource.value: "用户设置数据源",  # 跳转到工具调用节点
                DEUIChoice.CorrectPredictionVariable.value: "用户指定响应变量",  # 跳转到用户选择节点
                DEUIChoice.CorrectInterpretationVariable.value: "用户指定解释变量",  # 跳转到用户选择节点
                DEUIChoice.CorrectCategoticalVariables.value: "用户指定类别型变量",  # 跳转到用户选择节点
                DEUIChoice.EnterEsda.value: "探索性空间数据分析"  # 跳转到探索性空间数据分析节点
            }
        )
        subgraph_builder.add_edge("探索性空间数据分析", END)  # 用户确认信息后，进行探索性数据分析
        self.subgraph = subgraph_builder.compile(checkpointer=checkpointer)  # 父图会自动做好持久化，子图无需指定checkpointer

        # 可视化图结构
        save_graph_to_png(self.subgraph.get_graph(), 'data_explorer')

    '''
    图节点函数：请求用户指定数据源
    '''

    def user_set_datasource(self, state: ExploreDataState):
        print_agent_output(
            f"开始请求用户指定数据源... user_set_datasource",
            agent="DATAEXPLORER")
        self.interrput_event.set()  # 通知等待线程可以执行
        data_source = interrupt('')
        task = state.get("task")
        task.sample_file = data_source
        return {"task": task}

    '''
    图节点函数：请求用户做出最终确认
    '''

    def user_confirm_result(self, state: ExploreDataState):
        print_agent_output(
            f"开始请求用户做出最终确认... user_confirm_result",
            agent="DATAEXPLORER")
        self.interrput_event.set()  # 通知等待线程可以执行
        choice = interrupt('')
        if choice == DEUIChoice.EnterEsda:
            self.quit_chat = True  # 指示准备结束子图
        return {"user_choice": choice}

    '''
    执行探索性恐空间数据分析
    '''

    async def esda(self, state: ExploreDataState):
        print_agent_output(
            f"开始执行探索性数据分析... esda",
            agent="DATAEXPLORER")
        # 执行探索性数据分析
        esda_rsult = execute_analysis(state["task"].sample_file, state["regression_params"].prediction_variable,
                                      state["regression_params"].continuous_variables,
                                      state["regression_params"].categorical_variables)
        return {"esda_result": esda_rsult}

    '''
    图节点函数：请求用户做出做出是否重新指定数据源，还是直接指定预测变量列的选择。
    '''

    def user_choice_prediction_variable(self, state: ExploreDataState):
        print_agent_output(
            f"开始请求用户做出做出是否重新指定数据源，还是直接指定预测变量列的选择。... user_choice_prediction_variable",
            agent="DATAEXPLORER")
        self.interrput_event.set()  # 通知等待线程可以执行
        choice = interrupt('')
        return {"user_choice": choice}

    '''
    图节点函数：请求用户做出做出是否重新指定数据源，还是直接指定解释变量的选择。
    '''

    def user_choice_interpretation_variables(self, state: ExploreDataState):
        print_agent_output(
            f"开始请求用户做出做出是否重新指定数据源，还是直接指定解释变量的选择。... user_choice_interpretation_variables",
            agent="DATAEXPLORER")
        self.interrput_event.set()  # 通知等待线程可以执行
        choice = interrupt('')
        return {"user_choice": choice}

    '''
    图节点函数：请求用户做出做出是否重新指定数据源，还是直接指定类别型变量。
    '''

    def user_choice_categorical_variables(self, state: ExploreDataState):
        print_agent_output(
            f"开始请求用户做出做出是否重新指定数据源，还是直接指定类别型变量。... user_choice_categorical_variables",
            agent="DATAEXPLORER")
        self.interrput_event.set()  # 通知等待线程可以执行
        choice = interrupt('')
        return {"user_choice": choice}

    '''
    图节点函数：用户指定数据源中的预测列(预测变量)
    '''

    def user_set_prediction_variable(self, state: ExploreDataState):
        print_agent_output(f"开始请求用户指定数据源中的预测列(响应变量)... user_set_prediction_variable",
                           agent="DATAEXPLORER")
        self.interrput_event.set()  # 通知等待线程可以执行
        prediction_variable = interrupt('')

        # 如果用户更改了预测变量,则涉及重新调整解释变量和类别变量，并重新计算解释变量的特征重要性
        if prediction_variable != state["regression_params"].prediction_variable:
            # 1-确定解释变量
            schema = StructureDataDealer.get_structed_data_schema(state["task"].sample_file, 1)  # 获取结构化数据的schema
            del schema[prediction_variable]
            # 删除结构化数据中的坐标值列
            if config.CSV_GEOM_COL_X in schema:
                del schema[config.CSV_GEOM_COL_X]
            if config.CSV_GEOM_COL_Y in schema:
                del schema[config.CSV_GEOM_COL_Y]
            regression_params = RegressionParams()
            regression_params.prediction_variable = prediction_variable  # 用户指定的预测变量
            regression_params.interpretation_variables = [varname for varname in schema.keys()]  # 确定解释变量

            # 2-确定类别型变量
            self.show_ui_progress("从解释变量中解析类别型变量...")
            successful, categorical_variables_list = parse_categorical_variables(
                regression_params.interpretation_variables)
            if successful:  # 解析成功
                regression_params.categorical_variables = categorical_variables_list
                if len(categorical_variables_list) > 0:
                    regression_params.categorical_vars_detail = self.analyzing_categorical_variables(
                        state["task"].sample_file, categorical_variables_list)  # 重新对类别变量进行分析

                # 3-确定连续型变量:连续型变量=解释变量-类别型变量
                continuous_variables_list = []
                for var_name in schema:
                    if var_name not in categorical_variables_list:
                        continuous_variables_list.append(var_name)
                regression_params.continuous_variables = continuous_variables_list

                # 4-重新计算解释变量的特征重要性
                regression_params.features_importance = analyze_features_importance(state["task"].sample_file,
                                                                                    regression_params.prediction_variable,
                                                                                    regression_params.interpretation_variables,
                                                                                    categorical_variables_list)
                return {"regression_params": regression_params}  # 全新生成
            else:
                return {}
        else:  # 用户并未改变预测变量
            return {}

    '''
    图节点函数：用户指定数据源中的解释变量
    解释变量必须为连续型的数值列，或者为列表型的列
    '''

    def user_set_interpretation_variable(self, state: ExploreDataState):
        print_agent_output(f"开始请求用户指定数据源中的解释变量... user_set_interpretation_variable",
                           agent="DATAEXPLORER")
        self.interrput_event.set()  # 通知等待线程可以执行
        interpretation_variables_list = interrupt('')

        regression_params = state["regression_params"]
        # 如果用户更改了解释变量,则涉及重新调整解释变量和类别变量，并重新计算解释变量的特征重要性
        if not Counter(interpretation_variables_list) == Counter(regression_params.interpretation_variables):
            regression_params.interpretation_variables = interpretation_variables_list  # 确定解释变量
            # 1-确定类别型变量
            self.show_ui_progress("从解释变量中解析类别型变量...")
            successful, categorical_variables_list = parse_categorical_variables(interpretation_variables_list)
            if successful:  # 解析成功
                regression_params.categorical_variables = categorical_variables_list
                if len(categorical_variables_list) > 0:  # 存在类别型变量
                    regression_params.categorical_vars_detail = self.analyzing_categorical_variables(
                        state["task"].sample_file, categorical_variables_list)  # 重新对类别变量进行分析
                    # 2-确定连续型变量:连续型变量=解释变量-类别型变量
                    continuous_variables_list = []
                    for var_name in interpretation_variables_list:
                        if var_name not in categorical_variables_list:
                            continuous_variables_list.append(var_name)
                    regression_params.continuous_variables = continuous_variables_list

                    # 3-重新计算解释变量的特征重要性
                    regression_params.features_importance = analyze_features_importance(state["task"].sample_file,
                                                                                        regression_params.prediction_variable,
                                                                                        interpretation_variables_list,
                                                                                        categorical_variables_list)
                    return {"regression_params": regression_params}  # 全新生成
            else:  # 未能成功解析
                return {}
        else:  # 用户并未改变解释变量的设置
            return {}

    '''
    图节点函数：用户指定解释变量中哪些属于类别型变量
    '''

    def user_set_categorical_variables(self, state: ExploreDataState):
        print_agent_output(f"开始请求用户指定解释变量中哪些属于类别型变量... user_set_categorical_variables",
                           agent="DATAEXPLORER")
        self.interrput_event.set()  # 通知等待线程可以执行
        categorical_variables_list = interrupt('')
        state["regression_params"].categorical_variables = categorical_variables_list  # inplace修改
        state["regression_params"].categorical_vars_detail = self.analyzing_categorical_variables(
            state["task"].sample_file, categorical_variables_list)  # 重新对类别变量进行分析

        # 1-确定连续型变量:连续型变量=解释变量-类别型变量
        continuous_variables_list = []
        for var_name in state["regression_params"].interpretation_variables:
            if var_name not in categorical_variables_list:
                continuous_variables_list.append(var_name)
        state["regression_params"].continuous_variables = continuous_variables_list

        # 2-重新计算解释变量的特征重要性
        state["regression_params"].features_importance = analyze_features_importance(state["task"].sample_file,
                                                                                     state[
                                                                                         "regression_params"].prediction_variable,
                                                                                     state[
                                                                                         "regression_params"].interpretation_variables,
                                                                                     categorical_variables_list)
        return {"regression_params": state["regression_params"]}

    '''
    图节点函数：从数据源装载数据,并确认是否可以获取到预测变量列
    数据探索后的路由，存在三种情况：
    1、数据源明显存在问题，直接路由至acquire_datasource
    2、数据源中字段命名无法被LLM确定为预测变量，进入用户选择阶段（由用户指定要么更换数据源，要么直接指定预测变量）
    3、从数据源中可确定预测变量，路由至用户最终确认节点    
    '''

    async def analyze_variables(self, state: ExploreDataState):
        print_agent_output(f"开始确定数据中的解释变量和预测变量... 调用到了explore_data", agent="DATAEXPLORER")
        float_columns, integer_columns = self.parse_data_schema(state["task"].sample_file)
        if float_columns is not None:
            # 将坐标变量排除
            if config.CSV_GEOM_COL_X in float_columns:
                float_columns.remove(config.CSV_GEOM_COL_X)
            if config.CSV_GEOM_COL_Y in float_columns:
                float_columns.remove(config.CSV_GEOM_COL_Y)
        # 获取结构化数据表中的所有列
        if float_columns is None or len(float_columns) == 0:
            suggestion = "无法从指定的数据中获取解释变量和预测变量信息，请重新指定数据源。"
            return {"next_node": '选择更换数据源', "suggestion": suggestion}  # 需要用户再次指定数据源
        elif len(float_columns) < 3:
            suggestion = "指定的数据中数据列不足，无法用于回归预测，回归预测至少需要两个解释变量和一个预测变量，请重新指定数据源。"
            return {"next_node": '选择更换数据源', "suggestion": suggestion}  # 需要用户再次指定数据源

        # 提取结构化数据中所有的浮点值类型的列
        if len(float_columns) == 0:
            suggestion = "指定的数据中至少需要有一个浮点值类型的列作为预测变量，请重新指定数据源。"
            return {"next_node": '选择更换数据源', "suggestion": suggestion}  # 需要用户再次指定数据源

        regression_params = RegressionParams()
        # --------------------------------第一步：调用LLM分析预测变量--------------------------------
        self.show_ui_progress("正在从用户提供的数据中解析预测变量...")
        prediction_variable = await self.analyze_prediction_var(state["task"].sample_file, state["task"].summary,
                                                                state["task"].soil_property, float_columns)
        if prediction_variable is not None:  # 有分析结果
            if prediction_variable.successful:  # LLM分析成功
                if prediction_variable.name in float_columns:  # 给出的预测变量名称是有效的
                    regression_params.prediction_variable = prediction_variable.name  # 预测变量
                else:  # LLM给出的变量名称错误，因为它不在浮点类型的数据列中存在
                    return {"next_node": '需指定响应变量',
                            "regression_params": regression_params,
                            "suggestion": "无法从数据源中正确解析出预测变量，需要你更换数据源，或者直接指定预测变量"}  # 需要用户进一步直接指定解释变量
            else:  # LLM分析失败
                return {"next_node": '需指定响应变量',
                        "regression_params": regression_params,
                        "suggestion": "无法从数据源中解析出预测变量，需要你更换数据源，或者直接指定预测变量"}  # 需要用户进一步直接指定解释变量
        else:  # LLM调用失败
            return {"next_node": '需指定响应变量',
                    "regression_params": regression_params,
                    "suggestion": "从数据源解析预测变量出错"}  # 需要用户进一步直接指定解释变量

        # --------------------------------第二步：调用LLM分析类别型变量--------------------------------
        # 提取结构化数据中所有的整数值类型的列(包括bool类型的列)
        if len(integer_columns) > 0:  # 针对类别型的列进行处理
            self.show_ui_progress("正在从用户提供的数据中解析类别型变量...")
            categorical_variables = await self.analyze_categoritical_vars(state["task"].sample_file,
                                                                          state["task"].summary,
                                                                          integer_columns)
            if categorical_variables is not None:  # 有分析结果
                if categorical_variables.successful:  # LLM分析成功
                    if (categorical_variables.categorical_variables is not None and
                            set(categorical_variables.categorical_variables).issubset(
                                integer_columns)):  # 所有的类别型变量均在整数列中出现
                        pass
                    else:  # LLM给出的变量名称错误，因为它不在浮点类型的数据列中存在
                        return {"next_node": '需更正类别型变量',
                                "regression_params": regression_params,
                                "suggestion": "无法从数据源中正确解析出类别型变量，需要你更换数据源，或者直接指定类别型变量"}  # 需要用户进一步直接指定解释变量
                else:  # LLM分析失败
                    return {"next_node": '需更正类别型变量',
                            "regression_params": regression_params,
                            "suggestion": "无法从数据源中解析出类别型变量，需要你更换数据源，或者直接指定类别型变量"}  # 需要用户进一步直接指定解释变量
            else:  # LLM调用失败
                return {"next_node": '需更正类别型变量',
                        "regression_params": regression_params,
                        "suggestion": "从数据源解析类别型变量出错"}  # 需要用户进一步直接指定解释变量
        else:  # 没有候选的类别型列，则创建一个空的类别变量列表pydantic类
            categorical_variables = CategoricalVariables(successful=False, categorical_variables=[])

        regression_params.categorical_variables = categorical_variables.categorical_variables  # 类别型变量
        # --------------------------------第三步：确定所有的解释变量--------------------------------
        float_columns.remove(regression_params.prediction_variable)  # 将预测变量排除出去
        regression_params.interpretation_variables = float_columns + integer_columns  # 解释变量列表

        # --------------------------------第四步：确定所有的连续型变量--------------------------------
        categorical_variables_set = set(categorical_variables.categorical_variables) # 找出唯一值
        regression_params.continuous_variables = [x for x in regression_params.interpretation_variables if
                                                  x not in categorical_variables_set]

        # --------------------------------第五步：确定每个类别变量具体的类别值--------------------------------
        regression_params.categorical_vars_detail = self.analyzing_categorical_variables(state["task"].sample_file, categorical_variables.categorical_variables)

        # --------------------------------第五步：解释变量特征重要性分析--------------------------------
        regression_params.features_importance = analyze_features_importance(state["task"].sample_file,
                                                                            regression_params.prediction_variable,
                                                                            regression_params.interpretation_variables,
                                                                            categorical_variables.categorical_variables)

        return {"next_node": '确认变量分析结果', "regression_params": regression_params}  # 用户确认

    '''
    基于数据文件，任务描述以及浮点类型的列来分析哪个列适合作为预测列
    '''

    async def analyze_prediction_var(self, data_source: str, task_summary: str, soil_property: str,
                                     float_columns: list):
        print_agent_output(f"提交预测变量分析请求...  analyze_prediction_var", agent="DATAEXPLORER")

        try:
            # 1、首先快速分析，不做思考，以chat方式调用LLM
            chat_analyzing_prompt = """即将执行如下土壤属性制图任务：{}其中，需进行预测制图的土壤属性名称为：{}。
                                    为完成上述任务，准备了建模所需的数据，如下为数据中包含的所有相关的变量（列）名称：{}
                                    现在需要你甄别出这些变量中的哪一个是最接近待预测土壤属性的变量，给出这个变量的名称。
                                    """
            messages = [
                HumanMessage(f"{chat_analyzing_prompt}".format(task_summary, soil_property,
                                                               ",".join(float_columns)))]  # 第一个用户消息
            # 调用LLM分析变量类型
            response = call_llm_with_pydantic_output(messages, PredictionVariable)
            if response['parsing_error'] == None:
                analyzing_result = response['parsed']
                if analyzing_result.successful:  # llm给出了预测变量名称
                    if analyzing_result.name in float_columns:  # 给出的预测变量名称是有效的
                        return analyzing_result

            # 2 chat模式未能获取变量，则采用react模式，由agent进行分析
            async with MultiServerMCPClient(
                    {
                        "ExploreData": {
                            "url": "http://localhost:8001/sse",
                            "transport": "sse",
                        }
                    }
            ) as mcp_client:
                self.show_ui_progress("向Agent提交分析预测列的请求...")
                agent = create_react_agent(get_llm(), mcp_client.get_tools(),
                                           response_format=PredictionVariable)
                react_analyzing_prompt = """即将执行如下土壤属性制图任务：{}其中，需进行预测制图的土壤属性名称为：{}。
                        为完成上述任务，准备了建模所需的数据，如下为数据中包含的所有相关的变量（列）名称：{}
                        现在需要你甄别出这些变量中的哪一个是最接近待预测土壤属性的变量，给出这个变量的名称。
                        如果你无法从变量名称中分析出响应变量，可以打开如下数据文件，通过分析列（变量）的数据的值，来尝试确定哪个变量是待预测的土壤属性：{}，响应变量的数值一定是连续型的值。
                        """
                response = await agent.ainvoke({"messages": f"{react_analyzing_prompt}".format(task_summary,
                                                                                               soil_property,
                                                                                               ",".join(
                                                                                                   float_columns),
                                                                                               data_source)})
                return response["structured_response"]
        except Exception as e:
            print_agent_output(f"调用MCP服务提交分析预测列请求异常: {e}", "DATAEXPLORER")
            return None

    '''
    基于数据文件，任务描述以及整数类型的列来分析哪个列可能为类别型变量
    '''

    async def analyze_categoritical_vars(self, data_source, task_summary, integer_columns) -> CategoricalVariables:
        print_agent_output(f"提交类别型变量的分析请求...  analyze_categoritical_var", agent="DATAEXPLORER")
        try:
            # 1、首先调用chat llm快速进行分析
            chat_analyzing_prompt = """即将针对如下土壤属性制图任务进行回归建模：{}，
            为完成上述任务，准备了建模所需的数据，如下为数据中包含的所有相关的变量（列）名称：{}。
            现在需要你结合上述回归建模的业务要求，对这些变量进行分析，找出其中的类别型变量。
            """
            messages = [
                HumanMessage(f"{chat_analyzing_prompt}".format(task_summary, ",".join(integer_columns)))]  # 第一个用户消息
            # 调用LLM分析变量类型
            response = call_llm_with_pydantic_output(messages, CategoricalVariables)
            if response['parsing_error'] == None:
                analyzing_result = response['parsed']
                if analyzing_result.successful:  # llm给出了类别型变量名称
                    if set(analyzing_result.categorical_variables).issubset(integer_columns):
                        return analyzing_result

            # 2 chat模式未能获取变量，则采用react模式，由agent进行分析
            async with MultiServerMCPClient(
                    {
                        "ExploreData": {
                            "url": "http://localhost:8001/sse",
                            "transport": "sse",
                        }
                    }
            ) as mcp_client:
                self.show_ui_progress("向Agent提交分析类别型变量的请求...")
                agent = create_react_agent(get_llm(), mcp_client.get_tools(),
                                           response_format=CategoricalVariables)
                react_analyzing_prompt = """即将针对如下土壤属性制图任务进行回归建模：{}为完成上述任务，准备了建模所需的数据，
                如下为建模数据中包含的所有相关的变量（列）名称：{}。现在需要你结合上述回归建模的业务要求，对这些变量进行分析，找出其中的类别型变量。
                如何你无法从变量名称中分析出类别型变量，可以打开如下数据文件，通过分析列（变量）数据，来尝试确定类别型变量：{}，
                最后，应以Python的字符串列表格式将类别型变量名称返回。
                请注意：类别型列（变量）的特征是：列中包含的所有数据值存在有限数量的唯一值，并且唯一值的数量通常为2-20个，最多不超过50个。
                """
                response = await agent.ainvoke({"messages": f"{react_analyzing_prompt}".format(task_summary,
                                                                                               ",".join(
                                                                                                   integer_columns),
                                                                                               data_source)})

                return response["structured_response"]
        except Exception as e:
            print_agent_output(f"调用MCP服务提交分析类别型变量请求异常: {e}", "DATAEXPLORER")
            return None

    '''
    对类别型变量进行分析，最终会将所有类别变量的唯一值保存至.pkl文件存储，该文件和原始数据文件位于同一级目录下
    '''

    def analyzing_categorical_variables(self, structured_data_file: str, candidate_vars: list):
        df = pd.read_csv(structured_data_file)
        analysis_results = {}
        for candidate_var in candidate_vars:
            unique_value = natsorted(df[candidate_var].astype(str).unique())  # 转换为字符串、取唯一值，并按照字符串自然顺序排序
            analysis_results[candidate_var] = ','.join(unique_value)
        # cate_file_name = os.path.dirname(structured_data_file) + \
        #                  os.path.splitext(os.path.basename(structured_data_file))[0]
        # cate_file_name += '_cat.pkl'
        # with open(cate_file_name, 'wb') as f:
        #     pickle.dump(analysis_results, f)
        return analysis_results

    '''
    agent主控函数：定义一个接收任务的子图，完成对数据的探索分析
    采用Evaluator-Optimizer设计模式，在用户给出数据路径后，进行自动化的探索评估，经过多轮迭代，最终掌握用户所提供数据的构成
    '''

    async def async_run(self, global_state: TaskState) -> TaskState:
        print_agent_output(f"开始探索数据...", agent="DATAEXPLORER")

        # 初始化子图结构
        async with AsyncSqliteSaver.from_conn_string('./explore_data.db') as checkpointer:
            # 初始化子图
            self.init_workflow(checkpointer)

            thread_config = {"configurable": {"thread_id": global_state["task"].task_id}}

            # 检测当前节点是否已经执行完毕，是则跳过，同时获取历史状态，用于全局状态的恢复
            if global_state["task"].stage != TaskStage.DataExplore:  # 如果当前任务不处于确定任务总体信息的状态，则短路跳出
                state = await self.subgraph.aget_state(thread_config)
                return {"data_source_for_build_model": state.values["task"].sample_file,
                        "params_for_build_model": state.values["regression_params"]}

            # 启动子图的执行，进入accept_input节点
            # 检测该子图是否已执行完毕
            history_states = [state async for state in self.subgraph.aget_state_history(thread_config)]
            if len(history_states) > 0:  # 存在历史的check points,处于恢复模式（从长任务中恢复历史上的执行状态）
                if len(history_states[0].next) == 0:
                    print("checkpoint异常，可能是从调试中恢复！！！")
                    self.interrput_event.set()
                    self.quit_chat = True
                else:
                    await self.subgraph.ainvoke(None, history_states[0].config)
            else:  # 否则，初次执行
                await self.subgraph.ainvoke(input={"task": global_state["task"]}, config=thread_config)
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

                    input_finished_event = self.agent_signals.create_input_finished_event()  # 创建一个python的事件，用于阻塞线程，直到获取用户的反馈
                    input_finished_event.clear()  # 重置信号，以便用户给出反馈
                    if len(state.values) == 0:  # 首次来到这里，即：处于获取数据源的节点
                        self.agent_signals.user_set_data_source.emit(self.agent_signals.task_id, input_finished_event,
                                                                     suggestion)  # 请求用户指定数据源
                    else:  # 已经是非首次获取数据源的情况
                        if state.next is not None:  # 已经有明确的下一步
                            if state.next[0] == "用户设置数据源":  # 指定数据源
                                self.agent_signals.user_set_data_source.emit(self.agent_signals.task_id,
                                                                             input_finished_event,
                                                                             global_state[
                                                                                 "task"].sample_file,
                                                                             suggestion)  # 请求用户指定数据源
                            elif state.next[0] == "用户指定响应变量":
                                schema = StructureDataDealer.get_structed_data_schema(
                                    state.values["task"].sample_file,
                                    1)  # 获取结构化数据的schema
                                var_names = [str(var_name) for var_name in schema.keys()]
                                # 将坐标变量排除
                                if config.CSV_GEOM_COL_X in var_names:
                                    var_names.remove(config.CSV_GEOM_COL_X)
                                if config.CSV_GEOM_COL_Y in var_names:
                                    var_names.remove(config.CSV_GEOM_COL_Y)
                                self.agent_signals.user_confirm_predict_var.emit(self.agent_signals.task_id,
                                                                                 input_finished_event,
                                                                                 var_names,
                                                                                 state.values[
                                                                                     'regression_params'].prediction_variable,
                                                                                 suggestion)  # 请求用户直接指定预测变量
                            elif state.next[0] == "用户指定解释变量":
                                float_columns, integer_columns = self.parse_data_schema(
                                    state.values["task"].sample_file)
                                float_columns.remove(state.values["regression_params"].prediction_variable)  # 将预测变量排除
                                # 将坐标变量排除
                                if float_columns is not None:
                                    if config.CSV_GEOM_COL_X in float_columns:
                                        float_columns.remove(config.CSV_GEOM_COL_X)
                                    if config.CSV_GEOM_COL_Y in float_columns:
                                        float_columns.remove(config.CSV_GEOM_COL_Y)
                                self.agent_signals.user_confirm_interpretation_vars.emit(self.agent_signals.task_id,
                                                                                         input_finished_event,
                                                                                         float_columns + integer_columns,
                                                                                         state.values[
                                                                                             'regression_params'].interpretation_variables)  # 请求用户直接指定预测变量
                            elif state.next[0] == "用户指定类别型变量":  # 请求用户指定类别型变量
                                self.agent_signals.user_confirm_categorical_variables.emit(self.agent_signals.task_id,
                                                                                           input_finished_event,
                                                                                           state.values[
                                                                                               'regression_params'].interpretation_variables,
                                                                                           state.values[
                                                                                               'regression_params'].categorical_variables,
                                                                                           suggestion)  # 请求用户直接指定预测变量
                            elif state.next[0] == "选择确认响应变量的方式":
                                self.agent_signals.user_choice.emit(self.agent_signals.task_id,
                                                                    DEUIType.Prediction,
                                                                    input_finished_event, suggestion)
                            elif state.next[0] == "选择确认类别型变量的方式":
                                self.agent_signals.user_choice.emit(self.agent_signals.task_id,
                                                                    DEUIType.Categorical,
                                                                    input_finished_event, suggestion)
                            elif state.next[0] == "用户确认变量分析结果":
                                confirm_info = {"用于建模的数据源": state.values["task"].sample_file,
                                                "预测变量": state.values["regression_params"].prediction_variable,
                                                "解释变量中的连续型变量": ','.join(state.values[
                                                                                       "regression_params"].continuous_variables),
                                                "解释变量中的类别型变量": ','.join(state.values[
                                                                                       "regression_params"].categorical_variables),
                                                "解释变量": ','.join(
                                                    state.values["regression_params"].interpretation_variables),
                                                }
                                if state.values[
                                    'regression_params'].categorical_vars_detail is None:  # 由于反序列化结果为None，传入信号会出错
                                    state.values['regression_params'].categorical_vars_detail = {}  # 需要将None转换为空字典
                                self.agent_signals.user_confirm_data_explorer.emit(self.agent_signals.task_id,
                                                                                   input_finished_event,
                                                                                   confirm_info,
                                                                                   state.values[
                                                                                       "regression_params"].categorical_vars_detail,
                                                                                   state.values[
                                                                                       "regression_params"].features_importance)  # 请求用户指定数据源
                        else:  # 数据源已有，待评估
                            self.agent_signals.user_set_data_source.emit(self.agent_signals.task_id,
                                                                         input_finished_event,
                                                                         suggestion)  # 请求用户指定数据源
                    input_finished_event.wait()  # 阻塞，直到输入结束事件被触发(获取到了用户的反馈)

                    # 3、恢复interrupt的执行
                    await self.subgraph.ainvoke(Command(resume=input_finished_event.data), config=thread_config)

            # transform response back to the parent state
            last_state = await self.subgraph.aget_state(thread_config)
            latest_state_values = last_state.values
            self.agent_signals.data_source_ready.emit(self.agent_signals.task_id,
                                                      latest_state_values["task"].sample_file)
            # 将任务阶段信息写入数据库，指示进入建模阶段
            TaskDataAccess.update_task_stage(self.agent_signals.task_id, TaskStage.BuildModel)
            # 更新全局的State状态
            global_state["task"].stage = TaskStage.BuildModel  # 指示进入下一环节
            self.show_ui_progress("即将进入回归建模阶段...")
            return {"task": global_state["task"],
                    "params_for_build_model": latest_state_values["regression_params"],
                    "esda_result": latest_state_values["esda_result"]}  # 向主流程返回用于建模的数据源等信息
