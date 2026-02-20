from typing_extensions import TypedDict
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from data_access.task import TaskInfo
from agents.utils.views import DEUIChoice
from .base_data_structure import RegressionParams
from eda.analysis import EsdaAnalysisResult

'''
数据探索子图的State类
'''


# Define the schema for the datasource management
class ExploreDataState(TypedDict):
    task: TaskInfo  # 任务信息
    chat_messages: list[AIMessage | HumanMessage | SystemMessage | ToolMessage]  # 对话历史记录
    regression_params: RegressionParams  # 回归参数
    user_choice: DEUIChoice = DEUIChoice.NoneChoice  # 用户在UI层做出的选择，传递至Agent。0：重新设置数据源，1-指定预测变量，2-指定类别型变量
    next_node: str = ''  # 数据探索后的下一步节点的名称
    suggestion: str = ''  # llm输出的建议，反馈给用户，请求用户根据建议再一次给出输入
    esda_result: EsdaAnalysisResult  # 探索性数据分析的结果

