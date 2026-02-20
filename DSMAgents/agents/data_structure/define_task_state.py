from typing_extensions import TypedDict
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)

# Define the schema for the input
class DefineTaskState(TypedDict):
    soil_property: str      # 制图的土壤属性
    task_summary: str       # 由大语言模型总结得出的任务概述
    region_geo_environment:str  # 研究区域的地理环境背景信息
    chat_messages: list[AIMessage | HumanMessage | SystemMessage | ToolMessage]  # 对话历史记录
    suggestion: str = ''    # llm输出的建议，反馈给用户，请求用户根据建议再一次给出输入
    task_desc: str = ''     # 用户最近一次给出的任务描述
    user_choice:int = -1    # 用户确认任务时的选择，0：开启新一轮对话，重新进行任务分析，其它：分析结束，可以进入下一个环节