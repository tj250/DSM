from typing import Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


@dataclass
class ModelBasicInfo():
    """模型基本信息。"""

    model_name: str = field(default='', metadata={"description": "模型名称"})
    model_id: str = field(default='', metadata={"description": "模型ID"})


class BuildModelTaskAcceptState(BaseModel):
    """记录一次回归建模任务所需的输入信息。"""

    successful: bool = Field(description="是否成功地记录了建模任务所需的输入信息")
    model_basic_info: Optional[list[ModelBasicInfo]] = Field(
        description="模型的基本信息列表，包括每个模型的名称和对应的建模任务ID")


class BuildModelIsFinished(BaseModel):
    """一次回归建模任务相关过程是否结束的信息。"""

    finished: bool = Field(description="一次回归建模任务相关过程是否结束")

#
# class RegressionModelInfo(BaseModel):
#     """回归模型的参数信息。"""
#
#     model_id: str = Field(description="模型ID")
#     model_name: str = Field(description="模型名称")
#     R2: float = Field(description="R2指标")
#     RMSE: float = Field(description="RMSE指标")
#     start_time: str = Field(description="建模开始时间")
#     finished_time: str = Field(description="建模结束时间")
