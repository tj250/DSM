from typing import Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import datetime,math


# @dataclass
# class ModelBasicInfo():
#     """模型基本信息。"""
#
#     model_name: str = field(default='', metadata={"description": "模型名称"})
#     model_id:str = field(default='', metadata={"description": "模型ID"})


class PredictionTaskAcceptState(BaseModel):
    """记录一次回归预测所需的输入信息。"""

    successful: bool = Field(description="是否成功地记录了回归预测所需的输入信息")
    # task_id: Optional[str] = Field(description="预测任务的ID")



class PredictionIsFinished(BaseModel):
    """一次回归建模任务相关过程是否结束的信息。"""

    finished: bool = Field(description="一次回归预测相关过程是否结束")

