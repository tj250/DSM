from typing import Union
from typing import Optional
from pydantic import BaseModel, Field


class TaskAnalyzingResult(BaseModel):
    """分析一段自然语言文本，以确定该文本所描述的内容是否为土壤属性制图任务。当描述中包含了待预测的土壤属性以及研究区域的地理环境背景信息时，方可认为是有效的土壤属性制图任务。"""

    isvalid: bool = Field(description="是否为有效的土壤属性制图任务")
    soil_property: Optional[str] = Field(description="在文本描述为有效的土壤属性制图任务时,所确定的土壤属性名称")
    region_geo_environment: Optional[str] = Field(description="数字土壤制图研究区域的地理环境背景信息，通常包括诸如地形、气候、地理分区等")
    summary: Optional[str] = Field(description="在文本描述为有效的土壤属性制图任务时,对土壤属性制图任务的自然语言形式的优化描述")
    suggestion: Optional[str] = Field(description="在文本描述不是有效的土壤属性制图任务时,由大语言模型给出的有关如何改进任务描述的建议")
#
# class SuggestionWhenFailedToAnalysis(BaseModel):
#     """在不能确认任务描述为有效的土壤属性制图任务情况下，给出的修改任务描述的建议，以便用户改进任务描述。"""
#
#     suggestion: str = Field(description="由大语言模型给出的有关修改原始的任务描述的建议")
#
#
# class FinalResponse(BaseModel):
#     final_output: Union[TaskDescriptionWhenSuccessAnalysis,SuggestionWhenFailedToAnalysis]

class EvaluatePropertyResult(BaseModel):
    """分析一段自然语言文本是否描述了土壤属性制图中需要预测的土壤属性。"""

    isvalid: bool = Field(description="给出的自然语言文本中是否包含了有效的土壤属性")
    suggestion: Optional[str] = Field(description="在文本描述不包含有效的待预测土壤属性时,由大语言模型给出的有关如何改进准确描述土壤属性的建议")


class EvaluateEnvironmentBGResult(BaseModel):
    """分析一段自然语言文本是否描述了土壤属性制图中某个研究区域的地理环境背景。"""

    isvalid: bool = Field(description="给出的自然语言文本是否描述了土壤属性制图中某个研究区域的地理环境背景")
    suggestion: Optional[str] = Field(description="在文本描述不是有效的研究区域地理环境背景时,由大语言模型给出的有关如何改进对研究区域地理环境背景的建议")

