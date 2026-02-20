from typing import Optional
from pydantic import BaseModel, Field


class PredictionVariable(BaseModel):
    """用于回归建模的预测变量分析结果。"""

    successful: bool = Field(description="是否成功地分析出了预测变量")
    name: str = Field(description="预测变量名称")


# class RegressionVariables(BaseModel):
#     """用于回归建模的结构化数据的数据列分析结果。"""
#
#     successful: bool = Field(description="是否成功地分析出了预测变量")
#     predict_variable: Optional[str] = Field(description="预测变量")
#     categorical_variables: Optional[list[str]] = Field(default=None, description="解释变量中属于类别型变量的变量名称列表")

class CategoricalVariables(BaseModel):
    """用于回归建模的解释变量中属于类别型的变量的列表。"""

    successful: bool = Field(description="是否成功地分析出了类别型变量")
    categorical_variables: list[str] = Field(default=None, description="类别型变量的列表")
    # reason:Optional[str] = Field(description="当无法分析出类别型变量时，给出的详细原因说明")


class CategoricalVariablesAnaylisis(BaseModel):
    """对单个变量进行是否为类别型变量分析的结果。"""

    variable_name: str = Field(default=None, description="被分析的变量的名称")
    is_categorical_variable: bool = Field(default=True, description="被分析的变量是否确实为类别变量")
    categorical_values: Optional[list[str]] = Field(default=None,
                                                    description="在被分析变量为类别变量时，用逗号连接的各个类别值")
