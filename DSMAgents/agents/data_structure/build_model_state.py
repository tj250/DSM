from typing import TypedDict, Any
from data_access.task import TaskInfo
from typing_extensions import TypedDict
from .base_data_structure import RegressionParams, RegressionModelInfo
from agents.utils.views import BMActiveState
from eda.analysis import EsdaAnalysisResult
from agents.utils.views import BMUIChoice

# Define the schema for the Build Model
class BuildModelState(TypedDict):
    task: TaskInfo  # 土壤属性制图任务信息
    stage: int = 0     # 当前所处阶段
    params_for_build_model: RegressionParams  # 建模数据源相关的基本参数信息，对于回归，则是解释变量和预测变量信息
    build_model_state: BMActiveState = BMActiveState.Beginning  # 长时间的建模过程当前所处的状态
    algorithms_info: dict[str,str]  # 经过启发式规则过滤出来的算法名称及其对应的ID
    build_model_results:list[RegressionModelInfo] # 建模结果中各个算法的参数信息
    esda_result: EsdaAnalysisResult  # 探索性数据分析的结果
    choice:BMUIChoice = BMUIChoice.NoneChoice  # 用户做出的选择

