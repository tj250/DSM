from typing import TypedDict, Any
from data_access.task import TaskInfo
from .base_data_structure import RegressionParams
from eda.analysis import EsdaAnalysisResult

'''
土壤属性制图任务的信息，会伴随整个Graph在不同Node之间传递
'''
class TaskState(TypedDict):
    task: TaskInfo                      # 土壤属性制图任务信息
    params_for_build_model: RegressionParams | Any          # 建模数据源相关的基本参数信息，包括解释变量和预测变量信息
    esda_result: EsdaAnalysisResult     # 探索性数据分析的结果
    prediction_algorithms_id: str = ''   # 选定的用于预测的模型的ID（可能为单一算法模型，也可能为堆叠模型）
    # resume_mode:bool                    # 是否处于恢复模式
    # parent_ref:Any                      # 存储父图引用
