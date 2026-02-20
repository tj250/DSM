from typing_extensions import TypedDict
from agents.utils.views import UncertaintyType
from .base_data_structure import MappingMetrics,RegressionParams,EvaluatingMetrics
from data_access.task import TaskInfo
from DSMAlgorithms.base.dsm_base_model import AlgorithmsType
from eda.analysis import EsdaAnalysisResult

'''
数据探索子图的State类
'''


# Define the schema for the datasource management
class EvaluationState(TypedDict):
    task: TaskInfo  # 任务ID
    selected_algorithms: dict[str, AlgorithmsType|list[AlgorithmsType]]  # 用户所选择的算法的集合,key为算法ID，value为算法类型字符串
    covariates_path: str  # 协变量栅格文件存储的路径
    mapping_area_file: str  # 制图范围栅格文件
    params_for_build_model: RegressionParams  # 建模数据源相关的基本参数信息，对于回归，则是解释变量和预测变量信息
    mapping_results: list[MappingMetrics]  # 预测制图的结果
    evaluating_results:list[EvaluatingMetrics]   # 评估的结果
    uncertainty_metrics_type: list[UncertaintyType]         # 不确定分布的指标集合
    indepent_file:str                       # 独立集验证时的文件
    esda_result:EsdaAnalysisResult
