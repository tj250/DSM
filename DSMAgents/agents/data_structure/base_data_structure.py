from dataclasses import dataclass
from agents.utils.views import RegressionPredictionType
from DSMAlgorithms.base.dsm_base_model import AlgorithmsType


@dataclass
class PredictionResult():
    prediction_variable: str = ''  # 预测变量名称
    interpretation_variables: str = ''  # 解释变量名称
    categorical_variables: str = ''  # 解释变量中的类别型变量名称
    continuous_variables: str = ''  # 解释变量中的连续性变量


'''
回归所涉及的参数信息
'''


@dataclass
class RegressionParams():
    interpretation_variables: list = None  # 解释变量名称
    categorical_variables: list = None  # 解释变量中的类别型变量名称
    continuous_variables: list = None  # 解释变量中的连续性变量
    prediction_variable: str = ''  # 预测变量名称
    categorical_vars_detail: dict[str, str] = None  # 类别变量的细节，字典中的每个键是类别变量的名称，而内容则是用逗号分隔的变量值列表
    features_importance: dict = None  # 解释变量的特征重要性


'''
建模的参数指标信息
'''


@dataclass
class RegressionModelInfo():
    algorithms_id: str = ''  # 算法的唯一标识符
    algorithms_name:str | list[str] = ''  # 算法名称
    algorithms_type: AlgorithmsType | list[AlgorithmsType] = AlgorithmsType.UNDEFINED  # 算法类型
    CV_best_score: float = None
    R2: float = None
    RMSE: float = None
    stacking: bool = False  # 是否是堆叠模型


'''
制图预测结果指标
'''


@dataclass
class MappingMetrics():
    algorithms_id: str = ''  # 算法ID
    algorithms_name: str = ''
    algorithms_type: AlgorithmsType = AlgorithmsType.UNDEFINED  # 算法名称
    min_value: float = None  # 预测栅格像元的最小值
    max_value: float = None  # 预测栅格像元的最小值
    mean_value: float = None  # 预测栅格像元的均值
    CV_best_score: float = None  # 建模时交叉验证的最佳分数
    R2: float = None    # 测试集上的最佳分数
    RMSE: float = None  # 测试集上的RMSE


'''
评估指标（不确定性指标和独立集验证指标）
'''


@dataclass
class EvaluatingMetrics():
    algorithms_id: str = ''  # 算法ID
    algorithms_type: AlgorithmsType = AlgorithmsType.UNDEFINED  # 算法类型
    algorithms_name: list = None # 算法名称
    PICP: float = None
    MPIW: float = None
    R2: float = None    # 独立验证集上的最佳分数
    RMSE: float = None  # 独立验证集上的RMSE
