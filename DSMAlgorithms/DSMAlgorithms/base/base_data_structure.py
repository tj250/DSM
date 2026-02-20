from enum import Enum


'''
模型库当前支持的算法类型枚举
'''


class AlgorithmsType(Enum):
    UNDEFINED = '未定义'  # 未明确定义
    STACKING = '堆叠'  # 堆叠
    CK = 'Cokriging'  # 协同克里金
    RK = 'RegressionKriging'  # 回归克里金
    MGWR = 'MultiscaleGWR'  # 多尺度地理加权回归
    EN = 'ElasticNet'  # 弹性网络
    GLM = 'GeneralizedLinearModels'  # 广义线性回归
    PLSR = 'PartialLeastSquaresRegression'  # 偏最小二乘回归
    KNR = 'K-nearestNeighborRegression'  # K近邻回归
    SVR = 'SupportVectorRegression'  # 支持向量回归
    RFR = 'RandomForestRegression'  # 随机森林回归
    XGBR = 'XgboostRegression'  # 极限梯度提升回归
    XGBRFR = 'XgboostRFRegression'  # 极限梯度提升随机森林回归
    MLP = 'MultilayerPerceptron'  # 多层感知机
    CUSTOM = '自定义'   # 自定义的扩展算法


algorithms_dict = {AlgorithmsType.UNDEFINED: '未定义', AlgorithmsType.STACKING: '多算法的堆叠',
                   AlgorithmsType.CK: '协同克里金', AlgorithmsType.RK: '回归克里金',
                   AlgorithmsType.MGWR: '多尺度地理加权回归', AlgorithmsType.EN: '弹性网络',
                   AlgorithmsType.GLM: '广义线性模型', AlgorithmsType.PLSR: '偏最小二乘回归',
                   AlgorithmsType.KNR: 'K近邻回归', AlgorithmsType.SVR: '支持向量回归',
                   AlgorithmsType.RFR: '随机森林回归', AlgorithmsType.XGBR: '极限梯度提升回归',
                   AlgorithmsType.MLP: '多层感知机', AlgorithmsType.CUSTOM: '自定义', }

'''
响应变量的数据分布类型，与sklearn.linear_model.TweedieRegressor中power参数的取值保持一致
'''
class DataDistributionType(Enum):
    Unknown = -1               # 未明确定义
    Normal = 0                   # 正态分布
    Possion = 1                 # 泊松分布
    CompoundPossionGamma = 1.5   # 复合泊松-伽马分布
    Gamma = 2                     # 伽马分布
    InverseGaussin = 3   # 逆高斯分布


'''
对建模的解释变量数据进行变换的方式
'''
class DataTransformType(Enum):
    Nochange = 0         # 未明确定义
    ZScore = 1           # 正态分布
    Normalize = 2        # 泊松分布
