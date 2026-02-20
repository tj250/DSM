import pandas as pd
from DSMAlgorithms.base.dsm_base_model import DSMBaseModel, AlgorithmsType, DataTransformType
from DSMAlgorithms.sklearn_wrap.cokriging import CoKrigingRegressor
from sklearn.linear_model import ElasticNet, TweedieRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from DSMAlgorithms.sklearn_wrap.regression_kriging import RegressionKrigingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.ensemble import StackingRegressor
import algorithms_config

'''
弹性网络模型（用于预测）
'''


class ElasticNetModel(DSMBaseModel):
    def __init__(self, model:ElasticNet, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.EN, model, transform_info, mean_encoder)


'''
广义线性模型（用于预测）
'''


class GLMModel(DSMBaseModel):

    def __init__(self, model:TweedieRegressor, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.GLM, model, transform_info, mean_encoder)


'''
K近邻回归模型（用于预测）
'''


class KNRModel(DSMBaseModel):

    def __init__(self, model:KNeighborsRegressor, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.KNR, model, transform_info, mean_encoder)


'''
地理加权回归模型(仅用于预测)
'''

class MGWRModel(DSMBaseModel):
    def __init__(self, model, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.MGWR, model, transform_info, mean_encoder)


'''
MLP模型（用于预测）
'''


class MLPModel(DSMBaseModel):

    def __init__(self, model:MLPRegressor, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.MLP, model, transform_info, mean_encoder)


'''
偏最小二乘回归模型（用于预测）
'''


class PLSRModel(DSMBaseModel):

    def __init__(self, model:PLSRegression, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.PLSR, model, transform_info, mean_encoder)


'''
随机森林回归模型（用于预测）
'''


class RandomForestRegressionModel(DSMBaseModel):

    def __init__(self, model:RandomForestRegressor, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.RFR, model, transform_info, mean_encoder)

'''
协同克里金模型（用于预测）
'''


class CoKrigeModel(DSMBaseModel):

    def __init__(self, model: CoKrigingRegressor, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.CK, model, transform_info, mean_encoder)


'''
回归克里金模型（用于预测）
'''


class RegressionKrigeModel(DSMBaseModel):

    def __init__(self, model:RegressionKrigingRegressor, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.RK, model, transform_info, mean_encoder)


'''
支持向量回归模型（用于预测）
'''


class SVRModel(DSMBaseModel):

    def __init__(self, model:SVR, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.SVR, model, transform_info, mean_encoder)


'''
xgboost回归模型（用于预测）
'''


class XGBRModel(DSMBaseModel):

    def __init__(self, model:XGBRegressor, transform_info: dict):
        super().__init__(AlgorithmsType.XGBR, model, transform_info)


'''
xgboost 随机森林回归模型（用于预测）
在制图阶段，重建模型后会调用predict方法生成预测结果
'''


class XGBRFRModel(DSMBaseModel):

    def __init__(self, model:XGBRFRegressor, transform_info: dict):
        super().__init__(AlgorithmsType.XGBRFR, model, transform_info)


'''
自定义模型（用于预测）
在制图阶段，重建模型后会调用predict方法生成预测结果
'''

class CustomModel(DSMBaseModel):

    def __init__(self, model: StackingRegressor, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.CUSTOM, model, transform_info, mean_encoder)




'''
堆叠模型（用于预测）
'''

class StackingModel(DSMBaseModel):

    def __init__(self, model: StackingRegressor, transform_info: dict, mean_encoder: dict):
        super().__init__(AlgorithmsType.STACKING, model, transform_info, mean_encoder)

    '''
    重载预测方法，非常重要，不能沿用基类（DSMBaseModel）的predict方法，因为多个基模型对传入数据X的处理方式不同，需要在各个基模型的predict方法中各自处理，堆叠模型不能统一处理。
    '''
    def predict(self, X: pd.DataFrame):
        # 原始数据需要经过标准化或归一化处理
        for col in X.columns:
            if col != algorithms_config.CSV_GEOM_COL_X and col != algorithms_config.CSV_GEOM_COL_Y and not isinstance(X[col].dtype,
                                                                                                pd.CategoricalDtype):
                if DataTransformType(self.transform_info['transform_type']) == DataTransformType.Normalize:  # min-max归一化
                    X[col] = (X[col] - self.transform_info[col]['min']) / (
                                self.transform_info[col]['max'] - self.transform_info[col]['min'] + 1e-8)
                elif DataTransformType(self.transform_info['transform_type']) == DataTransformType.ZScore:
                    X[col] = (X[col] - self.transform_info[col]['mean']) / self.transform_info[col]['std']
        # 然后再路由至基回归器处理
        return self.model.predict(X)