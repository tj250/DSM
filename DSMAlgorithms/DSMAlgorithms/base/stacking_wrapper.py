import numpy as np
import pandas as pd
import algorithms_config
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import TweedieRegressor, GammaRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from DSMAlgorithms.sklearn_wrap.cokriging import CoKrigingRegressor
from DSMAlgorithms.sklearn_wrap.regression_kriging import RegressionKrigingRegressor
from DSMAlgorithms.sklearn_wrap.mgwr2 import MGWRegressor
from DSMAlgorithms.sklearn_wrap.custom_regressor import CustomRegressor
from DSMAlgorithms.base.base_data_structure import AlgorithmsType
from data.mean_encoder import MeanEncoder

'''
用于变换不同算法fit和predict时所需的数据，即：将原始数据转换为算法所需的输入
此方法内部执行两项工作：
1、针对无需坐标值列的算法，去除DataFrame中的坐标列
2、对连续型变量进行评价编码（除XGBoost算法所需数据外）
注意：传入的DataFrame在prepare_dataset时，已经进行了归一化/标准化处理，此处无需再行处理
'''

def transform_data(X: pd.DataFrame, algorithms_type: AlgorithmsType, mean_encoder:MeanEncoder = None):
    need_geometry = (algorithms_type == AlgorithmsType.MGWR or
                     algorithms_type == AlgorithmsType.CK or
                     algorithms_type == AlgorithmsType.RK)
    # 1、不需要坐标值列但是X列中包含了坐标值列（堆叠模型时，一个模型需要，但是两外一个又不需要情况夏会发生）,需要去除
    if not need_geometry and algorithms_config.CSV_GEOM_COL_X in X.columns:
        X.pop(algorithms_config.CSV_GEOM_COL_X)
        X.pop(algorithms_config.CSV_GEOM_COL_Y)

    # 2、对类别型变量进行处理
    if algorithms_type == AlgorithmsType.XGBR or algorithms_type == AlgorithmsType.XGBRFR:  # XGBoost类算法，无需特别处理
        return X
    else:
        X_encoder = mean_encoder.transform(X)  # 非XGBoost中的回归算法，采用与建模时相同的平均编码器进行平均编码
        return X_encoder


'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class CustomModelStackingWrapper(CustomRegressor):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.CUSTOM
        newargs = algorithms_kwargs.copy()
        if 'mean_encoder' in newargs:
            del newargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**newargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new

'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class RegressionKrigingRegressorStackingWrapper(RegressionKrigingRegressor):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.RK
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new

'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class CoKrigingRegressorStackingWrapper(CoKrigingRegressor):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.CK
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)


    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new

'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class MGWRStackingWrapper(MGWRegressor):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.MGWR
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)


    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    # 此预测方法仅在堆叠建模时调用，单一建模或预测制图时不会调用，因此，对于传入X的处理方式不同于MGWRegressor本身的predict方法
    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)  # 调用回归器前，数据必须预先处理好，回归器自身的predict方法不会进行更多处理（仅使用并丢弃坐标列）

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new

'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class ElasticNetStackingWrapper(ElasticNet):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.EN
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new


'''
仅用于模型堆叠时调用
'''
class TweedieRegressorStackingWrapper(TweedieRegressor):
    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.GLM
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new

'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class KNeighborsRegressorStackingWrapper(KNeighborsRegressor):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.KNR
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new

'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class MLPRegressorStackingWrapper(MLPRegressor):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.MLP
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new


'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class PLSRegressionStackingWrapper(PLSRegression):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.PLSR
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new

'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''

class RandomForestRegressorStackingWrapper(RandomForestRegressor):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.RFR
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new

'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class SVRStackingWrapper(SVR):

    def __init__(self, **algorithms_kwargs):
        self.mean_encoder = algorithms_kwargs['mean_encoder']       # 单独记录下来，fit和predict中需要
        self.kwargs_inner = algorithms_kwargs                       # 必须记录下所有参数，StackingRegressor内部克隆时恢复所有参数值
        self.algorithms_type = AlgorithmsType.SVR
        if 'mean_encoder' in algorithms_kwargs:
            del algorithms_kwargs['mean_encoder']                   # 删除，因为构造方法中不能有这个参数
        super().__init__(**algorithms_kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), self.algorithms_type, self.mean_encoder)
        return super().predict(new_X)

    def get_params(self, deep=True):
        parent_params_new = self.kwargs_inner
        parent_params_new['mean_encoder'] = self.mean_encoder
        return parent_params_new


'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class XGBRegressorStackingWrapper(XGBRegressor):

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), AlgorithmsType.XGBR)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), AlgorithmsType.XGBR)
        return super().predict(new_X)




'''
用于堆叠的包装器(解决堆叠时输入的X中包含坐标的问题)
'''
class XGBRFRegressorStackingWrapper(XGBRFRegressor):

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X = transform_data(X.copy(), AlgorithmsType.XGBRFR)
        super().fit(new_X, y)
        return self

    def predict(self, X: pd.DataFrame):
        new_X = transform_data(X.copy(), AlgorithmsType.XGBRFR)
        return super().predict(new_X)
