import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from pykrige.uk import UniversalKriging
import algorithms_config
import math

'''
协同克里金回归器
'''


class CoKrigingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 variogram_model='spherical',  # linear, power, gaussian, spherical, exponential
                 nlags=6,
                 weight=True,
                 exact_values=True,
                 anisotropy_scaling=1.0,
                 anisotropy_angle=0.0,  # 各向异性的角度，CCW,0-360度
                 feature_names_in_=None,  # fit时的特征列
                 ):
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.weight = weight
        self.exact_values = exact_values
        self.anisotropy_scaling = anisotropy_scaling
        self.anisotropy_angle = anisotropy_angle
        self.feature_names_in_ = feature_names_in_


    def __sklearn_tags__(self):  # 重要，否则堆叠过程中不认这个回归器
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    '''
    X:所有环境协变量的值
    y：预测变量的值
    '''

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        new_X_arr = X.copy().values
        covars = new_X_arr[:, :-2]  # 转换未numpy数组后再切片
        self.uk = UniversalKriging(
            new_X_arr[:, -2:-1], new_X_arr[:, -1:], y,
            variogram_model=self.variogram_model,
            nlags=self.nlags,
            weight=self.weight,
            exact_values=self.exact_values,
            anisotropy_scaling=self.anisotropy_scaling,
            anisotropy_angle=self.anisotropy_angle,
            # 如下两项是非可调参数，这两项引入了环境协变量用于预测
            drift_terms=['specified'],  # 引入漂移项（drift terms）来调整模型对数据的拟合，以适应非平稳特性。
            # 手动指定空间坐标之外的附加变量作为漂移项，这些变量通常与目标变量存在物理或统计相关性（如高程对气温的影响）
            specified_drift=[covars[:, i] for i in range(covars.shape[1])],
        )
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def predict(self, X: pd.DataFrame):
        X_arr = X.to_numpy()
        # 由于uk.execute()对内存需求量大，因此需要控制，以分块方式处理，以下代码支持16GB内存下的预测。
        block_bytes = 10 * 1024 * 1024  # 每10M一个块
        blocks_elements = int(block_bytes / np.dtype(np.float32).itemsize / X_arr.shape[1])  # 计算每一块对应的元素数量
        blocks = int(math.ceil(X_arr.shape[0] / blocks_elements))  # 总块数
        z_pred = None
        for index in range(blocks):  # 遍历每一块
            start_index = index * blocks_elements  # X_arr的每一块的第一维度的起始索引
            end_index = X_arr.shape[0] if index == blocks - 1 else (index + 1) * blocks_elements  # X_arr的每一块的第一维度的结束索引
            covars = X_arr[start_index:end_index, :-2]
            z_pred_index, _ = self.uk.execute('points', X_arr[start_index:end_index, -2:-1],
                                              X_arr[start_index:end_index, -1:],
                                              specified_drift_arrays=[covars[:, i] for i in range(covars.shape[1])])
            if z_pred is None:
                z_pred = z_pred_index  # 初次复制
            else:
                z_pred = np.concatenate((z_pred, z_pred_index))  # 拼接
        return z_pred

    def get_params(self, deep=True):
        return {
            'variogram_model': self.variogram_model,
            'nlags': self.nlags,
            'weight': self.weight,
            'exact_values': self.exact_values,
            'anisotropy_scaling': self.anisotropy_scaling,
            'anisotropy_angle': self.anisotropy_angle,
            'feature_names_in_': self.feature_names_in_,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
