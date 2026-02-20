import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from pykrige.rk import RegressionKriging
from sklearn.metrics import r2_score

import algorithms_config

'''
回归克里金回归器
'''


class RegressionKrigingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,regression_model=None,  # 趋势项的回归方法，必须是来自sklearn中定义的类
                 variogram_model='spherical',
                 n_closest_points=10,
                 nlags=6,
                 weight=True,
                 exact_values=True,
                 anisotropy_scaling=(1.0, 1.0),
                 anisotropy_angle=(0.0, 0.0),
                 feature_names_in_=None,  # fit时的特征列
                 ):
        self.regression_model = regression_model
        self.variogram_model = variogram_model
        self.n_closest_points = n_closest_points
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

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.rk = RegressionKriging(
            regression_model=self.regression_model,
            variogram_model=self.variogram_model,
            n_closest_points=self.n_closest_points,
            nlags=self.nlags,
            weight=self.weight,
            exact_values=self.exact_values,
            anisotropy_scaling=self.anisotropy_scaling,
            anisotropy_angle=self.anisotropy_angle,
            # 以下为非可调参数
            method='ordinary',
            coordinates_type='euclidean',
        )
        new_X = X.copy()
        self.rk.fit(new_X.values[:, :-2], new_X.values[:, -2:], y)
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def predict(self, X: pd.DataFrame):
        X_arr = X.to_numpy()
        return self.rk.predict(X_arr[:, :-2], X_arr[:, -2:])

    def get_params(self, deep=True):
        return {
            'variogram_model': self.variogram_model,
            'nlags': self.nlags,
            'weight': self.weight,
            'regression_model': self.regression_model,
            'n_closest_points': self.n_closest_points,
            'exact_values': self.exact_values,
            'anisotropy_scaling': self.anisotropy_scaling,
            'anisotropy_angle': self.anisotropy_angle,
            'feature_names_in_': self.feature_names_in_,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
