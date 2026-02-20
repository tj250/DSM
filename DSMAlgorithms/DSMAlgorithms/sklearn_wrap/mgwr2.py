import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
import algorithms_config


'''
多尺度地理加权回归器（MGWR）的sklearn包装器
'''


class MGWRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 kernel='spherical',  # 'bisquare','gaussian','exponential'
                 search_method='golden_section',  # 'golden_section','interval'
                 criterion='AICc',  # 'AICc','AIC','BIC','CV'
                 ):
        self.kernel = kernel
        self.search_method = search_method
        self.criterion = criterion

    def __sklearn_tags__(self):  # 重要，否则堆叠过程中不认这个回归器
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    '''
    X:所有环境协变量的值
    y：预测变量的值
    '''

    def fit(self, train_X: pd.DataFrame, train_y: np.ndarray):
        coords_train = []
        for row in train_X.itertuples():
            coords_train.append([getattr(row, algorithms_config.CSV_GEOM_COL_X), getattr(row, algorithms_config.CSV_GEOM_COL_Y)])
        X = train_X.copy()
        X.pop(algorithms_config.CSV_GEOM_COL_X)
        X.pop(algorithms_config.CSV_GEOM_COL_Y)
        y = train_y.reshape((-1, 1))

        # multi=True,必须开启此参数，数值为True，才能是多尺度地理加权回归
        # fixed=False,代表使用自适应带宽（NN）而不是固定带宽
        selector = Sel_BW(coords_train, y, X.values, multi=True, fixed=False, kernel=self.kernel, constant=True)  # 带宽选择器
        # 需要特别注意：init_multi的值（初始带宽值）需要介于multi_bw_min和之间，否则出错
        # init_multi=100,  # 初始带宽默认为100米
        # multi_bw_min=[random.randint(10,100) for _ in range(len(X.columns))]+[0],       # 最后的[0]代表附加了截距项，最小带宽范围为10米到100米
        # multi_bw_max=[random.randint(100,5000) for _ in range(len(X.columns))]+[0],     # 最后的[0]代表附加了截距项,最大带宽范围为100米至5公里
        selector.search(search_method=self.search_method, criterion=self.criterion, max_iter_multi=20)  # 自动选择最佳带宽，最耗时的一步
        self.coords_train = np.array(coords_train)
        # 运行MGWR模型,根据数据进行拟合建模
        mgwr_model = MGWR(coords_train, y, X.values, selector, kernel=self.kernel, constant=True)  # 只有kernel可以作为可调参数，其它必须固定，第一个可调参数
        mgwr_results = mgwr_model.fit()
        # 提取回归系数
        coefficients = mgwr_results.params  # 局部回归系数
        self.intercepts = coefficients[:, 0]  # 截距项
        self.slopes = coefficients[:, 1:]  # 自变量系数
        return self

    def predict(self, X: pd.DataFrame):
        # 1、提取坐标
        coords_prediction = []
        for row in X.itertuples():  # 遍历每一行
            coords_prediction.append([getattr(row, algorithms_config.CSV_GEOM_COL_X), getattr(row, algorithms_config.CSV_GEOM_COL_Y)])
        coords_prediction = np.array(coords_prediction)
        X_new = X.copy()
        X_new.pop(algorithms_config.CSV_GEOM_COL_X)
        X_new.pop(algorithms_config.CSV_GEOM_COL_Y)
        X_arr = X_new.values
        # 逐个记录进行预测
        predicted_salt_content = []
        for i, coord in enumerate(coords_prediction):
            # 找到测试点最近的训练点索引
            distances = np.linalg.norm(self.coords_train - coord, axis=1)
            nearest_index = np.argmin(distances)
            # 使用最近训练点的回归系数进行预测
            intercept = self.intercepts[nearest_index]
            slope = self.slopes[nearest_index]
            prediction = intercept + np.dot(X_arr[i], slope)
            predicted_salt_content.append(prediction)
        return np.array(predicted_salt_content)

    def get_params(self, deep=True):
        return {
            'kernel': self.kernel,
            'search_method': self.search_method,
            'criterion': self.criterion,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
