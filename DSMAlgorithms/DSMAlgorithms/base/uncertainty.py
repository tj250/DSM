import numpy as np
import time
import pandas as pd
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsRegressor

# 不确定性估计及其评估的概念
# https://blog.csdn.net/xys430381_1/article/details/119531335

'''
对DSM进行不确定性制图和评估的类
'''


class Uncertainty:
    def __init__(self, model):
        self.prediction_model = model  # 预测模型
        self.preds = None  # 预测结果

    """
    X: 训练集特征 (n_samples, n_features)
    y: 训练集目标值 (n_samples,)
    X_new: 待预测特征 (n_new_samples, n_features),一般为属性图中的所有像素对应的环境协变量（特征）
    n_bootstraps: 重抽样次数
    bootstrap_scale: 每次从X中进行抽样的比例
    """

    def bootstrap_prediction_interval(self, X: pd.DataFrame, y: pd.Series, X_new: pd.DataFrame, bootstrap_times=100, bootstrap_scale=0.8):
        print(f"有放回抽样次数：{bootstrap_times}")
        X[y.name] = y.values  # 将y拼接到X
        # 存储Bootstrap预测结果
        bootstrap_preds = np.zeros((bootstrap_times, X_new.shape[0]))

        for i in range(bootstrap_times):
            # 抽样
            sampled_X = X.sample(frac=bootstrap_scale, replace=False, random_state=int(time.time()))  # 按80%比例抽样，每个样本只会重新入选一次
            y_boot = sampled_X.pop(y.name)  # 分离y
            # 拟合模型并预测
            try:
                self.prediction_model.model.fit(sampled_X, y_boot)  # 用预测模型内部保持的model进行拟合，X已经是经过各种处理的数值
                X_new_copy = X_new.copy()
                bootstrap_preds[i] = self.prediction_model.predict(X_new_copy)  # 用预测模型直接预测，在预测模型内部会对X_new进行必要的变换处理
            except Exception as e:
                print(f'模型拟合失败，错误信息：{e}')
                return False
            print(f'完成第{i+1}次抽样后的预测')
            print(f'最大值：{np.max(bootstrap_preds[i])}，最小值：{np.min(bootstrap_preds[i])}，')

        self.preds = bootstrap_preds
        return True

    '''
    获取预测标准差分布
    '''

    def get_std_error_dist(self):
        return np.std(self.preds, axis=0)

    '''
    获取分位数分布
    alpha:分位数值，如果为0.05，则分位数区间为(0,0.05,0.95,1)
    '''

    def get_percentile_dist(self, alpha=0.05):
        # 计算预测区间
        lower = np.percentile(self.preds, 100 * alpha, axis=0)
        upper = np.percentile(self.preds, 100 * (1 - alpha), axis=0)
        return lower, upper

    '''
    获取变异系数分布
    变异系数，又被称为“标准差率”或“离散系数”，是一个用于衡量观测值离散程度和变异程度的统计指标。
    在需求预测的语境中，它反映了各期数据的稳定性，因此也可以被视为一种“稳定系数”。该系数通过标准差与平均数的比值来计算
    '''

    def get_variation_coefficient_dist(self):
        return np.std(self.preds, axis=0) / np.mean(self.preds, axis=0)

    '''
    获取95%置信区间半宽分布:均值 +/- 1.96*标准差,返回值可用于生成95%置信区间的高值和低值图
    '''

    def get_confidence_interval_half_width_dist(self):
        return (np.mean(self.preds, axis=0) - 1.96 * np.std(self.preds, axis=0),
                np.mean(self.preds, axis=0) + 1.96 * np.std(self.preds, axis=0))

    '''
    计算PICP(Prediction Interval Coverage Probability，预测区间覆盖概率)：用来评估预测模型的准确性和可靠性的指标。它表示实际观测值落在预测区间内的概率。
    y:每个像素上的预测值（此处认为就是实际观测值）
    
    算法内部将计算y落在
    '''

    def calculate_PICP(self, y: np.array, alpha=0.05):
        lower, upper = self.get_percentile_dist(alpha)
        return float(np.mean((y > lower) & (y <= upper)) * 100)

    '''
    计算MPIW(Mean Prediction Interval Width，平均预测区间宽度)：衡量预测模型的精确度，值越小，越精确
    '''
    def calculate_MPIW(self):
        """计算平均预测区间宽度(MPIW)"""
        return float(np.mean(np.max(self.preds, axis=0) - np.min(self.preds, axis=0)))

# -------------------1、自助法(Bootstrap)集成
'''
通过重复采样生成多套预测表面，计算像素值的离散度
1、对原始样本进行有放回抽样生成Bootstrap数据集
2、用每套数据训练独立预测模型
3、统计所有模型在相同位置的预测值标准差

计算指定分位数的上下界数据
model:回归模型，必须具有fit方法和predict方法
train_dataset：建模的数据集
X_test：目标计算数据集
n_bootstraps:计算次数
'''


def compute_quantile_lower_upper(model, train_dataset: np.ndarray, X_target: np.ndarray, quantile=5,
                                 n_bootstraps=100) -> (np.ndarray, np.ndarray):
    predictions = np.zeros((n_bootstraps, len(X_target)))

    # 重复计算N次（每次计算中均会进行建模和预测）
    for i in range(n_bootstraps):
        # 1、重新划分样本
        X_train_boot, y_train_boot = resample(X_train, y_train)
        # 2、用重新划分的样本重新建模
        model.fit(X_train_boot, y_train_boot)
        # 3、用建立好的模型进行预测
        predictions[i] = model.predict(X_target)

    # 计算预测区间
    lower = np.percentile(predictions, quantile, axis=0)
    upper = np.percentile(predictions, 100 - quantile, axis=0)
    return upper, lower


# --------------------------------2、蒙特卡洛Dropout近似
# 采用贝叶斯神经网络或蒙特卡洛Dropout等方法，通过多次前向传播获得预测分布。该方法需要将土壤属性数据转换为深度学习兼容格式，并设计合适的损失函数

# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import KFold
#
# # 交叉验证不确定性估计
# kf = KFold(n_splits=5)
# predictions = np.zeros((len(X_test), kf.get_n_splits(X_train)))
#
# for i, (train_idx, test_idx) in enumerate(kf.split(X_train)):
#     knn = KNeighborsRegressor(n_neighbors=5)
#     knn.fit(X_train[train_idx], y_train[train_idx])
#     predictions[:, i] = knn.predict(X_test)
#
# # 计算预测标准差
# std_pred = np.std(predictions, axis=1)


# https://zhuanlan.zhihu.com/p/31898139608
# ------------------------------概率预测指标计算方法
# PICP: predicition interval coverage probability
# WS: winkler score
def evaluate_PICP_WS(y_pred_upper, y_pred_lower, y_test, confidence):
    # Reshape to 2D array for standardization
    y_pred_upper = np.reshape(y_pred_upper, (len(y_pred_upper), 1))
    y_pred_lower = np.reshape(y_pred_lower, (len(y_pred_lower), 1))

    y_pred_upper = sc_load.inverse_transform(y_pred_upper)
    y_pred_lower = sc_load.inverse_transform(y_pred_lower)
    y_test = sc_load.inverse_transform(y_test)

    # Ravel for ease of computation
    y_pred_upper = y_pred_upper.ravel()
    y_pred_lower = y_pred_lower.ravel()
    y_test = y_test.ravel()

    # Find out of bound indices for WS
    idx_oobl = np.where((y_test < y_pred_lower) > 0)
    idx_oobu = np.where((y_test > y_pred_upper) > 0)

    PICP = np.sum((y_test > y_pred_lower) & (y_test <= y_pred_upper)) / len(y_test) * 100
    WS = np.sum(np.sum(y_pred_upper - y_pred_lower) +
                np.sum(2 * (y_pred_lower[idx_oobl[0]] - y_test[idx_oobl[0]]) / confidence) +
                np.sum(2 * (y_test[idx_oobu[0]] - y_pred_upper[idx_oobu[0]]) / confidence)) / len(y_test)

    print("PICP of testing set: {:.2f}%".format(PICP))
    print("WS of testing set: {:.2f}".format(WS))

    return PICP, WS


#  ---------------残差克里金插值法：通过空间自相关分析量化预测误差的不确定性：
# ‌实施要点‌：
# 1、需先检验数据正态性（如pH值需对数转换）‌
# 2、变异函数参数（基台值、变程）直接影响不确定性量化结果
import numpy as np
from sklearn.model_selection import KFold
from pykrige.ok import OrdinaryKriging


def residual_kriging(X, y, model, grid_x, grid_y):
    """X: 样本坐标, y: 实测值, model: 预测模型, grid: 目标网格"""
    kf = KFold(n_splits=5)
    residuals = []

    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        residuals.extend(y[test_idx] - pred)

    # 克里金插值残差
    ok = OrdinaryKriging(X[:, 0], X[:, 1], np.array(residuals))
    z, ss = ok.execute('grid', grid_x, grid_y)
    return z, np.sqrt(ss)  # 返回残差和标准差
