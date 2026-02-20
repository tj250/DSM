import numpy as np
import pandas as pd
from scipy import stats

'''
封装了对数据中的异常值进行检测的各类方法

推荐使用dixon_grubbs_test,可以针对不同样本量的数据
'''


class DataExceptionTest():
    """
    对异常值进行检测
    data:待检测数据
    is_normality:数据是否服从正态分布
    """

    @staticmethod
    def exception_test(data: list, is_normality=True) -> np.ndarray:
        if len(data) < 30:
            return DataExceptionTest.iqr_test(data)  # 样本数未超过30
        else:
            if is_normality:
                return DataExceptionTest.three_sigma_test(data) # 样本数等于或超过30且整体符合整体分布，则采用3σ准则检测
            else:
                return DataExceptionTest.iqr_test(data)

    '''
    使用四分位距法（IQR）检测数据的异常值,异常值为小于Q1-1.5×IQR或大于Q3+1.5×IQR的数据点，不依赖分布假设‌
    '''

    @staticmethod
    def iqr_test(data):
        # 计算四分位数
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        # 定义异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 检测异常值
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return np.array(outliers), lower_bound, upper_bound

    '''
    使用拉依达准则（3σ准则）检测数据的异常值,适用于正态分布且样本量≥30的数据，若数据点与均值的绝对偏差超过3倍标准差则判定为异常值‌
    '''

    @staticmethod
    def three_sigma_test(data):
        # 计算统计量
        mean = np.mean(data)
        std = np.std(data)

        # 定义异常值阈值
        threshold = 3 * std

        # 检测异常值
        outliers = [x for x in data if abs(x - mean) > threshold]
        return np.array(outliers), mean - threshold, mean + threshold


