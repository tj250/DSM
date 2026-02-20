import pandas as pd
from abc import ABC, abstractmethod

"""
模型构建器包装器基类,该类为建模过程提供了一个公共基类，包含了多个公用的属性以及一个build方法
prop_name:预测的属性名称
csv_file:样点和协变量信息所在的文件
category_vars:协变量当中的类别型变量
"""


class DSMBuildModel(ABC):
    """
    prop_name:建模的属性名
    category_vars:类别信息字典
    max_iter:最大迭代次数
    """

    def __init__(self, prop_name: str, category_vars: dict):
        self.prop_name = prop_name  # 待预测的属性名
        self.category_vars = category_vars

    def set_limesoda_sample_file(self, file_name: str):
        self.limesoda_sample_file = file_name

    '''
    所有子类都必须实现的建模方法
    algorithms_id：算法ID
    train_X：训练集的环境协变量数据
    train_y：训练集的预测变量
    test_X：测试集的环境协变量数据
    test_y：测试集的预测变量
    zscore_normalize：前述数据的z-score标准化或min-max归一化信息
    
    返回值，建模是否成功。可能会优于如下原因，导致建模失败：
    1、当采用小数据集时，由于建模内部会进行交叉验证，一些算法（比如xgboost）要求验证集的记录数要高于特征数
    2、当模型内部预测时，由于数据有异常值（包括自变量和因变量），产生奇异矩阵，会导致建模失败
    '''

    @abstractmethod
    def build(self, algorithms_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict) -> bool:
        pass
