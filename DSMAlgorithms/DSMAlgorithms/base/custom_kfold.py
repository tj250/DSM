import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
import pandas as pd
import algorithms_config

def adjusted_r2(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n_samples = len(y_true)
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

'''
分位数折叠分割
'''
class QuantileKFold:
    def __init__(self, n_splits=5, shuffle=True):
        self.n_splits = n_splits
        self.shuffle = shuffle

    def split(self, X, y, groups=None):
        # 按目标变量分位数分层
        quantiles = np.quantile(y, np.linspace(0, 1, self.n_splits + 1))
        strata = np.digitize(y, quantiles[:-1])

        # 标准K折划分（基于分层标签）
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
        return kf.split(X, strata)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

'''
顾及空间自相关性的折叠分割
有些情况下，我们想知道在特定组别上训练出来的模型是否能很好地泛化到未见过的组别上。为了衡量这一点，我们需要确保测试集中的所有样本都来自训练集中完全没有的组。
'''
class SpatialBlockKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y, groups):
        # 使用GroupKFold按空间块划分,GroupKFold 会保证同一个group的数据不会同时出现在训练集和测试集上。
        # 因为如果训练集中包含了每个group的几个样例，可能训练得到的模型能够足够灵活地从这些样例中学习到特征，在测试集上也会表现很好。
        # 但一旦遇到一个新的group它就会表现很差。
        gkf = GroupKFold(n_splits=self.n_splits)
        return gkf.split(X, y, groups)

'''
特定于limesoda数据集的交叉验证分割器
'''
class LimeSoDaFold:
    def __init__(self, folds_file_path:str):
        self.folds_file_path = folds_file_path

    def split(self, X, y, groups):
        # 使用GroupKFold按空间块划分,GroupKFold 会保证同一个group的数据不会同时出现在训练集和测试集上。
        # 因为如果训练集中包含了每个group的几个样例，可能训练得到的模型能够足够灵活地从这些样例中学习到特征，在测试集上也会表现很好。
        # 但一旦遇到一个新的group它就会表现很差。
        gkf = GroupKFold(n_splits=self.n_splits)
        return gkf.split(X, y, groups)


def custom_cv_split(folds_file_path):
    """完全自定义的分割生成器"""
    df = pd.read_csv(folds_file_path)
    indices = df.values
    for i in range(10):
        train_idx = np.where(indices != i+1)[0]
        test_idx = np.where(indices == i+1)[0]
        yield train_idx, test_idx

class LimeSodaSplit():
    """
    自定义KFold类，继承自sklearn的KFold
    支持自定义交叉验证折数和其他参数配置
    """

    def __init__(self, limesoda_sample_file:str, folds_file_path:str, feature_name:str, need_geom_column:bool=False):
        """
        初始化自定义KFold类

        参数:
        n_splits (int): 交叉验证折数，默认为5
        shuffle (bool): 是否在分割前打乱数据，默认为False
        random_state (int): 随机种子，用于重现结果
        """
        df = pd.read_csv(limesoda_sample_file)
        self.y = df.pop(feature_name)
        if need_geom_column and algorithms_config.CSV_GEOM_COL_X in df.columns:  # 需要保留坐标值列
            coord_x = df.pop(algorithms_config.CSV_GEOM_COL_X)
            coord_y = df.pop(algorithms_config.CSV_GEOM_COL_Y)
            df_concat = pd.concat([df, coord_x], axis=1)
            df = pd.concat([df_concat, coord_y], axis=1)
        else:
            if algorithms_config.CSV_GEOM_COL_X in df.columns:
                df.pop(algorithms_config.CSV_GEOM_COL_X)
            if algorithms_config.CSV_GEOM_COL_Y in df.columns:
                df.pop(algorithms_config.CSV_GEOM_COL_Y)
        self.X = df

        df_fold = pd.read_csv(folds_file_path)
        self.indices = df_fold.values

    def get_fold_n(self, indice):
        """
        生成训练集和测试集的索引

        参数:
        X: 特征数据
        y: 目标变量（可选）
        groups: 分组信息（可选）

        返回:
        迭代器，包含训练集和测试集的索引
        """
        # train_idx = np.where(self.indices != indice)[0]
        # test_idx = np.where(self.indices == indice)[0]
        X_train = self.X.loc[self.indices[self.X.index] != indice]
        X_test = self.X.loc[self.indices[self.X.index] == indice]
        y_train = self.y.loc[self.indices[self.y.index] != indice]
        y_test = self.y.loc[self.indices[self.y.index] == indice]
        return X_train, X_test, y_train, y_test

class CustomKFold(KFold):
    """
    自定义KFold类，继承自sklearn的KFold
    支持自定义交叉验证折数和其他参数配置
    """

    def __init__(self, folds_file_path, n_splits=10, shuffle=False, random_state=None):
        """
        初始化自定义KFold类

        参数:
        n_splits (int): 交叉验证折数，默认为5
        shuffle (bool): 是否在分割前打乱数据，默认为False
        random_state (int): 随机种子，用于重现结果
        """
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.folds_file_path = folds_file_path

    def split(self, X, y=None, groups=None):
        """
        生成训练集和测试集的索引

        参数:
        X: 特征数据
        y: 目标变量（可选）
        groups: 分组信息（可选）

        返回:
        迭代器，包含训练集和测试集的索引
        """
        return custom_cv_split(self.folds_file_path)

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        返回交叉验证的折数
        """
        return self.n_splits