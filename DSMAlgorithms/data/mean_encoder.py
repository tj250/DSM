import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import product


class MeanEncoder:
    def __init__(self, categorical_features: list[str], n_splits=5, target_type='classification',
                 prior_weight_func=None,
                 learned_stats={}):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode

        :param n_splits: the number of splits used in mean encoding

        :param target_type: str, 'regression' or 'classification'

        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = learned_stats

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        # col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg([('mean', 'mean'), ('beta', 'size')])
        # col_avg_y = X_train.groupby(variable)['pred_temp'].agg(['mean', 'size']).rename(
        #     columns={'mean': 'mean', 'size': 'beta'}) # old style
        col_avg_y = X_train.groupby(variable, observed=True)['pred_temp'].agg(['mean', 'size']).rename(
            columns={'mean': 'mean', 'size': 'beta'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test_temp = X_test.join(col_avg_y, on=variable)
        nf_test_temp[variable] = nf_test_temp[variable].astype('object')  # category列转换为object列,否则调用fillna出错
        with pd.option_context("future.no_silent_downcasting", True):
            nf_test = nf_test_temp.fillna(prior)[nf_name].infer_objects(copy=False).values
        nf_test_temp[variable] = nf_test_temp[variable].astype('category')
        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        拟合并变换数据
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        if len(self.categorical_features) == 0:  # 如果不包含类别特征，直接将X返回
            return X
        origin_columns_names = X.columns.tolist()
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)
        rename_dict = {}
        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan  # 添加新的列
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
                X_new.pop(variable)  # 对原始算法的更改，删除原有的列
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                rename_dict[nf_name] = variable
                X_new.loc[:, nf_name] = np.nan  # 添加新的列
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
                X_new.pop(variable)  # 对原始算法的更改，删除原有的列

        X_new.rename(columns=rename_dict, inplace=True)  # 重命名为原来的名称
        X_new = X_new.reindex(columns=origin_columns_names)  # 按照初始的列顺序排列
        return X_new

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        if len(self.categorical_features) == 0:  # 如果不包含类别特征，直接将X返回
            return X
        origin_columns_names = X.columns.tolist()
        X_new = X.copy()
        rename_dict = {}
        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
                X_new.pop(variable)  # 对原始算法的更改，删除原有的列
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                rename_dict[nf_name] = variable
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    temp = X_new[[variable]].join(col_avg_y, on=variable)
                    temp[variable] = temp[variable].astype('object')  # category列转换为object列,否则调用fillna出错
                    with pd.option_context("future.no_silent_downcasting", True):
                        temp2 = temp.fillna(prior)[nf_name].infer_objects(copy=False).values
                    X_new[nf_name] += temp2
                    # X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                    #     nf_name] # 原有代码，会出错，由于X_new中的variable类型为Categorical，fillna会报错，需修改为object
                X_new[nf_name] /= self.n_splits
                X_new.pop(variable)  # 对原始算法的更改，删除原有的列

        X_new.rename(columns=rename_dict, inplace=True)  # 重命名为原来的名称
        X_new = X_new.reindex(columns=origin_columns_names)  # 按照初始的列顺序排列
        return X_new

    '''
    将MeanEncoder对象序列化为字典
    '''

    def serialize(self) -> dict:
        return {'categorical_features': self.categorical_features,
                'learned_stats': self.learned_stats,
                'n_splits': self.n_splits,
                'target_type': self.target_type}

    '''
    将字典反序列化为MeanEncoder对象
    '''

    @staticmethod
    def Deserialize(serialization: dict):
        return MeanEncoder(serialization['categorical_features'],
                           n_splits=serialization['n_splits'],
                           target_type=serialization['target_type'],
                           learned_stats=serialization['learned_stats'])
