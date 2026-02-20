import time
import pandas as pd
import numpy as np
import algorithms_config
from xgboost import XGBRegressor
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from bayes_opt import BayesianOptimization
from DSMAlgorithms.base.save_model import save_regressor_result
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit
from DSMAlgorithms.sklearn_wrap.custom_regressor import CustomRegressor
from data.data_dealer import mean_encode_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet

'''
自定义模型的包装器（仅用于单一模型的建模，不会在stacking时使用）
'''


class CustomModelWrap(DSMBuildModel):

    def __init__(self, model_class, algorithms_name: str, model_args: dict[str, tuple], syn_eval_args: str,
                 nested_args_name: str, nested_args_types: dict[str, str], special_enum_conversion_args: dict[str,dict],
                 enum_conversion_args: dict, complex_lamda_args: dict, model_kwargs: dict[str, tuple], prop_name: str,
                 category_vars: dict, need_geometry: bool):
        super().__init__(prop_name, category_vars)
        self.model_class = model_class  # 记录下模型的类，用于后续调用其构造方法、fit方法、predict方法
        self.model_args = model_args  # 记录下自定义模型的构造器的位置参数
        self.model_syn_eval_args = syn_eval_args  # 模型的动态评估参数名称列表
        self.model_kwargs = model_kwargs  # 记录下自定义模型的构造器的关键字参数
        self.model_enum_conversion_args = enum_conversion_args
        self.model_complex_lamda_args = complex_lamda_args
        self.model_nested_args_name = nested_args_name
        self.model_nested_args_types = nested_args_types
        self.model_nested_enum_args = special_enum_conversion_args
        self.algorithms_name = algorithms_name
        self.need_geometry = need_geometry

    """
    生成自定义模型的回归模型
    algorithms_id:算法的ID
    train_X:训练集的X
    train_y:训练集的y
    test_X:验证集的X
    test_y:验证集的y
    """

    def build(self, algorithms_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict) -> bool:
        # 对于含有geometry的情形，将原有给的geometry列拆分为两列
        if self.need_geometry:
            train_X[algorithms_config.CSV_GEOM_COL_X] = train_X[algorithms_config.DF_GEOM_COL].apply(
                lambda coord: coord.x)
            train_X[algorithms_config.CSV_GEOM_COL_Y] = train_X[algorithms_config.DF_GEOM_COL].apply(
                lambda coord: coord.y)
            train_X.pop(algorithms_config.DF_GEOM_COL)
        train_X_encoder, mean_encoder = mean_encode_dataset(self.category_vars, train_X, train_y)
        t1 = time.time()

        # 定义目标函数
        def model_evaluate(**kwargs):
            nest_args = {}
            for key, value in kwargs.items():               # 0-提取所有nested参数
                if key.startswith('nested_'):
                    if self.model_nested_args_types[key] == 'int':
                        nest_args[key] = round(value)
                    elif self.model_nested_args_types[key] == 'float':
                        nest_args[key] = float(value)
                    elif self.model_nested_args_types[key] == 'bool':
                        nest_args[key] = round(value) == 1
                    elif self.model_nested_args_types[key] == 'enum':
                        nest_args[key] = f"'{self.model_nested_enum_args[key][str(round(value))]}'" # 只支持参数类型为字符串
                    else:
                        nest_args[key] = value
            kwargs = {k: v for k, v in kwargs.items() if k not in nest_args}
            new_kwargs = self.model_args.copy()             # 1-固定参数
            new_kwargs['model_class'] = self.model_class    # 2-特意增加一个模型类参数，用于传递到CustomRegressor内部
            new_kwargs |= kwargs                            # 3-追加可调参数
            if self.model_syn_eval_args is not None:        # 4-对固定参数中的动态评估参数进行处理
                dyn_args_names = self.model_syn_eval_args.split(',')
                if len(nest_args) > 0:  # 如果存在嵌套参数，则将参数值格式化进入eval参数中
                    new_kwargs[self.model_nested_args_name] = new_kwargs[self.model_nested_args_name].format(**nest_args)
                for dyn_arg_name in dyn_args_names:
                    new_kwargs[dyn_arg_name] = eval(new_kwargs[dyn_arg_name])  # 执行动态的eval
            if self.model_enum_conversion_args is not None:
                for key, value in self.model_enum_conversion_args.items():  # 5-对可调参数中的枚举值进行转换
                    if key in new_kwargs:
                        new_kwargs[key] = value[str(round(new_kwargs[key]))]
            if self.model_complex_lamda_args is not None:
                for key, value in self.model_complex_lamda_args.items():  # 6-对需lamda处理的参数进行转换
                    if key in new_kwargs:
                        lamda_func = eval(value)
                        new_kwargs[key] = lamda_func(new_kwargs[key])
            custom_regressor = CustomRegressor(**new_kwargs)
            if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                limesoda_split = LimeSodaSplit(self.limesoda_sample_file,
                                               algorithms_config.LIMESODA_FOLDS_FILE_PATH,
                                               self.prop_name, self.need_geometry)
                y_true_all = []
                y_pred_all = []
                for i in range(algorithms_config.KFOLD):
                    X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i + 1)
                    custom_regressor.fit(X_train, y_train)
                    y_pred = custom_regressor.predict(X_val)
                    y_true_all.extend(y_val.values)
                    y_pred_all.extend(y_pred)
                y_true_all = np.array(y_true_all)
                y_pred_all = np.array(y_pred_all)
                score = r2_score(y_true_all, y_pred_all)
            else:  # 采用分折的平均R2来计算
                scorer = make_scorer(
                adjusted_r2,
                n_features=len(train_X.columns),  # 需要排除两个坐标值列
                greater_is_better=True
                )
                score = np.mean(
                    cross_val_score(custom_regressor, train_X_encoder, train_y, cv=algorithms_config.KFOLD,
                                    scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2'))
            return score

        # 创建贝叶斯优化对象
        optimizer = BayesianOptimization(
            f=model_evaluate,
            pbounds=self.model_kwargs,
            random_state=algorithms_config.RANDOM_STATE,
            verbose=1
        )

        # 进行贝叶斯优化，需要捕获异常，优于优化过程中，可能因为个别模型的数据导致奇异矩阵等问题而导致建模失败
        try:
            optimizer.maximize(
                init_points=algorithms_config.BAYES_INIT_POINTS,
                n_iter=algorithms_config.BYEAS_ITER_TIMES
            )
        except Exception as e:
            print(f'贝叶斯优化建模失败，错误信息：{e}')
            return False

        # 使用最优参数创建最优模型
        best_model_params = optimizer.max['params']
        for key, value in best_model_params.items():
            if isinstance(value, int):
                best_model_params[key] = int(best_model_params[key])

        nest_args = {}
        for key, value in best_model_params.items():  # 0-提取所有nested参数
            if key.startswith('nested_'):
                if self.model_nested_args_types[key] == 'int':
                    nest_args[key] = round(value)
                elif self.model_nested_args_types[key] == 'float':
                    nest_args[key] = float(value)
                elif self.model_nested_args_types[key] == 'bool':
                    nest_args[key] = round(value) == 1
                elif self.model_nested_args_types[key] == 'enum':
                    nest_args[key] = f"'{self.model_nested_enum_args[key][str(round(value))]}'"  # 只支持参数类型为字符串
                else:
                    nest_args[key] = value
        best_model_params = {k: v for k, v in best_model_params.items() if k not in nest_args}
        new_best_model_kwargs = self.model_args.copy()
        new_best_model_kwargs['model_class'] = self.model_class
        new_best_model_kwargs |= best_model_params
        if self.model_syn_eval_args is not None:  # 存在需要动态执行的参数
            dyn_args_names = self.model_syn_eval_args.split(',')
            if len(nest_args) > 0:  # 如果存在嵌套参数，则将参数值格式化进入eval参数中
                new_best_model_kwargs[self.model_nested_args_name] = new_best_model_kwargs[self.model_nested_args_name].format(**nest_args)
            for dyn_arg_name in dyn_args_names:
                new_best_model_kwargs[dyn_arg_name] = eval(new_best_model_kwargs[dyn_arg_name])  # 执行动态的eval
        if self.model_enum_conversion_args is not None:
            for key, value in self.model_enum_conversion_args.items():  # 5-对可调参数中的枚举值进行转换
                if key in new_best_model_kwargs:
                    new_best_model_kwargs[key] = value[str(round(new_best_model_kwargs[key]))]
        if self.model_complex_lamda_args is not None:
            for key, value in self.model_complex_lamda_args.items():  # 6-对需lamda处理的参数进行转换
                if key in new_best_model_kwargs:
                    lamda_func = eval(value)
                    new_best_model_kwargs[key] = lamda_func(new_best_model_kwargs[key])
        best_model = CustomRegressor(**new_best_model_kwargs)

        # 训练
        best_model.fit(train_X_encoder, train_y)
        best_model_score = optimizer.max['target']

        t2 = time.time()
        # print("优化用时：{}秒".format(t2 - t1))
        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_model.predict(train_X_encoder)
            save_regressor_result(algorithms_id, self.algorithms_name, best_model, new_best_model_kwargs,
                                  best_model_score,
                                  train_y, train_predictions, train_X.columns, zscore_normalize, mean_encoder)
        else:
            # geometry列变为两列
            if self.need_geometry:
                test_X[algorithms_config.CSV_GEOM_COL_X] = test_X[algorithms_config.DF_GEOM_COL].apply(
                    lambda coord: coord.x)
                test_X[algorithms_config.CSV_GEOM_COL_Y] = test_X[algorithms_config.DF_GEOM_COL].apply(
                    lambda coord: coord.y)
                test_X.pop(algorithms_config.DF_GEOM_COL)
            test_predictions = best_model.predict(test_X)
            save_regressor_result(algorithms_id, self.algorithms_name, best_model, new_best_model_kwargs,
                                  best_model_score,
                                  test_y, test_predictions, train_X.columns, zscore_normalize, mean_encoder)
        return True
