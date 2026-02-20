import time
import os
import numpy as np
import pandas as pd
import algorithms_config
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, ElasticNet, TweedieRegressor, GammaRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from DSMAlgorithms.base.stacking_wrapper import (RegressionKrigingRegressorStackingWrapper, CoKrigingRegressorStackingWrapper,
                                                 TweedieRegressorStackingWrapper,CustomModelStackingWrapper,
                                                 ElasticNetStackingWrapper, KNeighborsRegressorStackingWrapper,
                                                 MLPRegressorStackingWrapper, RandomForestRegressorStackingWrapper,
                                                 SVRStackingWrapper, XGBRegressorStackingWrapper,
                                                 PLSRegressionStackingWrapper, MGWRStackingWrapper)
from DSMAlgorithms.base.save_model import save_regressor_result
from data.data_dealer import mean_encode_dataset
from db_access.algorithms_parameters import AlgorithmsDataAccess
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
from DSMAlgorithms.base.base_data_structure import AlgorithmsType, algorithms_dict, DataDistributionType
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit
from utility import load_module_from_file,get_sklearn_style_class

'''
堆叠包装器(用于建模)
'''


class StackingWrap(DSMBuildModel):
    """
    构造对象进行建模
    prop_name:预测的属性名称
    csv_file:样点和协变量信息所在的文件
    category_vars:协变量当中的类别型变量
    models:堆叠的模型列表
    """

    def __init__(self, prop_name: str, category_vars: dict, algorithms_name: list[str]):
        super().__init__(prop_name, category_vars)
        self.algorithms_names = algorithms_name

    '''
    将可变参数中的值转换为特定算法的参数
    algorithms_type：算法类型
    '''

    def prepare_algorithms_params(self, algorithms_name: str, **kwargs) -> dict:
        algorithms_params = {}
        if algorithms_name == AlgorithmsType.XGBR.value:
            algorithms_params['n_estimators'] = int(kwargs['xgbr_n_estimators'])
            algorithms_params['max_depth'] = int(kwargs['xgbr_max_depth'])
            algorithms_params['subsample'] = kwargs['xgbr_subsample']
            algorithms_params['colsample_bytree'] = kwargs['xgbr_colsample_bytree']
            algorithms_params['gamma'] = kwargs['xgbr_gamma']
            algorithms_params['learning_rate'] = kwargs['xgbr_learning_rate']
            algorithms_params['reg_lambda'] = int(kwargs['xgbr_reg_lambda'])
            algorithms_params['min_child_weight'] = int(kwargs['xgbr_min_child_weight'])
        elif algorithms_name == AlgorithmsType.RFR.value:
            max_depth_use_none = True if int(kwargs['rfr_max_depth_use_none']) == 1 else False
            if int(kwargs['rfr_max_features_type']) == 1:
                max_features_actual = 'sqrt'
            elif int(kwargs['rfr_max_features_type']) == 2:
                max_features_actual = 'log2'
            elif int(kwargs['rfr_max_features_type']) == 3:
                max_features_actual = None
            else:
                max_features_actual = kwargs['rfr_max_features']  # 百分比
            algorithms_params['n_estimators'] = int(kwargs['rfr_n_estimators'])
            algorithms_params['max_depth'] = None if max_depth_use_none else int(kwargs['rfr_max_depth'])
            algorithms_params['min_samples_split'] = int(kwargs['rfr_min_samples_split'])
            algorithms_params['min_samples_leaf'] = int(kwargs['rfr_min_samples_leaf'])
            algorithms_params['max_features'] = max_features_actual
            algorithms_params['bootstrap'] = True if int(kwargs['rfr_bootstrap']) == 1 else False
            algorithms_params['max_samples'] = kwargs['rfr_max_samples'] if int(kwargs['rfr_bootstrap']) == 1 else None
        elif algorithms_name == AlgorithmsType.MLP.value:
            param_dist = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],  # 网络结构
                'activation': ['relu', 'tanh'],  # 激活函数
                # 优化器,‘adam’在相对较大的数据集上效果比较好（几千个样本或者更多），对小数据集来说，lbfgs收敛更快效果也更好
                'solver': ['sgd', 'adam', 'lbfgs'],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],  # 学习率,用于权重更新,只有当solver为’sgd’时使用
            }
            algorithms_params['hidden_layer_sizes'] = param_dist['hidden_layer_sizes'][
                int(kwargs['mlp_hidden_layer_type']) - 1]
            algorithms_params['activation'] = param_dist['activation'][int(kwargs['mlp_activation_type']) - 1]
            algorithms_params['solver'] = param_dist['solver'][int(kwargs['mlp_solver_type']) - 1]
            algorithms_params['alpha'] = kwargs['mlp_alpha']
            algorithms_params['learning_rate'] = param_dist['learning_rate'][int(kwargs['mlp_learning_rate_type']) - 1]
            algorithms_params['learning_rate_init'] = kwargs['mlp_learning_rate_init']
        elif algorithms_name == AlgorithmsType.PLSR.value:
            algorithms_params['n_components'] = int(kwargs['plsr_n_components'])
        elif algorithms_name == AlgorithmsType.GLM.value:
            param_dist = {
                'solver': ['lbfgs', 'newton-cholesky'],  # 优化算法
            }
            algorithms_params['alpha'] = kwargs['glm_alpha']
            algorithms_params['fit_intercept'] = True if int(kwargs['glm_fit_intercept']) == 1 else False
            algorithms_params['solver'] = param_dist['solver'][int(kwargs['glm_solver']) - 1]
            algorithms_params['power'] = kwargs['glm_power']
        elif algorithms_name == AlgorithmsType.EN.value:
            algorithms_params['alpha'] = kwargs['en_alpha']
            algorithms_params['l1_ratio'] = kwargs['en_l1_ratio']
            algorithms_params['tol'] = kwargs['en_tol']
        elif algorithms_name == AlgorithmsType.MGWR.value:
            param_dist = {
                'kernel': ['bisquare', 'gaussian', 'exponential'],
                'search_method': ['golden_section', 'interval'],
                'criterion': ['AICc','AIC','BIC','CV'],
            }
            algorithms_params['kernel'] = param_dist['kernel'][int(kwargs['mgwr_kernel']) - 1]
            algorithms_params['search_method'] = param_dist['search_method'][int(kwargs['mgwr_search_method']) - 1]
            algorithms_params['criterion'] = param_dist['criterion'][int(kwargs['mgwr_criterion']) - 1]
        elif algorithms_name == AlgorithmsType.SVR.value:
            param_dist = {'kernel': ['linear', 'rbf', 'poly']}  # 核函数类型
            algorithms_params['kernel'] = param_dist['kernel'][int(kwargs['svr_kernel']) - 1]
            algorithms_params['C'] = kwargs['svr_C']
            algorithms_params['gamma'] = kwargs['svr_gamma']
            algorithms_params['epsilon'] = kwargs['svr_epsilon']
            algorithms_params['degree'] = int(kwargs['svr_degree'])
            algorithms_params['coef0'] = kwargs['svr_coef0']
        elif algorithms_name == AlgorithmsType.KNR.value:
            param_dist = {
                'weights': ['uniform', 'distance'],  # 权重计算方式
                'algorithm': ['brute', 'kd_tree', 'ball_tree', 'auto'],  # 算法类型
            }
            algorithms_params['n_neighbors'] = int(kwargs['knr_n_neighbors'])
            algorithms_params['weights'] = param_dist['weights'][int(kwargs['knr_weights']) - 1]
            algorithms_params['algorithm'] = param_dist['algorithm'][int(kwargs['knr_algorithm']) - 1]
            algorithms_params['p'] = kwargs['knr_p']
        elif algorithms_name == AlgorithmsType.RK.value:
            param_dist = {
                # 用于回归的模型，如RandomForestRegressor，LinearRegression，SVR
                'regression_model': ['rfr', 'xgbr', 'svr', 'en', 'glm', 'plsr', 'knr'],
                # 残差变异函数类型:linear, power, gaussian, spherical, exponential, hole-effect等，默认的为linear模型,hole-effect仅针对于一维问题。
                'variogram_model': ['gaussian', 'spherical', 'exponential'],
                # 'variogram_parameters'：变异函数参数，根据所选方差模型确定。不提供的话则采用“软”（soft）L1范式最小化方案。
                # 特别注意：不同变异函数的参数不同，参数必须完整且按照顺序指定
                # 参考：https://geostat-framework.readthedocs.io/projects/pykrige/en/latest/generated/pykrige.ok.OrdinaryKriging.html
                # linear
                # {'slope': slope, 'nugget': nugget}
                # power
                #    {'scale': scale, 'exponent': exponent, 'nugget': nugget}
                # gaussian, spherical, exponential and hole-effect:
                # {'sill': s, 'range': r, 'nugget': n}
                # OR
                #    {'psill': p, 'range': r, 'nugget': n}
            }
            algorithms_params['regression_model_type'] = int(kwargs['rk_regression_model'])
            algorithms_params['variogram_model'] = param_dist['variogram_model'][int(kwargs['rk_variogram_model']) - 1]
            algorithms_params['nlags'] = int(kwargs['rk_nlags'])
            algorithms_params['weight'] = True if int(kwargs['rk_weight']) == 1 else False
            algorithms_params['exact_values'] = True if int(kwargs['rk_exact_values']) == 1 else False
            algorithms_params['n_closest_points'] = int(kwargs['rk_n_closest_points'])
            algorithms_params['anisotropy_scaling'] = (kwargs['rk_anisotropy_scaling'], kwargs['rk_anisotropy_scaling'])
            algorithms_params['anisotropy_angle'] = (
            kwargs['rk_anisotropy_angle'], kwargs['rk_anisotropy_angle'], kwargs['rk_anisotropy_angle'])
        elif algorithms_name == AlgorithmsType.CK.value:
            param_dist = {
                'variogram_model': ['linear', 'power', 'gaussian', 'spherical', 'exponential'],  # 指定要使用的变异函数模型
                'pseudo_inv_type': ['pinv', 'pinvh']  # 计算伪逆矩阵的算法
            }
            algorithms_params['variogram_model'] = param_dist['variogram_model'][int(kwargs['ck_variogram_model']) - 1]
            algorithms_params['nlags'] = int(kwargs['ck_nlags'])
            algorithms_params['weight'] = True if int(kwargs['ck_weight']) == 1 else False
            algorithms_params['exact_values'] = True if int(kwargs['ck_exact_values']) == 1 else False
            algorithms_params['anisotropy_scaling'] = kwargs['ck_anisotropy_scaling']
            algorithms_params['anisotropy_angle'] = kwargs['ck_anisotropy_angle']
            # algorithms_params['pseudo_inv'] = True if int(kwargs['ck_pseudo_inv']) == 1 else False
            # algorithms_params['pseudo_inv_type'] = param_dist['pseudo_inv_type'][int(kwargs['ck_pseudo_inv_type']) - 1]
        else: # 自定义模型
            for key,value in kwargs.items():
                if key.startswith(algorithms_name):
                    algorithms_params[key[len(algorithms_name)+1:]] = kwargs[key]

        return algorithms_params

    '''
    构建堆叠模型
    '''

    def build(self, stacking_algorithms_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict, data_distribution_type: DataDistributionType) -> bool:
        # 将原有给的geometry列拆分为两列
        if algorithms_config.DF_GEOM_COL in train_X.columns:
            train_X[algorithms_config.CSV_GEOM_COL_X] = train_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.x)
            train_X[algorithms_config.CSV_GEOM_COL_Y] = train_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.y)
            train_X.pop(algorithms_config.DF_GEOM_COL)
        # 在建模阶段，应根据样点数据生成一个平均编码器，不同的模型，包括未来预测的数据应统一采用此平均编码器
        _, mean_encoder = mean_encode_dataset(self.category_vars, train_X, train_y)
        # 定义元模型
        meta_model = Ridge()  # meta_model = LinearRegression()
        model_type_list = [AlgorithmsType.RFR, AlgorithmsType.XGBR, AlgorithmsType.SVR, AlgorithmsType.EN,
                           AlgorithmsType.GLM, AlgorithmsType.PLSR, AlgorithmsType.KNR]
        t1 = time.time()
        self.bayesian_iter_times = 0

        # 定义目标函数
        def stacking_evaluate(**kwargs):
            model_params = {}
            for algorithms_name in self.algorithms_names:
                model_params[algorithms_name] = self.prepare_algorithms_params(algorithms_name, **kwargs)

            train_base_models = []
            for algorithms_name, parameters in model_params.items():
                if algorithms_name == AlgorithmsType.XGBR.value:
                    train_base_models.append((algorithms_name, XGBRegressorStackingWrapper(**parameters,
                                                                                                 enable_categorical=True,
                                                                                                 random_state=algorithms_config.RANDOM_STATE)))
                elif algorithms_name == AlgorithmsType.RFR.value:
                    parameters['mean_encoder'] = mean_encoder
                    train_base_models.append(
                        (algorithms_name,
                         RandomForestRegressorStackingWrapper(**parameters, random_state=algorithms_config.RANDOM_STATE)))
                elif algorithms_name == AlgorithmsType.MLP.value:
                    parameters['mean_encoder'] = mean_encoder
                    train_base_models.append(
                        (algorithms_name,
                         MLPRegressorStackingWrapper(**parameters, early_stopping=True, max_iter=algorithms_config.MLP_MAX_ITER,
                                                     random_state=algorithms_config.RANDOM_STATE)))
                elif algorithms_name == AlgorithmsType.PLSR.value:
                    parameters['mean_encoder'] = mean_encoder
                    parameters['max_iter'] = algorithms_config.PLSR_MAX_ITER
                    train_base_models.append(
                        (algorithms_name, PLSRegressionStackingWrapper(**parameters)))
                elif algorithms_name == AlgorithmsType.GLM.value:
                    parameters['mean_encoder'] = mean_encoder
                    parameters['max_iter'] = algorithms_config.GLM_MAX_ITER
                    train_base_models.append(
                            (algorithms_name, TweedieRegressorStackingWrapper(**parameters)))
                elif algorithms_name == AlgorithmsType.EN.value:
                    parameters['mean_encoder'] = mean_encoder
                    parameters['max_iter'] = algorithms_config.EN_MAX_ITER
                    train_base_models.append(
                        (algorithms_name, ElasticNetStackingWrapper(**parameters)))
                elif algorithms_name == AlgorithmsType.SVR.value:
                    parameters['mean_encoder'] = mean_encoder
                    parameters['max_iter'] = algorithms_config.SVR_MAX_ITER
                    train_base_models.append((algorithms_name, SVRStackingWrapper(**parameters)))
                elif algorithms_name == AlgorithmsType.KNR.value:
                    parameters['mean_encoder'] = mean_encoder
                    train_base_models.append(
                        (algorithms_name, KNeighborsRegressorStackingWrapper(**parameters)))
                elif algorithms_name == AlgorithmsType.RK.value:
                    regression_model_type = model_type_list[int(parameters['regression_model_type']) - 1]  # 获取回归模型的名称
                    del parameters['regression_model_type']
                    reg_params = self.prepare_algorithms_params(regression_model_type.value, **kwargs)
                    if regression_model_type == AlgorithmsType.RFR:
                        reg_model = RandomForestRegressor(**reg_params, random_state=algorithms_config.RANDOM_STATE)
                    elif regression_model_type == AlgorithmsType.XGBR:
                        reg_model = XGBRegressor(**reg_params, enable_categorical=True,
                                                 random_state=algorithms_config.RANDOM_STATE)
                    elif regression_model_type == AlgorithmsType.SVR:
                        reg_model = SVR(**reg_params, max_iter=algorithms_config.SVR_MAX_ITER)
                    elif regression_model_type == AlgorithmsType.EN:
                        reg_model = ElasticNet(**reg_params, max_iter=algorithms_config.EN_MAX_ITER)
                    elif regression_model_type == AlgorithmsType.GLM:
                        reg_model = TweedieRegressor(**reg_params, max_iter=algorithms_config.GLM_MAX_ITER)
                    elif regression_model_type == AlgorithmsType.PLSR:
                        reg_model = PLSRegression(**reg_params, max_iter=algorithms_config.PLSR_MAX_ITER)
                    elif regression_model_type == AlgorithmsType.KNR:
                        reg_model = KNeighborsRegressor(**reg_params)

                    parameters['regression_model'] = reg_model  # 将原来的int值替换为回归模型对象
                    parameters['mean_encoder'] = mean_encoder
                    rk = RegressionKrigingRegressorStackingWrapper(**parameters)
                    train_base_models.append((algorithms_name, rk))
                elif algorithms_name == AlgorithmsType.CK.value:
                    parameters['mean_encoder'] = mean_encoder
                    train_base_models.append(
                        (algorithms_name, CoKrigingRegressorStackingWrapper(**parameters, )))
                elif algorithms_name == AlgorithmsType.MGWR.value:
                    parameters['mean_encoder'] = mean_encoder
                    train_base_models.append(
                        (algorithms_name, MGWRStackingWrapper(**parameters, )))
                else:
                    parameters['mean_encoder'] = mean_encoder
                    # 获取模型的类
                    custom_model_info = AlgorithmsDataAccess.query_custom_model_info(algorithms_name)
                    py_file = os.path.join(os.getcwd(), algorithms_name + ".py")
                    AlgorithmsDataAccess.save_as_pyfile(custom_model_info.class_code, py_file, True)  # 将代码保持为文件
                    loaded_module = load_module_from_file(py_file)  # 从.py文件装载python模块
                    model_class = get_sklearn_style_class(loaded_module)  # 从模块中解析出第一个包含fit和method方法的类名称
                    algorithms_pbounds = AlgorithmsDataAccess.retrival_custom_model_pbounds(
                        algorithms_name)  # 获取自定义模型的关键字参数的边界
                    algorithms_nested_args_types = AlgorithmsDataAccess.retrival_custom_model_nested_args_types(
                        algorithms_name)
                    nest_args = {}
                    for key, value in parameters.items():  # 0-提取所有nested参数
                        if key.startswith('nested_'):
                            if algorithms_nested_args_types[key] == 'int':
                                nest_args[key] = round(value)
                            elif algorithms_nested_args_types[key] == 'float':
                                nest_args[key] = float(value)
                            elif algorithms_nested_args_types[key] == 'bool':
                                nest_args[key] = round(value) == 1
                            elif algorithms_nested_args_types[key] == 'enum':
                                nest_args[
                                    key] = f"'{algorithms_nested_args_types[key][str(round(value))]}'"  # 只支持参数类型为字符串
                            else:
                                nest_args[key] = value
                    parameters = {k: v for k, v in parameters.items() if k not in nest_args}
                    # 构造自定义回归器的参数
                    new_kwargs = custom_model_info.special_args.copy()      # 自定义模型的固有参数
                    new_kwargs['model_class'] = model_class                 # 特意增加一个模型类参数，用于传递到CustomRegressor内部
                    new_kwargs |= parameters                                # 自定义模型的关键字参数
                    if custom_model_info.dyn_eval_args is not None:                # 4-对固定参数中的动态评估参数进行处理
                        dyn_args_names = custom_model_info.dyn_eval_args.split(',')
                        if len(nest_args) > 0:  # 如果存在嵌套参数，则将参数值格式化进入eval参数中
                            new_kwargs[custom_model_info.nested_args_name] = new_kwargs[custom_model_info.nested_args_name].format(
                                **nest_args)
                        for dyn_arg_name in dyn_args_names:
                            new_kwargs[dyn_arg_name] = eval(new_kwargs[dyn_arg_name])  # 执行动态的eval
                    if custom_model_info.enum_conversion_args is not None:
                        for key, value in custom_model_info.enum_conversion_args.items():  # 5-对可调参数中的枚举值进行转换
                            if key in new_kwargs:
                                new_kwargs[key] = value[str(round(new_kwargs[key]))]
                    if custom_model_info.complex_lamda_args is not None:
                        for key, value in custom_model_info.complex_lamda_args.items():  # 6-对需lamda处理的参数进行转换
                            if key in new_kwargs:
                                lamda_func = eval(value)
                                new_kwargs[key] = lamda_func(new_kwargs[key])
                    train_base_models.append(
                        (algorithms_name, CustomModelStackingWrapper(**new_kwargs, )))

            # 创建堆叠回归期，用于堆叠优化
            # quantile_cv = QuantileKFold(n_splits=algorithms_config.KFOLD)  # 生成5折分位数交叉验证数据生成器,仅作为训练final_estimator而使用。
            # stacking_regressor = StackingRegressor(estimators=train_base_models, final_estimator=meta_model)
            if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                stacking_regressor = StackingRegressor(estimators=train_base_models, final_estimator=meta_model)
                limesoda_split = LimeSodaSplit(self.limesoda_sample_file,
                                               algorithms_config.LIMESODA_FOLDS_FILE_PATH,
                                               self.prop_name, True)
                y_true_all = []
                y_pred_all = []
                for i in range(algorithms_config.KFOLD):
                    X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i + 1)
                    stacking_regressor.fit(X_train, y_train)
                    y_pred = stacking_regressor.predict(X_val)
                    y_true_all.extend(y_val.values)
                    y_pred_all.extend(y_pred)
                y_true_all = np.array(y_true_all)
                y_pred_all = np.array(y_pred_all)
                score = r2_score(y_true_all, y_pred_all)
            else:  # 采用分折的平均R2来计算
                stacking_regressor = StackingRegressor(estimators=train_base_models, final_estimator=meta_model)
                scorer = make_scorer(
                adjusted_r2,
                n_features=len(train_X.columns) - 1,  # 需要排除两个坐标值列
                greater_is_better=True
                )
                score = np.mean(
                    cross_val_score(stacking_regressor, train_X, train_y, cv=algorithms_config.KFOLD,
                                    scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2', n_jobs=-1))
            self.bayesian_iter_times += 1
            # print(
            #     f"迭代次数：{self.bayesian_iter_times}, {'调整后的R2' if algorithms_config.USE_ADJUSTED_R2 else 'R2'}:{score:.4f}")
            return score

        pbounds = AlgorithmsDataAccess.retrival_all_params_pbounds()  # 把所有参数都提取出来，供优化器生成随机参数
        pbounds['glm_power'] = (1.01, 1.99) if data_distribution_type == DataDistributionType.CompoundPossionGamma else (data_distribution_type.value, data_distribution_type.value)
        pbounds['plsr_n_components'] = (pbounds['plsr_n_components'][0],
                                    min(int(len(train_X) * (1-1/algorithms_config.KFOLD)*0.8),  # 子回归器和堆叠回归器需要两次分折，堆叠回归器默认五折
                                        len(train_X.columns) - 2 if algorithms_config.CSV_GEOM_COL_X in train_X.columns else len(train_X.columns)))  # 要去掉两个坐标值列
        # 创建贝叶斯优化对象
        optimizer = BayesianOptimization(
            f=stacking_evaluate,
            pbounds=pbounds,
            random_state=algorithms_config.RANDOM_STATE,
            verbose=0
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
        best_Stacking_params = optimizer.max['params']
        best_base_models = []
        for algorithms_name in self.algorithms_names:  # 遍历每个堆叠算法
            # 根据算法类型得到相应的参数
            parameters = self.prepare_algorithms_params(algorithms_name, **best_Stacking_params)
            if algorithms_name == AlgorithmsType.XGBR.value:
                best_base_models.append(
                    (algorithms_name, XGBRegressorStackingWrapper(**parameters, enable_categorical=True,
                                                                        random_state=algorithms_config.RANDOM_STATE)))
            elif algorithms_name == AlgorithmsType.RFR.value:
                parameters['mean_encoder'] = mean_encoder
                best_base_models.append(
                    (algorithms_name,
                     RandomForestRegressorStackingWrapper(**parameters, random_state=algorithms_config.RANDOM_STATE)))
            elif algorithms_name == AlgorithmsType.MLP.value:
                parameters['mean_encoder'] = mean_encoder
                best_base_models.append(
                    (algorithms_name,
                     MLPRegressorStackingWrapper(**parameters, early_stopping=True, max_iter=algorithms_config.MLP_MAX_ITER,
                                                 random_state=algorithms_config.RANDOM_STATE)))
            elif algorithms_name == AlgorithmsType.PLSR.value:
                parameters['mean_encoder'] = mean_encoder
                parameters['max_iter'] = algorithms_config.PLSR_MAX_ITER
                best_base_models.append(
                    (algorithms_name, PLSRegressionStackingWrapper(**parameters)))
            elif algorithms_name == AlgorithmsType.GLM.value:
                parameters['mean_encoder'] = mean_encoder
                parameters['max_iter'] = algorithms_config.GLM_MAX_ITER
                best_base_models.append(
                        (algorithms_name, TweedieRegressorStackingWrapper(**parameters)))
            elif algorithms_name == AlgorithmsType.EN.value:
                parameters['mean_encoder'] = mean_encoder
                parameters['max_iter'] = algorithms_config.EN_MAX_ITER
                best_base_models.append(
                    (algorithms_name, ElasticNetStackingWrapper(**parameters)))
            elif algorithms_name == AlgorithmsType.SVR.value:
                parameters['mean_encoder'] = mean_encoder
                parameters['max_iter'] = algorithms_config.SVR_MAX_ITER
                best_base_models.append(
                    (algorithms_name, SVRStackingWrapper(**parameters)))
            elif algorithms_name == AlgorithmsType.KNR.value:
                parameters['mean_encoder'] = mean_encoder
                best_base_models.append(
                    (algorithms_name, KNeighborsRegressorStackingWrapper(**parameters)))
            elif algorithms_name == AlgorithmsType.RK.value:
                regression_model_type = model_type_list[int(parameters['regression_model_type']) - 1]  # 获取回归模型的名称
                del parameters['regression_model_type']  # 使用完丢弃
                reg_model_params = self.prepare_algorithms_params(regression_model_type.value, **best_Stacking_params)
                if regression_model_type == AlgorithmsType.RFR:
                    reg_model = RandomForestRegressor(**reg_model_params, random_state=algorithms_config.RANDOM_STATE)
                elif regression_model_type == AlgorithmsType.XGBR:
                    reg_model = XGBRegressor(**reg_model_params, enable_categorical=True,
                                             random_state=algorithms_config.RANDOM_STATE)
                elif regression_model_type == AlgorithmsType.SVR:
                    reg_model = SVR(**reg_model_params, max_iter=algorithms_config.SVR_MAX_ITER)
                elif regression_model_type == AlgorithmsType.EN:
                    reg_model = ElasticNet(**reg_model_params, max_iter=algorithms_config.EN_MAX_ITER)
                elif regression_model_type == AlgorithmsType.GLM:
                    reg_model = TweedieRegressor(**reg_model_params, max_iter=algorithms_config.GLM_MAX_ITER)
                elif regression_model_type == AlgorithmsType.PLSR:
                    reg_model = PLSRegression(**reg_model_params, max_iter=algorithms_config.PLSR_MAX_ITER)
                elif regression_model_type == AlgorithmsType.KNR:
                    reg_model = KNeighborsRegressor(**reg_model_params)
                elif regression_model_type == AlgorithmsType.MLP:
                    reg_model = MLPRegressor(**reg_model_params, early_stopping=True, max_iter=algorithms_config.MLP_MAX_ITER,
                                             random_state=algorithms_config.RANDOM_STATE)
                parameters['regression_model'] = reg_model  # 设置回归模型对象
                parameters['mean_encoder'] = mean_encoder
                rk = RegressionKrigingRegressorStackingWrapper(**parameters)
                best_base_models.append((algorithms_name, rk))
            elif algorithms_name == AlgorithmsType.CK.value:
                parameters['mean_encoder'] = mean_encoder
                best_base_models.append(
                    (algorithms_name, CoKrigingRegressorStackingWrapper(**parameters)))
            elif algorithms_name == AlgorithmsType.MGWR.value:
                parameters['mean_encoder'] = mean_encoder
                best_base_models.append(
                    (algorithms_name, MGWRStackingWrapper(**parameters)))
            else:  # 自定义模型
                parameters['mean_encoder'] = mean_encoder
                # 获取模型的类
                custom_model_info = AlgorithmsDataAccess.query_custom_model_info(algorithms_name)
                py_file = os.path.join(os.getcwd(), algorithms_name + ".py")
                AlgorithmsDataAccess.save_as_pyfile(custom_model_info.class_code, py_file, True)  # 将代码保持为文件
                loaded_module = load_module_from_file(py_file)  # 从.py文件装载python模块
                model_class = get_sklearn_style_class(loaded_module)  # 从模块中解析出第一个包含fit和method方法的类名称

                # 构造自定义回归器的参数
                new_kwargs = custom_model_info.special_args.copy()  # 自定义模型的固有参数
                new_kwargs['model_class'] = model_class  # 特意增加一个模型类参数，用于传递到CustomRegressor内部
                new_kwargs |= parameters  # 自定义模型的关键字参数
                best_base_models.append(
                    (algorithms_name, CustomModelStackingWrapper(**new_kwargs, )))

        best_Stacking = StackingRegressor(estimators=best_base_models, final_estimator=meta_model,
                                          cv=algorithms_config.KFOLD)

        # 训练得到最优模型
        best_Stacking.fit(train_X, train_y)
        # 预测
        # y_pred_best = best_Stacking.predict(X_Val)
        # 输出最优模型下的评估指标
        # best_Stacking_score = r2_score(train_labels, y_pred_best)
        train_r2 = optimizer.max['target']
        print(
            f"即将结束当前堆叠（{'|'.join(self.algorithms_names)}）的预测-------------------------")
        # 输出最优参数
        print("最佳参数:", best_Stacking_params)
        print("最佳score: ", train_r2)
        print("KFold: {}, 实际搜索次数：{}".format(algorithms_config.KFOLD, self.bayesian_iter_times))

        t2 = time.time()
        print("优化用时：{}秒".format(t2 - t1))
        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_Stacking.predict(train_X)
            save_regressor_result(stacking_algorithms_id, 'stacking', best_Stacking, best_Stacking_params,
                                  train_r2,
                                  train_y, train_predictions, train_X.columns, zscore_normalize, mean_encoder)
        else:
            test_X[algorithms_config.CSV_GEOM_COL_X] = test_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.x)
            test_X[algorithms_config.CSV_GEOM_COL_Y] = test_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.y)
            test_X.pop(algorithms_config.DF_GEOM_COL)
            # 对测试集进行平均编码
            test_X_encoder = mean_encoder.transform(test_X)
            test_predictions = best_Stacking.predict(test_X_encoder)
            save_regressor_result(stacking_algorithms_id, 'stacking', best_Stacking, best_Stacking_params,
                                  train_r2,
                                  test_y, test_predictions, train_X.columns, zscore_normalize, mean_encoder)
        return True