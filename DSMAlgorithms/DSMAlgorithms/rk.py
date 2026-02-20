import random
import numpy as np
import time
import pandas as pd
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from bayes_opt import BayesianOptimization
from DSMAlgorithms.base.save_model import save_regressor_result
import algorithms_config
from data.data_dealer import mean_encode_dataset
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from DSMAlgorithms.sklearn_wrap.regression_kriging import RegressionKrigingRegressor
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit

'''
回归克里金的包装器（用于单一模型建模）
'''


class RegressionKrigeWrap(DSMBuildModel):
    """
    生成回归克里金模型
    train_X:训练集的X
    train_y:训练集的y
    test_X:验证集的X
    test_y:验证集的y

    返回值为bool类型，表示建模过程是否成功。特别是在小样本情况下，优化过程中数据可能会产生奇异矩阵而导致优化建模失败
    """

    def build(self, algorithms_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict) -> bool:  # 定义参数网格
        # 将原有给的geometry列拆分为两列
        train_X[algorithms_config.CSV_GEOM_COL_X] = train_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.x)
        train_X[algorithms_config.CSV_GEOM_COL_Y] = train_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.y)
        train_X.pop(algorithms_config.DF_GEOM_COL)
        # 平均编码
        train_X_encoder, mean_encoder = mean_encode_dataset(self.category_vars, train_X, train_y)
        # X和y分割为训练和验证集
        # X_train, X_val, y_train, y_val = train_test_split(train_X_encoder, train_y,
        #                                                                           test_size=1 / algorithms_config.KFOLD,
        #                                                                           random_state=algorithms_config.RANDOM_STATE)
        t1 = time.time()
        # 定义参数分布
        nlags_min = 4
        nlags_max = 10
        n_closest_points_min = 5
        n_closest_points_max = 30
        param_dist = {
            # 用于回归的模型，如RandomForestRegressor，LinearRegression，SVR
            'regression_model': [RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=4,
                                                       min_samples_leaf=3,
                                                       max_features='sqrt', oob_score=True),
                                 SVR(C=0.1, gamma="auto"),
                                 LinearRegression(copy_X=True, fit_intercept=False)],
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
            'nlags': randint(nlags_min, nlags_max),  # 半变异函数的平均箱数，默认为6
            'weight': [True, False],  # 在自动计算变异函数模型时，是否应在较小滞后处对半方差赋予更大的权重。
            'exact_values': [True, False],  # 如果为 True，插值会在输入位置提供输入值。如果为 False，插值会考虑输入位置输入值的方差/块金，并且不会像精确插值器那样工作
            'n_closest_points': randint(n_closest_points_min, n_closest_points_max),  # Ordinary Kriging中使用的最临近点数量
            'anisotropy_scaling': [(num/10, num/10) for num in range(10)],  # 各向异性比例,用于处理空间相关性在不同方向上存在差异的现象
            'anisotropy_angle': [(num, num, num) for num in range(360)],
        }
        if not algorithms_config.USE_BAYES_OPTI:  # 随机搜索
            # 创建回归克里金模型
            # 克里金（kriging）插值是在有限区域内对区域化变量进行无偏最优估计的一种方法。无偏指的是估计值和实际值之差的期望等于零，
            # 最优指的是估计值和实际值的方差最小。基于这一特点使得克里金插值的效果比其他插值方法要好很多。
            # 在普通克里金（kriging）模型中，通过计算预测点附近的已知值的加权平均来获得预测值。其只有在样本值具备空间相关性时才有意义。
            # coordinates_type:'euclidean' or  ‘geographic’，依据传入坐标是否为平面坐标还是地理坐标而定
            rk = RegressionKrigingRegressor()
            # 参数搜索
            kfold = KFold(n_splits=algorithms_config.KFOLD, shuffle=True)
            # 随机搜索（10次采样）
            param_search = RandomizedSearchCV(
                estimator=rk,
                param_distributions=param_dist,
                n_iter=algorithms_config.RANDOM_ITER_TIMES,
                cv=custom_cv_split(algorithms_config.LIMESODA_FOLDS_FILE_PATH),
                random_state=algorithms_config.RANDOM_STATE,
                scoring=algorithms_config.RS_SCOROLING, refit=algorithms_config.RS_REFIT, verbose=algorithms_config.VERBOSE
            )
            param_search.fit(train_X_encoder, train_y.values)

            print("---------------------{}的最优模型(回归克里金)----------------------------".format(
                self.prop_name))
            best_RegressionKrige_params = param_search.best_params_
            best_RegressionKrige_score = param_search.best_score_
            # 输出最优参数
            print("最佳参数:", best_RegressionKrige_params)
            print("最佳score: ", best_RegressionKrige_score)
            print("KFold: {}, 随机搜索次数：{}".format(algorithms_config.KFOLD, algorithms_config.RANDOM_ITER_TIMES))
            train_r2 = np.max(param_search.cv_results_["mean_test_r2"])
            best_RegressionKrige = param_search.best_estimator_
        else:
            # 定义目标函数
            def reg_krige_evaluate(regression_model, variogram_model, nlags, weight, exact_values, n_closest_points,
                                   anisotropy_scaling, anisotropy_angle, reg_model_rf_n_estimators,
                                   reg_model_rf_max_depth,
                                   reg_model_rf_min_samples_split, reg_model_rf_min_samples_leaf,
                                   reg_model_svr_C, reg_model_svr_epsilon):
                if int(regression_model) == 1:
                    reg_model = RandomForestRegressor(n_estimators=int(reg_model_rf_n_estimators),
                                                      max_depth=int(reg_model_rf_max_depth),
                                                      min_samples_split=int(reg_model_rf_min_samples_split),
                                                      min_samples_leaf=int(reg_model_rf_min_samples_leaf),
                                                      max_features='sqrt',
                                                      oob_score=True)
                elif int(regression_model) == 2:
                    reg_model = SVR(kernel='rbf', C=reg_model_svr_C, epsilon=reg_model_svr_epsilon)
                else:
                    reg_model = LinearRegression(copy_X=True, fit_intercept=True)

                rk = RegressionKrigingRegressor(regression_model=reg_model,
                                                variogram_model=param_dist['variogram_model'][round(variogram_model)],
                                                nlags=round(nlags),
                                                weight=True if round(weight) == 1 else False,
                                                exact_values=True if round(exact_values) == 1 else False,
                                                n_closest_points=round(n_closest_points),
                                                anisotropy_scaling=(anisotropy_scaling, anisotropy_scaling),
                                                anisotropy_angle=(anisotropy_angle, anisotropy_angle, anisotropy_angle)
                                                )
                if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                    limesoda_split = LimeSodaSplit(self.limesoda_sample_file,
                                                   algorithms_config.LIMESODA_FOLDS_FILE_PATH,
                                                   self.prop_name, True)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(algorithms_config.KFOLD):
                        X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i+1)
                        rk.fit(X_train, y_train)
                        y_pred = rk.predict(X_val)
                        y_true_all.extend(y_val.values)
                        y_pred_all.extend(y_pred)
                    y_true_all = np.array(y_true_all)
                    y_pred_all = np.array(y_pred_all)
                    score = r2_score(y_true_all, y_pred_all)
                else:   # 采用分折的平均R2来计算
                    scorer = make_scorer(
                        adjusted_r2,
                        n_features=len(train_X.columns) - 2,  # 需要排除两个坐标值列
                        greater_is_better=True
                    )
                    score = np.mean(
                        cross_val_score(rk, train_X_encoder, train_y, cv=algorithms_config.KFOLD,
                                        scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2', n_jobs=-1))
                return score

            pbounds = {"regression_model": (0, len(param_dist['regression_model']) - 1),
                       "variogram_model": (0, len(param_dist['variogram_model']) - 1),
                       "nlags": (nlags_min, nlags_max),
                       "weight": (1, 2),
                       "exact_values": (1, 2),
                       "n_closest_points": (n_closest_points_min, n_closest_points_max),
                       "anisotropy_scaling": (1, 10),
                       "anisotropy_angle": (0, 360),
                       "reg_model_rf_n_estimators": (50, 300),
                       "reg_model_rf_max_depth": (3, 15),
                       "reg_model_rf_min_samples_split": (2, 20),
                       "reg_model_rf_min_samples_leaf": (1, 5),
                       "reg_model_svr_C": (0.1, 1000),
                       "reg_model_svr_epsilon": (0.01, 0.2),
                       }
            # 创建贝叶斯优化对象
            optimizer = BayesianOptimization(
                f=reg_krige_evaluate,
                pbounds=pbounds,
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
            best_RegressionKrige_params = optimizer.max['params']
            regression_model = int(best_RegressionKrige_params['regression_model'])
            if int(regression_model) == 1:
                best_reg_model = RandomForestRegressor(
                    n_estimators=int(best_RegressionKrige_params["reg_model_rf_n_estimators"]),
                    max_depth=int(best_RegressionKrige_params["reg_model_rf_max_depth"]),
                    min_samples_split=int(best_RegressionKrige_params["reg_model_rf_min_samples_split"]),
                    min_samples_leaf=int(best_RegressionKrige_params["reg_model_rf_min_samples_leaf"]),
                    max_features='sqrt',
                    oob_score=True
                    )
            elif int(regression_model) == 2:
                best_reg_model = SVR(kernel='rbf', C=best_RegressionKrige_params["reg_model_svr_C"],
                                     epsilon=best_RegressionKrige_params["reg_model_svr_epsilon"])
            else:
                best_reg_model = LinearRegression(copy_X=True, fit_intercept=False)
            best_RegressionKrige = RegressionKrigingRegressor(regression_model=best_reg_model,
                                                              variogram_model=param_dist['variogram_model'][
                                                                  round(best_RegressionKrige_params['variogram_model'])],
                                                              nlags=round(best_RegressionKrige_params['nlags']),
                                                              weight=True if round(
                                                                  best_RegressionKrige_params[
                                                                      'weight']) == 1 else False,
                                                              exact_values=True if round(
                                                                  best_RegressionKrige_params[
                                                                      'exact_values']) == 1 else False,
                                                              n_closest_points=round(
                                                                  best_RegressionKrige_params['n_closest_points']),
                                                              anisotropy_scaling=(
                                                                  best_RegressionKrige_params['anisotropy_scaling'],
                                                                  best_RegressionKrige_params['anisotropy_scaling']),
                                                              anisotropy_angle=(
                                                              best_RegressionKrige_params['anisotropy_angle'],
                                                              best_RegressionKrige_params['anisotropy_angle'],
                                                              best_RegressionKrige_params['anisotropy_angle']),
                                                              )
            # 训练
            best_RegressionKrige.fit(train_X_encoder, train_y)
            train_r2 = optimizer.max['target']

        t2 = time.time()
        print(f"优化用时：{t2 - t1:.1f}秒")

        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_RegressionKrige.predict(train_X_encoder)
            save_regressor_result(algorithms_id, 'regression_krige', best_RegressionKrige, best_RegressionKrige_params,
                                  train_r2,
                                  train_y.to_numpy(), train_predictions, train_X.columns, zscore_normalize,
                                  mean_encoder)
        else:
            # geometry列变为两列
            test_X[algorithms_config.CSV_GEOM_COL_X] = test_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.x)
            test_X[algorithms_config.CSV_GEOM_COL_Y] = test_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.y)
            test_X.pop(algorithms_config.DF_GEOM_COL)
            # 对测试集进行平均编码
            test_X_encoder = mean_encoder.transform(test_X)

            # 在测试集上做出预测
            test_predictions = best_RegressionKrige.predict(test_X_encoder)  # 将协变量和坐标值进行拼接后传入
            # 保存结果
            save_regressor_result(algorithms_id, 'regression_krige', best_RegressionKrige, best_RegressionKrige_params,
                                  train_r2,
                                  test_y.to_numpy(), test_predictions, train_X.columns, zscore_normalize, mean_encoder)
        return True
