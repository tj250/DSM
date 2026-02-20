import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from bayes_opt import BayesianOptimization
from DSMAlgorithms.base.save_model import save_regressor_result
import algorithms_config
from data.data_dealer import mean_encode_dataset
from scipy.stats import uniform, randint
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit


'''
随机森林回归的包装器（用于建模）
'''


class RandomForestRegressionWrapper(DSMBuildModel):

    """
    生成随机森林回归模型
    train_X:训练集的X
    train_y:训练集的y
    test_X:验证集的X
    test_y:验证集的y
    """

    def build(self, algorithms_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict) -> bool:
        train_X_encoder, mean_encoder = mean_encode_dataset(self.category_vars, train_X, train_y)
        # X_train, X_val, y_train, y_val = train_test_split(train_X_encoder, train_y, test_size=1 / algorithms_config.KFOLD,
        #                                                   random_state=algorithms_config.RANDOM_STATE)

        t1 = time.time()
        # 定义参数分布
        n_estimators_dist_min = 50
        n_estimators_dist_max = 500
        max_depth_dist_min = 3
        max_depth_dist_max = 10
        min_samples_split_dist_min = 2
        min_samples_split_dist_max = 20
        min_samples_leaf_dist_min = 1
        min_samples_leaf_dist_max = 10
        max_features_dist_min = 0.5
        max_features_dist_max = 1
        max_samples_dist_min = 0.5
        max_samples_dist_max = 1.0
        param_dist = {'n_estimators': randint(n_estimators_dist_min, n_estimators_dist_max),
                      'max_depth': [None] + list(np.arange(max_depth_dist_min, max_depth_dist_max)),
                      'min_samples_split': randint(min_samples_split_dist_min, min_samples_split_dist_max),
                      'min_samples_leaf': randint(min_samples_leaf_dist_min, min_samples_leaf_dist_max),
                      'max_features': ['sqrt', 'log2', None] + list(
                          np.linspace(max_features_dist_min, max_features_dist_max, 10)),
                      # 随机搜索时，去除bootstrap参数，因为可能引发异常 ValueError: `max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`.
                      # 'bootstrap': [True, False],
                      'max_samples': uniform(max_samples_dist_min, max_samples_dist_max - max_samples_dist_min)
                      }
        if not algorithms_config.USE_BAYES_OPTI:  # 随机搜索
            # 参数搜索
            kfold = KFold(n_splits=algorithms_config.KFOLD, shuffle=True)
            rfr = RandomForestRegressor(random_state=algorithms_config.RANDOM_STATE)
            param_search = RandomizedSearchCV(estimator=rfr, param_distributions=param_dist,
                                              n_iter=algorithms_config.RANDOM_ITER_TIMES,
                                              cv=custom_cv_split(algorithms_config.LIMESODA_FOLDS_FILE_PATH), random_state=algorithms_config.RANDOM_STATE,
                                              scoring=algorithms_config.RS_SCOROLING, refit=algorithms_config.RS_REFIT, verbose=algorithms_config.VERBOSE)
            param_search.fit(train_X_encoder, train_y)

            print("---------------------{}的最优模型(随机森林回归)----------------------------".format(
                self.prop_name))
            best_RFR_params = param_search.best_params_
            best_RFR_score = param_search.best_score_
            # 输出最优参数
            print("最佳参数:", best_RFR_params)
            print("最佳score: ", best_RFR_score)
            print("KFold: {}, 随机搜索次数：{}".format(algorithms_config.KFOLD, algorithms_config.RANDOM_ITER_TIMES))
            best_RFR = param_search.best_estimator_
            if algorithms_config.RS_REFIT == 'r2':
                train_r2 = np.max(param_search.cv_results_["mean_test_r2"])
            else:
                y_train_pred = best_RFR.predict(train_X_encoder)
                train_r2 = r2_score(train_y, y_train_pred)  # 在训练集上重新计算一下全量样本的R2
        else:
            # 定义目标函数
            def rfr_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap,
                             max_samples, max_depth_use_none, max_features_type):
                max_depth_use_none = True if round(max_depth_use_none) == 1 else False
                if round(max_features_type)==1:
                    max_features_actual = 'sqrt'
                elif round(max_features_type) == 2:
                    max_features_actual = 'log2'
                elif round(max_features_type)==3:
                    max_features_actual = None
                else:
                    max_features_actual=max_features  # 百分比
                rfr = RandomForestRegressor(n_estimators=round(n_estimators),
                                              max_depth=None if max_depth_use_none else round(max_depth),
                                              min_samples_split=round(min_samples_split),
                                              min_samples_leaf=round(min_samples_leaf),
                                              max_features=max_features_actual,
                                              bootstrap=True if round(bootstrap) == 1 else False,
                                              max_samples=max_samples if round(bootstrap) == 1 else None,
                                              random_state=algorithms_config.RANDOM_STATE)
                if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                    limesoda_split = LimeSodaSplit(self.limesoda_sample_file, folds_file_path=algorithms_config.LIMESODA_FOLDS_FILE_PATH, feature_name=self.prop_name)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(algorithms_config.KFOLD):
                        X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i+1)
                        rfr.fit(X_train, y_train)
                        y_pred = rfr.predict(X_val)
                        y_true_all.extend(y_val.values)
                        y_pred_all.extend(y_pred)
                    y_true_all = np.array(y_true_all)
                    y_pred_all = np.array(y_pred_all)
                    score = r2_score(y_true_all, y_pred_all)
                else:   # 采用分折的平均R2来计算
                    scorer = make_scorer(
                        adjusted_r2,
                        n_features=len(train_X.columns),  # 需要排除两个坐标值列
                        greater_is_better=True
                    )
                    score = np.mean(
                        cross_val_score(rfr, train_X_encoder, train_y, cv=algorithms_config.KFOLD,
                                        scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2', n_jobs=-1))
                return score

            pbounds = {"n_estimators": (n_estimators_dist_min, n_estimators_dist_max),
                       "max_depth": (max_depth_dist_min, max_depth_dist_max),
                       "min_samples_split": (min_samples_split_dist_min, min_samples_split_dist_max),
                       "min_samples_leaf": (min_samples_leaf_dist_min, min_samples_leaf_dist_max),
                       "max_features": (max_features_dist_min, max_features_dist_max),
                       "bootstrap": (1, 2),
                       "max_samples": (max_samples_dist_min, max_samples_dist_max),
                       "max_depth_use_none": (1, 5),  # 是否本次预测max_depth使用None,1:None,其它，指定深度：(max_depth_dist_min, max_depth_dist_max)
                       "max_features_type": (1, 10)}  # 本次预测max_features使用哪种类型,1:sqrt,2:log2,3:None,4-10:
            # 创建贝叶斯优化对象
            optimizer = BayesianOptimization(
                f=rfr_evaluate,
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
            best_RFR_params = optimizer.max['params']
            max_depth_use_none = True if round(best_RFR_params['max_depth_use_none']) == 1 else False
            if round(best_RFR_params['max_features_type']) == 1:
                max_features_actual = 'sqrt'
            elif round(best_RFR_params['max_features_type']) == 2:
                max_features_actual = 'log2'
            elif round(best_RFR_params['max_features_type']) == 3:
                max_features_actual = None
            else:
                max_features_actual = best_RFR_params['max_features']  # 百分比
            best_RFR = RandomForestRegressor(n_estimators=round(best_RFR_params['n_estimators']),
                                  max_depth=None if max_depth_use_none else round(best_RFR_params['max_depth']),
                                  min_samples_split=round(best_RFR_params['min_samples_split']),
                                  min_samples_leaf=round(best_RFR_params['min_samples_leaf']),
                                  max_features=max_features_actual,
                                  bootstrap=True if round(best_RFR_params['bootstrap']) == 1 else False,
                                  max_samples=best_RFR_params['max_samples'] if round(best_RFR_params['bootstrap']) == 1 else None,
                                  random_state=algorithms_config.RANDOM_STATE)
            # 训练
            best_RFR.fit(train_X_encoder, train_y)
            train_r2 = optimizer.max['target']

        t2 = time.time()
        print(f"优化用时：{t2 - t1:.1f}秒")

        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_RFR.predict(train_X_encoder)
            save_regressor_result(algorithms_id, 'rfr', best_RFR, best_RFR_params, train_r2,
                                  train_y.to_numpy(), train_predictions, train_X.columns, zscore_normalize, mean_encoder)
        else:
            test_X_encoder = mean_encoder.transform(test_X).values  # Mean Encoding所需代码
            test_predictions = best_RFR.predict(test_X_encoder)
            save_regressor_result(algorithms_id, 'rfr', best_RFR, best_RFR_params, train_r2,
                                  test_y.to_numpy(), test_predictions, train_X.columns, zscore_normalize, mean_encoder)
        return True