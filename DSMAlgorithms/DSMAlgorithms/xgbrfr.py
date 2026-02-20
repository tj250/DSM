import time
import pandas as pd
import numpy as np
from xgboost import XGBRFRegressor
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from bayes_opt import BayesianOptimization
from DSMAlgorithms.base.save_model import save_regressor_result
import algorithms_config
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit

'''
xgboost随机森林回归的包装器（用于建模）
'''


class XGBRFRWrap(DSMBuildModel):
    """
    生成xgboost Random Forest回归模型
    train_X:训练集的X
    train_y:训练集的y
    test_X:验证集的X
    test_y:验证集的y

    subsample,有放回抽样方式训练，每次训练中使用的总样本数据的比例，如果为1，则每次训练使用所有样本，数据量少时，可以用1
    colsample_bytree,当构建每一颗树时，采样的特征数量比例，除此之外，还有如下参数可调
        --colsample_bylevel，当一棵树中每增加一级深度时，采样的特征数量比例
        --colsample_bynode，每次节点分裂时，采样的特征数量比例
    reg_lambda:L2正则项对权重的影响，范围：[0, inf)，增大，模型更保守，防止过拟合
    gamma：在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
    """

    def build(self, model_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict) -> bool:
        # 依据搜索策略来分两个量级控制参数范围，一方面降低过拟合，另外一方面使得预测不至于太差
        t1 = time.time()
        param_dist = {'n_estimators': [50, 100, 150, 200, 250, 300],
                      'max_depth': [3, 4, 5, 6, 7, 8],
                      'subsample': [0.6, 0.7, 0.8, 0.9],
                      'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                      'reg_lambda': [1, 2, 4, 8, 16, 32],
                      'gamma': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                      'min_child_weight': [3, 4, 5, 6, 7, 8]}
        if not algorithms_config.USE_BAYES_OPTI:
            # 参数搜索
            kfold = KFold(n_splits=algorithms_config.KFOLD, shuffle=True)
            xgbrfr = XGBRFRegressor(booster='gbtree', random_state=algorithms_config.RANDOM_STATE, enable_categorical=True)
            param_search = RandomizedSearchCV(xgbrfr, param_distributions=param_dist,
                                              n_iter=algorithms_config.RANDOM_ITER_TIMES,
                                              cv=custom_cv_split(algorithms_config.LIMESODA_FOLDS_FILE_PATH),
                                              scoring=algorithms_config.RS_SCOROLING, refit=algorithms_config.RS_REFIT,
                                              verbose=algorithms_config.VERBOSE)
            param_search.fit(train_X, train_y)

            print("---------------------{}的最优模型(XGBoost Random Forest回归)----------------------------".format(
                self.prop_name))
            best_XGBRFR_params = param_search.best_params_
            best_XGBRFR_score = param_search.best_score_
            # 输出最优参数
            print("最佳参数:", best_XGBRFR_params)
            print("最佳score: ", best_XGBRFR_score)
            print("KFold: {}, 随机搜索次数：{}".format(algorithms_config.KFOLD, algorithms_config.RANDOM_ITER_TIMES))
            best_XGBRFR = param_search.best_estimator_
            if algorithms_config.RS_REFIT == 'r2':
                train_r2 = np.max(param_search.cv_results_["mean_test_r2"])
            else:
                y_train_pred = best_XGBRFR.predict(train_X)
                train_r2 = r2_score(train_y, y_train_pred)  # 在训练集上重新计算一下全量样本的R2
        else:
            # 定义目标函数
            def xgbrf_evaluate(n_estimators, max_depth, subsample, colsample_bytree, reg_lambda, gamma,
                               min_child_weight):
                xgbrfr = XGBRFRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    gamma=gamma,
                    enable_categorical=True,
                    min_child_weight=min_child_weight,
                    random_state=algorithms_config.RANDOM_STATE
                )
                if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                    limesoda_split = LimeSodaSplit(self.limesoda_sample_file, folds_file_path=algorithms_config.LIMESODA_FOLDS_FILE_PATH, feature_name=self.prop_name)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(algorithms_config.KFOLD):
                        X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i+1)
                        xgbrfr.fit(X_train, y_train)
                        y_pred = xgbrfr.predict(X_val)
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
                        cross_val_score(xgbrfr, train_X, train_y, cv=algorithms_config.KFOLD,
                                        scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2', n_jobs=-1))
                return score

            pbounds = {"n_estimators": (min(param_dist['n_estimators']), max(param_dist['n_estimators'])),
                       "max_depth": (min(param_dist['max_depth']), max(param_dist['max_depth'])),
                       "subsample": (min(param_dist['subsample']), max(param_dist['subsample'])),
                       "colsample_bytree": (min(param_dist['colsample_bytree']), max(param_dist['colsample_bytree'])),
                       "reg_lambda": (min(param_dist['reg_lambda']), max(param_dist['reg_lambda'])),
                       "gamma": (min(param_dist['gamma']), max(param_dist['gamma'])),
                       # "learning_rate": (min(param_dist['learning_rate']), max(param_dist['learning_rate'])),
                       "min_child_weight": (min(param_dist['min_child_weight']), max(param_dist['min_child_weight'])),
                       }
            # 创建贝叶斯优化对象
            optimizer = BayesianOptimization(
                f=xgbrf_evaluate,
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
            best_XGBRFR_params = optimizer.max['params']
            best_XGBRFR_params['max_depth'] = int(best_XGBRFR_params['max_depth'])
            best_XGBRFR_params['n_estimators'] = int(best_XGBRFR_params['n_estimators'])
            best_XGBRFR_params['reg_lambda'] = int(best_XGBRFR_params['reg_lambda'])
            best_XGBRFR_params['min_child_weight'] = int(best_XGBRFR_params['min_child_weight'])
            best_XGBRFR = XGBRFRegressor(**best_XGBRFR_params, enable_categorical=True,
                                         random_state=algorithms_config.RANDOM_STATE)

            # 重新拟合，以训练得到一个最优模型
            best_XGBRFR.fit(train_X, train_y)
            train_r2 = optimizer.max['target']

        t2 = time.time()
        print(f"优化用时：{t2 - t1:.0f}秒")

        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_XGBRFR.predict(train_X)
            save_regressor_result(model_id, 'xgbrfr', best_XGBRFR, best_XGBRFR_params, train_r2,
                                  train_y, train_predictions, train_X.columns, zscore_normalize)
        else:
            test_predictions = best_XGBRFR.predict(test_X)
            save_regressor_result(model_id, 'xgbrfr', best_XGBRFR, best_XGBRFR_params, train_r2,
                                  test_y, test_predictions, train_X.columns, zscore_normalize)
        return True
