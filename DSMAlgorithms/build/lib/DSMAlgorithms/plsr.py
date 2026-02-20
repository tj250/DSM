import time
import pandas as pd
import numpy as np
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from bayes_opt import BayesianOptimization
from DSMAlgorithms.base.save_model import save_regressor_result
import algorithms_config
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import randint
from sklearn.metrics import mean_squared_error
from data.data_dealer import mean_encode_dataset
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit


'''
偏最小二乘回归的包装器（用于建模）
'''


class PLSRWrap(DSMBuildModel):

    """
    生成plsr模型
    train_X:训练集的X
    train_y:训练集的y
    test_X:验证集的X
    test_y:验证集的y
    """

    def build(self, algorithms_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict) -> bool:
        # 平均编码数据集并做分割
        train_X_encoder, mean_encoder = mean_encode_dataset(self.category_vars, train_X, train_y)
        # X_train, X_val, y_train, y_val = train_test_split(train_X_encoder, train_y, test_size=1 / algorithms_config.KFOLD,
        #                                                   random_state=algorithms_config.RANDOM_STATE)
        t1 = time.time()
        # 定义参数分布
        n_components_dist_min = 2
        # 最大的n_components不能超过样本数量-1,也不能超过特征数
        n_components_dist_max = min(len(train_X.columns), int(len(train_X_encoder)*((algorithms_config.KFOLD-1)/algorithms_config.KFOLD))-1)
        param_dist = {
            'n_components': randint(n_components_dist_min, n_components_dist_max),  # 随机抽取潜在变量
        }
        if not algorithms_config.USE_BAYES_OPTI:  # 随机搜索
            # 参数搜索
            pls = PLSRegression(max_iter=algorithms_config.PLSR_MAX_ITER)
            # 使用 KFold 进行交叉验证
            kfold = KFold(n_splits=algorithms_config.KFOLD, shuffle=True)
            print('开始param_search')
            param_search = RandomizedSearchCV(pls, param_distributions=param_dist, n_iter=algorithms_config.RANDOM_ITER_TIMES,
                                              cv=custom_cv_split(algorithms_config.LIMESODA_FOLDS_FILE_PATH), scoring=algorithms_config.RS_SCOROLING,
                                              refit=algorithms_config.RS_REFIT, verbose=algorithms_config.VERBOSE)
            print('开始fit')
            try:
                param_search.fit(train_X_encoder, train_y)
            except Exception as e:
                print(f'随机搜索建模失败，错误信息：{e}')
                return False

            print("---------------------{}的最优模型(PLSR)----------------------------".format(self.prop_name))
            # 输出最优参数
            best_PLS_params = param_search.best_params_
            best_MLP_score = param_search.best_score_
            print("最佳参数:", best_PLS_params)
            print("最佳score: ", best_MLP_score)
            print("KFold: {}, 随机搜索次数：{}".format(algorithms_config.KFOLD, algorithms_config.RANDOM_ITER_TIMES))
            # 使用最优参数创建最优模型
            best_PLS = param_search.best_estimator_
            if algorithms_config.RS_REFIT == 'r2':
                train_r2 = np.max(param_search.cv_results_["mean_test_r2"])
            else:
                y_train_pred = best_PLS.predict(train_X_encoder)
                train_r2 = r2_score(train_y, y_train_pred)  # 在训练集上重新计算一下全量样本的R2
        else:
            # 计算AIC和BIC指标
            def computing_aic_bic(n_components, y_gt, y_pred):
                mse = mean_squared_error(y_gt, y_pred)
                k = n_components * 2 + 1  # 主成分系数+截距项:（主成分权重数量） + （回归系数数量） + 1
                n = len(y_gt)
                log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - (1 / (2 * mse)) * np.sum((y_gt - y_pred) ** 2)

                # 计算AIC和BIC
                aic = 2 * k - 2 * log_likelihood
                bic = np.log(n) * k - 2 * log_likelihood
                return aic, bic

            # 定义目标函数
            def pls_evaluate(n_components):
                plsr = PLSRegression(n_components=round(n_components), max_iter=algorithms_config.PLSR_MAX_ITER)

                if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                    limesoda_split = LimeSodaSplit(self.limesoda_sample_file, folds_file_path=algorithms_config.LIMESODA_FOLDS_FILE_PATH, feature_name=self.prop_name)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(algorithms_config.KFOLD):
                        X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i+1)
                        plsr.fit(X_train, y_train)
                        y_pred = plsr.predict(X_val)
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
                        cross_val_score(plsr, train_X_encoder, train_y, cv=algorithms_config.KFOLD,
                                        scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2', n_jobs=-1))
                return score

            pbounds = {"n_components": (n_components_dist_min, n_components_dist_max)}
            # 创建贝叶斯优化对象
            optimizer = BayesianOptimization(
                f=pls_evaluate,
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
            best_PLS_params = optimizer.max['params']
            best_PLS = PLSRegression(n_components=round(best_PLS_params['n_components']), max_iter=algorithms_config.PLSR_MAX_ITER)
            # 训练
            best_PLS.fit(train_X_encoder, train_y)
            # 预测
            # y_pred_best = best_PLS.predict(train_dataset)
            # 输出最优模型下的评估指标
            # best_MLP_score = r2_score(train_labels, y_pred_best)
            train_r2 = optimizer.max['target']


        t2 = time.time()
        # print(f"优化用时：{:.1f}秒".format(t2 - t1))
        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_PLS.predict(train_X)
            save_regressor_result(algorithms_id, 'plsr', best_PLS, best_PLS_params, train_r2,
                                  train_y, train_predictions, train_X.columns, zscore_normalize, mean_encoder)
        else:
            test_predictions = best_PLS.predict(test_X)
            save_regressor_result(algorithms_id, 'plsr', best_PLS, best_PLS_params, train_r2,
                                  test_y, test_predictions, train_X.columns, zscore_normalize, mean_encoder)
        return True