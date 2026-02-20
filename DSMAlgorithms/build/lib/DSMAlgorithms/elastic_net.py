import numpy as np
from sklearn.linear_model import ElasticNet
import time
import pandas as pd
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from bayes_opt import BayesianOptimization
from DSMAlgorithms.base.save_model import save_regressor_result
import algorithms_config
from data.data_dealer import mean_encode_dataset
from sklearn.metrics import mean_squared_error
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit


'''
弹性网络模型的包装器（用于建模）
'''


class ElasticNetWrap(DSMBuildModel):
    """
    计算弹性网络的AIC和BIC指标
    k:非零系数数量+截距项
    """

    def computing_aic_bic(self, k, y_gt, y_pred):
        mse = mean_squared_error(y_gt, y_pred)
        n = len(y_gt)
        log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - (1 / (2 * mse)) * np.sum((y_gt - y_pred) ** 2)

        # 计算AIC和BIC
        aic = 2 * k - 2 * log_likelihood
        bic = np.log(n) * k - 2 * log_likelihood
        return aic, bic

    '''
    生成弹性网络模型
    train_X:训练集的X
    train_y:训练集的y
    test_X:验证集的X
    test_y:验证集的y
    '''

    def build(self, algorithms_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict) -> bool:
        train_X_encoder, mean_encoder = mean_encode_dataset(self.category_vars, train_X, train_y)
        # X_train, X_val, y_train, y_val = train_test_split(train_X_encoder, train_y, test_size=1 / algorithms_config.KFOLD,
        #                                                   random_state=algorithms_config.RANDOM_STATE)

        t1 = time.time()
        # 定义参数分布
        param_dist = {
            'alpha': np.logspace(-4, 0, 100),  # 正则化强度
            'l1_ratio': np.linspace(0, 1, 20),  # L1/L2混合比例
            'tol': [1e-4, 1e-5, 1e-6]  # 收敛阈值
        }
        if not algorithms_config.USE_BAYES_OPTI:  # 随机搜索
            # 参数搜索
            kfold = KFold(n_splits=algorithms_config.KFOLD, shuffle=True)
            elastic_net = ElasticNet(max_iter=algorithms_config.EN_MAX_ITER)
            param_search = RandomizedSearchCV(estimator=elastic_net, param_distributions=param_dist,
                                              n_iter=algorithms_config.RANDOM_ITER_TIMES,
                                              cv=custom_cv_split(algorithms_config.LIMESODA_FOLDS_FILE_PATH), random_state=algorithms_config.RANDOM_STATE,
                                              scoring=algorithms_config.RS_SCOROLING, refit=algorithms_config.RS_REFIT, verbose=algorithms_config.VERBOSE)
            param_search.fit(train_X_encoder, train_y)

            print("---------------------{}的最优模型(弹性网络)----------------------------".format(
                self.prop_name))
            best_EN_params = param_search.best_params_
            best_EN_score = param_search.best_score_
            # 输出最优参数
            print("最佳参数:", best_EN_params)
            print("最佳score: ", best_EN_score)
            print("KFold: {}, 随机搜索次数：{}".format(algorithms_config.KFOLD, algorithms_config.RANDOM_ITER_TIMES))
            best_EN = param_search.best_estimator_
            if algorithms_config.RS_REFIT == 'r2':
                train_r2 = np.max(param_search.cv_results_["mean_test_r2"])
            else:
                y_train_pred = best_EN.predict(train_X_encoder)
                train_r2 = r2_score(train_y, y_train_pred)  # 在训练集上重新计算一下全量样本的R2
        else:
            # 定义目标函数
            def en_evaluate(alpha, l1_ratio, tol):
                en = ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    tol=float(tol),
                    max_iter = algorithms_config.EN_MAX_ITER
                )
                if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                    limesoda_split = LimeSodaSplit(self.limesoda_sample_file,
                                                   folds_file_path=algorithms_config.LIMESODA_FOLDS_FILE_PATH,
                                                   feature_name=self.prop_name)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(algorithms_config.KFOLD):
                        X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i + 1)
                        en.fit(X_train, y_train)
                        y_pred = en.predict(X_val)
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
                        cross_val_score(en, train_X_encoder, train_y, cv=algorithms_config.KFOLD,
                                        scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2', n_jobs=-1))
                return score

            pbounds = {"alpha": (min(param_dist['alpha']), max(param_dist['alpha'])),
                       "l1_ratio": (min(param_dist['l1_ratio']), max(param_dist['l1_ratio'])),
                       "tol": (min(param_dist['tol']), max(param_dist['tol']))}
            # 创建贝叶斯优化对象
            optimizer = BayesianOptimization(
                f=en_evaluate,
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
            best_EN_params = optimizer.max['params']
            best_EN = ElasticNet(**best_EN_params, max_iter=algorithms_config.EN_MAX_ITER)

            # 训练
            best_EN.fit(train_X_encoder, train_y)
            train_r2 = optimizer.max['target']

        t2 = time.time()
        print(f"优化用时：{t2 - t1:.1f}秒")

        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_EN.predict(train_X_encoder)
            save_regressor_result(algorithms_id, 'elastic_net', best_EN, best_EN_params, train_r2,
                                  train_y.to_numpy(), train_predictions, train_X.columns, zscore_normalize, mean_encoder)
        else:
            test_X_encoder = mean_encoder.transform(test_X).values  # Mean Encoding所需代码
            test_predictions = best_EN.predict(test_X_encoder)
            save_regressor_result(algorithms_id, 'elastic_net', best_EN, best_EN_params, train_r2,
                                  test_y.to_numpy(), test_predictions, train_X.columns, zscore_normalize, mean_encoder)
        return True