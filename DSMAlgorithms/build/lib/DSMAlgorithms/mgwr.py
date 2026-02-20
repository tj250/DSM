import numpy as np
import time
import pandas as pd
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from bayes_opt import BayesianOptimization
from data.data_dealer import mean_encode_dataset
from DSMAlgorithms.base.save_model import save_regressor_result
from .sklearn_wrap.mgwr2 import MGWRegressor
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
import algorithms_config
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit

'''
MGWR的包装器（用于建模）
'''


class MGWRWrap(DSMBuildModel):
    '''
    生成MGWR模型
    train_X:训练集的X
    train_y:训练集的y
    test_X:验证集的X
    test_y:验证集的y
    '''

    def build(self, algorithms_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict) -> bool:  # 定义参数网格
        # 将原有给的geometry列拆分为两列
        train_X[algorithms_config.CSV_GEOM_COL_X] = train_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.x)
        train_X[algorithms_config.CSV_GEOM_COL_Y] = train_X[algorithms_config.DF_GEOM_COL].apply(lambda coord: coord.y)
        train_X.pop(algorithms_config.DF_GEOM_COL)
        # 平均编码连续型变量
        train_X_encoder, mean_encoder = mean_encode_dataset(self.category_vars, train_X, train_y)
        # X和y分割为训练和验证集,分割后优化时的训练集和验证集变得固定了
        # X_train, X_val, y_train, y_val = train_test_split(train_X_encoder, train_y,
        #                                                                           test_size=1 / algorithms_config.KFOLD,
        #                                                                           random_state=algorithms_config.RANDOM_STATE)
        t1 = time.time()

        # 定义参数分布
        param_dist = {
            'kernel': ['bisquare', 'gaussian', 'exponential'],
            'search_method': ['golden_section', 'interval'],
            'criterion': ['AICc','AIC','BIC','CV'],
        }
        if not algorithms_config.USE_BAYES_OPTI:  # 随机搜索
            mgwregressor = MGWRegressor()
            # 参数搜索
            kfold = KFold(n_splits=algorithms_config.KFOLD, shuffle=True)
            # 随机搜索（10次采样）
            param_search = RandomizedSearchCV(
                estimator=mgwregressor,
                param_distributions=param_dist,
                n_iter=algorithms_config.RANDOM_ITER_TIMES,
                cv=custom_cv_split(algorithms_config.LIMESODA_FOLDS_FILE_PATH),
                random_state=algorithms_config.RANDOM_STATE,
                scoring=algorithms_config.RS_SCOROLING, refit=algorithms_config.RS_REFIT, verbose=algorithms_config.VERBOSE
            )
            try:
                param_search.fit(train_X_encoder, train_y.values)
            except Exception as e:
                print(f'随机搜索建模失败，错误信息：{e}')
                return False

            print("---------------------{}的最优模型(MGWR)----------------------------".format(
                self.prop_name))
            best_MGWR_params = param_search.best_params_
            best_MGWR_score = param_search.best_score_
            # 输出最优参数
            print("最佳参数:", best_MGWR_params)
            print("最佳score: ", best_MGWR_score)
            print("KFold: {}, 随机搜索次数：{}".format(algorithms_config.KFOLD, algorithms_config.RANDOM_ITER_TIMES))
            best_MGWR = param_search.best_estimator_
            if algorithms_config.RS_REFIT == 'r2':
                train_r2 = np.max(param_search.cv_results_["mean_test_r2"])
            else:
                y_train_pred = best_MGWR.predict(train_X_encoder)
                train_r2 = r2_score(train_y, y_train_pred)  # 在训练集上重新计算一下全量样本的R2
        else:
            # 定义目标函数
            def mgwr_evaluate(kernel, search_method, criterion):
                mgwr_regressor = MGWRegressor(kernel=param_dist['kernel'][round(kernel)],
                                  search_method=param_dist['search_method'][round(search_method)],
                                  criterion=param_dist['criterion'][round(criterion)],)
                if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                    limesoda_split = LimeSodaSplit(self.limesoda_sample_file, algorithms_config.LIMESODA_FOLDS_FILE_PATH, self.prop_name, True)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(algorithms_config.KFOLD):
                        X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i+1)
                        mgwr_regressor.fit(X_train, y_train)
                        y_pred = mgwr_regressor.predict(X_val)
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
                    print(f'kernel:{param_dist['kernel'][int(kernel)]}, search_method:{param_dist['search_method'][int(search_method)]},criterion:{param_dist['criterion'][int(criterion)]}')
                    score = np.mean(
                        cross_val_score(mgwr_regressor, train_X_encoder, train_y.values, cv=algorithms_config.KFOLD,
                                        scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2'))
                return score

            pbounds = {"kernel": (0, len(param_dist['kernel']) - 1),
                       "search_method": (0, len(param_dist['search_method']) - 1),
                       "criterion": (0, len(param_dist['criterion']) - 1),}
            # 创建贝叶斯优化对象
            optimizer = BayesianOptimization(
                f=mgwr_evaluate,
                pbounds=pbounds,
                random_state=algorithms_config.RANDOM_STATE,
                verbose=1
            )

            # 进行贝叶斯优化，需要捕获异常，优于优化过程中，可能因为个别模型的数据导致奇异矩阵等问题而导致建模失败
            try:
                optimizer.maximize(
                    init_points=algorithms_config.MGWR_BAYES_INIT_POINTS,
                    n_iter=algorithms_config.MGWR_BYEAS_ITER_TIMES
                )
            except Exception as e:
                print(f'贝叶斯优化建模失败，错误信息：{e}')
                return False

            # 使用最优参数创建最优模型
            best_MGWR_params = optimizer.max['params']
            best_MGWR_params['kernel'] = param_dist['kernel'][round(best_MGWR_params['kernel'])]
            best_MGWR_params['search_method'] = param_dist['search_method'][round(best_MGWR_params['search_method'])]
            best_MGWR_params['criterion'] = param_dist['criterion'][round(best_MGWR_params['criterion'])]

            best_MGWR = MGWRegressor(**best_MGWR_params)
            # 训练
            best_MGWR.fit(train_X_encoder, train_y)
            train_r2 = optimizer.max['target']

        t2 = time.time()
        print(f"优化用时：{t2 - t1:.1f}秒")

        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_MGWR.predict(train_X_encoder)
            save_regressor_result(algorithms_id, 'mgwr', best_MGWR, best_MGWR_params,
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
            test_predictions = best_MGWR.predict(test_X_encoder)  # 将协变量和坐标值进行拼接后传入
            # 保存结果
            save_regressor_result(algorithms_id, 'mgwr', best_MGWR, best_MGWR_params,
                                  train_r2,
                                  test_y.to_numpy(), test_predictions, train_X.columns, zscore_normalize, mean_encoder)

            # best_model = {'train_coordinates': coords_train, 'intercepts': intercepts, 'slopes': slopes}
            # # 保存结果
            # save_regressor_result(algorithms_id, 'mgwr', best_model, None, best_mgwr_score,
            #                       test_y.to_numpy(), predicted_salt_content, train_X.columns, zscore_normalize, mean_encoder)
        return True