import time
import pandas as pd
import numpy as np
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from bayes_opt import BayesianOptimization
from DSMAlgorithms.base.save_model import save_regressor_result
import algorithms_config
from sklearn.neural_network import MLPRegressor
from scipy.stats import loguniform
from data.data_dealer import mean_encode_dataset
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit


'''
多层感知机的包装器（用于建模）
'''


class MLPWrap(DSMBuildModel):

    """
    生成MLP模型
    train_X:训练集的X
    train_y:训练集的y
    test_X:验证集的X
    test_y:验证集的y
    """

    def build(self, algorithms_id: str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame,
              test_y: pd.Series, zscore_normalize: dict) -> bool:
        train_X_encoder, mean_encoder = mean_encode_dataset(self.category_vars, train_X, train_y)
        t1 = time.time()
        # 定义参数分布
        alpha_min = 1e-5
        alpha_max = 1e-2
        learning_rate_init_min = 1e-5
        learning_rate_init_max = 1e-2
        param_dist = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],  # 网络结构
            'activation': ['relu', 'tanh'],  # 激活函数
            # 优化器,‘adam’在相对较大的数据集上效果比较好（几千个样本或者更多），对小数据集来说，lbfgs收敛更快效果也更好
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': loguniform(alpha_min, alpha_max),  # L2正则化强度（对数均匀分布）
            'learning_rate': ['constant', 'invscaling', 'adaptive'],  # 学习率,用于权重更新,只有当solver为’sgd’时使用
            # 初始学习率，控制更新权重的补偿，只有当solver=’sgd’ 或’adam’时使用
            'learning_rate_init': loguniform(learning_rate_init_min, learning_rate_init_max),
            # 随机优化的minibatches的大小,batch_size=min(200,n_samples)，如果solver是’lbfgs’，分类器将不使用minibatch
            'batch_size': [8, 16, 32],
        }
        if not algorithms_config.USE_BAYES_OPTI:  # 随机搜索
            # 参数搜索
            mlp = MLPRegressor(random_state=algorithms_config.RANDOM_STATE, early_stopping=True, max_iter=algorithms_config.MLP_MAX_ITER)
            # 使用 KFold 进行交叉验证
            kfold = KFold(n_splits=algorithms_config.KFOLD, shuffle=True)
            print('开始param_search')
            param_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=algorithms_config.RANDOM_ITER_TIMES,
                                              cv=custom_cv_split(algorithms_config.LIMESODA_FOLDS_FILE_PATH),
                                              scoring=algorithms_config.RS_SCOROLING, refit=algorithms_config.RS_REFIT,
                                              verbose=algorithms_config.VERBOSE)
            print('开始fit')
            param_search.fit(train_X_encoder, train_y)

            print("---------------------{}的最优模型(MLP)----------------------------".format(self.prop_name))
            # 输出最优参数
            best_MLP_params = param_search.best_params_
            best_MLP_score = param_search.best_score_
            print("最佳参数:", best_MLP_params)
            print("最佳score: ", best_MLP_score)
            print("KFold: {}, 随机搜索次数：{}".format(algorithms_config.KFOLD, algorithms_config.RANDOM_ITER_TIMES))
            # 使用最优参数创建最优模型
            best_MLP = param_search.best_estimator_
            if algorithms_config.RS_REFIT == 'r2':
                train_r2 = np.max(param_search.cv_results_["mean_test_r2"])
            else:
                y_train_pred = best_MLP.predict(train_X_encoder)
                train_r2 = r2_score(train_y, y_train_pred)  # 在训练集上重新计算一下全量样本的R2
        else:
            # 定义目标函数
            def mlp_evaluate(hidden_layer_type, activation_type, solver_type, alpha, learning_rate_type,
                             learning_rate_init, batch_size):
                mlp_regressor = MLPRegressor(early_stopping=True,
                                     hidden_layer_sizes=param_dist['hidden_layer_sizes'][round(hidden_layer_type)],
                                     activation=param_dist['activation'][round(activation_type)],
                                     solver=param_dist['solver'][round(solver_type)],
                                     alpha=alpha,
                                     learning_rate=param_dist['learning_rate'][round(learning_rate_type)],
                                     learning_rate_init=learning_rate_init,
                                     batch_size=round(batch_size),
                                     max_iter=algorithms_config.MLP_MAX_ITER,
                                     random_state=algorithms_config.RANDOM_STATE)

                if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                    limesoda_split = LimeSodaSplit(self.limesoda_sample_file, folds_file_path=algorithms_config.LIMESODA_FOLDS_FILE_PATH, feature_name=self.prop_name)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(algorithms_config.KFOLD):
                        X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i+1)
                        mlp_regressor.fit(X_train, y_train)
                        y_pred = mlp_regressor.predict(X_val)
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
                        cross_val_score(mlp_regressor, train_X_encoder, train_y, cv=algorithms_config.KFOLD,
                                        scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2', n_jobs=-1))
                return score

            pbounds = {"hidden_layer_type": (0, len(param_dist['hidden_layer_sizes']) - 1),  # 五种模型结构
                       "activation_type": (0, len(param_dist['activation']) - 1),  # 三种激活函数
                       "solver_type": (0, len(param_dist['solver']) - 1),  # 两类优化器
                       "alpha": (alpha_min, alpha_max),  # L2正则化强度（对数均匀分布）
                       "learning_rate_type": (0, len(param_dist['learning_rate']) - 1),  # 学习率的类型
                       "learning_rate_init": (learning_rate_init_min, learning_rate_init_max),  # 初始学习率
                       "batch_size": (min(param_dist['batch_size']), max(param_dist['batch_size'])),  # 小批量大小
                       }
            # 创建贝叶斯优化对象
            optimizer = BayesianOptimization(
                f=mlp_evaluate,
                pbounds=pbounds,
                random_state=algorithms_config.RANDOM_STATE,
                verbose=1
            )

            # 进行贝叶斯优化，需要捕获异常，优于优化过程中，可能因为个别模型的数据导致奇异矩阵等问题而导致建模失败
            try:
                optimizer.maximize(
                    init_points=algorithms_config.BAYES_INIT_POINTS,
                    n_iter=algorithms_config.BYEAS_ITER_TIMES)
            except Exception as e:
                print(f'贝叶斯优化建模失败，错误信息：{e}')
                return False

            # 使用最优参数创建最优模型
            best_MLP_params = optimizer.max['params']
            best_MLP = MLPRegressor(
                hidden_layer_sizes=param_dist['hidden_layer_sizes'][round(best_MLP_params['hidden_layer_type'])],
                activation=param_dist['activation'][round(best_MLP_params['activation_type'])],
                solver=param_dist['solver'][round(best_MLP_params['solver_type'])],
                alpha=best_MLP_params['alpha'],
                learning_rate=param_dist['learning_rate'][round(best_MLP_params['learning_rate_type'])],
                learning_rate_init=best_MLP_params['learning_rate_init'],
                batch_size=round(best_MLP_params['batch_size']),
                early_stopping=True,
                max_iter=algorithms_config.MLP_MAX_ITER,
                random_state=algorithms_config.RANDOM_STATE)
            # 训练
            best_MLP.fit(train_X_encoder, train_y)
            # 预测
            # y_pred_best = best_MLP.predict(train_dataset)
            # 输出最优模型下的评估指标
            # best_MLP_score = r2_score(train_labels, y_pred_best)
            train_r2 = optimizer.max['target']

        t2 = time.time()
        print(f"优化用时：{t2 - t1:.1f}秒")
        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_MLP.predict(train_X)
            save_regressor_result(algorithms_id, 'mlp', best_MLP, best_MLP_params, train_r2,
                                  train_y, train_predictions, train_X.columns, zscore_normalize, mean_encoder)
        else:
            test_predictions = best_MLP.predict(test_X)
            save_regressor_result(algorithms_id, 'mlp', best_MLP, best_MLP_params, train_r2,
                                  test_y, test_predictions, train_X.columns, zscore_normalize, mean_encoder)
        return True