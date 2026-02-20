import numpy as np
import time
import pandas as pd
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from bayes_opt import BayesianOptimization
from data.data_dealer import mean_encode_dataset
from DSMAlgorithms.base.save_model import save_regressor_result
from scipy.stats import randint, uniform
from .sklearn_wrap.cokriging import CoKrigingRegressor
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
import algorithms_config
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, LimeSodaSplit

'''
协同克里金的包装器（用于建模）
'''


class CoKrigeWrap(DSMBuildModel):
    '''
    生成协同克里金模型
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
        nlags_min = 4
        nlags_max = 10
        anisotropy_scaling_min = 1
        anisotropy_scaling_max = 10
        anisotropy_angle_min = 0
        anisotropy_angle_max = 360
        param_dist = {
            'variogram_model': ['linear', 'power', 'gaussian', 'spherical', 'exponential'],  # 指定要使用的变异函数模型
            'nlags': randint(nlags_min, nlags_max),  # 半变异函数的平均箱数，默认为6
            'weight': [True, False],  # 在自动计算变异函数模型时，是否应在较小滞后处对半方差赋予更大的权重。
            'exact_values': [True, False],  # 如果为 True，插值会在输入位置提供输入值。如果为 False，插值会考虑输入位置输入值的方差/块金，并且不会像精确插值器那样工作
            'anisotropy_scaling': uniform(anisotropy_scaling_min, anisotropy_scaling_max),
            # 各向异性比例,用于处理空间相关性在不同方向上存在差异的现象
            'anisotropy_angle': uniform(anisotropy_angle_min, anisotropy_angle_max)  # 各向异性角度，以degree为单位
            # 'pseudo_inv': [True,False],  # 是否使用伪逆克里金矩阵求解克里金系统。如果为 True，则数值稳定性更高，冗余点会被平均。但这可能需要更多时间。
            # 'pseudo_inv_type': ['pinv', 'pinvh']  # 计算伪逆矩阵的算法
        }
        if not algorithms_config.USE_BAYES_OPTI:  # 随机搜索
            ck = CoKrigingRegressor()
            # 参数搜索
            kfold = KFold(n_splits=algorithms_config.KFOLD, shuffle=True)
            # 随机搜索（10次采样）
            param_search = RandomizedSearchCV(
                estimator=ck,
                param_distributions=param_dist,
                n_iter=algorithms_config.RANDOM_ITER_TIMES,
                cv=custom_cv_split(algorithms_config.LIMESODA_FOLDS_FILE_PATH),
                random_state=algorithms_config.RANDOM_STATE,
                scoring=algorithms_config.RS_SCOROLING, refit=algorithms_config.RS_REFIT, verbose=algorithms_config.VERBOSE
            )
            param_search.fit(train_X_encoder, train_y)

            print("---------------------{}的最优模型(协同克里金)----------------------------".format(
                self.prop_name))
            best_CoKrige_params = param_search.best_params_
            best_CoKrige_score = param_search.best_score_
            # 输出最优参数
            print("最佳参数:", best_CoKrige_params)
            print("最佳score: ", best_CoKrige_score)
            print("KFold: {}, 随机搜索次数：{}".format(algorithms_config.KFOLD, algorithms_config.RANDOM_ITER_TIMES))
            best_CoKrige = param_search.best_estimator_
            if algorithms_config.RS_REFIT == 'r2':
                train_r2 = np.max(param_search.cv_results_["mean_test_r2"])
            else:
                y_train_pred = best_CoKrige.predict(train_X)
                train_r2 = r2_score(train_y, y_train_pred)  # 在训练集上重新计算一下全量样本的R2
        else:
            # 定义目标函数
            def cokrige_evaluate(variogram_model, nlags, weight, exact_values, anisotropy_scaling, anisotropy_angle):
                ck = CoKrigingRegressor(variogram_model=param_dist['variogram_model'][round(variogram_model)],
                                        nlags=round(nlags),
                                        weight=True if round(weight) == 1 else False,
                                        exact_values=True if round(exact_values) == 1 else False,
                                        anisotropy_scaling=float(anisotropy_scaling),
                                        anisotropy_angle=float(anisotropy_angle),
                                        )
                if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                    limesoda_split = LimeSodaSplit(self.limesoda_sample_file, algorithms_config.LIMESODA_FOLDS_FILE_PATH, self.prop_name, True)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(algorithms_config.KFOLD):
                        X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i+1)
                        ck.fit(X_train, y_train)
                        y_pred = ck.predict(X_val)
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
                        cross_val_score(ck, train_X_encoder, train_y, cv=algorithms_config.KFOLD,
                                        scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2', n_jobs=-1))
                return score

            pbounds = {"variogram_model": (0, len(param_dist['variogram_model']) - 1),
                       "nlags": (nlags_min, nlags_max),
                       "weight": (1, 2),
                       "exact_values": (1, 2),
                       "anisotropy_scaling": (anisotropy_scaling_min, anisotropy_scaling_max),
                       "anisotropy_angle": (anisotropy_angle_min, anisotropy_angle_max)}
            # 创建贝叶斯优化对象
            optimizer = BayesianOptimization(
                f=cokrige_evaluate,
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
            best_CoKrige_params = optimizer.max['params']
            best_CoKrige_params['variogram_model'] = param_dist['variogram_model'][
                round(best_CoKrige_params['variogram_model'])]
            best_CoKrige_params['nlags'] = round(best_CoKrige_params['nlags'])
            best_CoKrige_params['weight'] = True if round(best_CoKrige_params['weight']) == 1 else False
            best_CoKrige_params['exact_values'] = True if round(best_CoKrige_params['exact_values']) == 1 else False
            best_CoKrige_params['anisotropy_scaling'] = best_CoKrige_params['anisotropy_scaling']
            best_CoKrige_params['anisotropy_angle'] = best_CoKrige_params['anisotropy_angle']
            best_CoKrige = CoKrigingRegressor(**best_CoKrige_params)
            # 训练
            best_CoKrige.fit(train_X_encoder, train_y.values)
            train_r2 = optimizer.max['target']

        t2 = time.time()
        print(f"优化用时：{t2 - t1:.1f}秒")

        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_CoKrige.predict(train_X_encoder)
            save_regressor_result(algorithms_id, 'cokrige', best_CoKrige, best_CoKrige_params,
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
            test_predictions = best_CoKrige.predict(test_X_encoder)  # 将协变量和坐标值进行拼接后传入
            # 保存结果
            save_regressor_result(algorithms_id, 'cokrige', best_CoKrige, best_CoKrige_params,
                                  train_r2,
                                  test_y.to_numpy(), test_predictions, train_X.columns, zscore_normalize, mean_encoder)
        return True