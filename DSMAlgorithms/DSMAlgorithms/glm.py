import numpy as np
import time
import pandas as pd
from sklearn.metrics import (make_scorer)
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score, train_test_split
from bayes_opt import BayesianOptimization
from DSMAlgorithms.base.save_model import save_regressor_result
import algorithms_config
from data.data_dealer import mean_encode_dataset
from scipy.stats import loguniform, uniform
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_gamma_deviance
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, LimeSoDaFold, custom_cv_split, r2_score, LimeSodaSplit
from DSMAlgorithms.base.base_data_structure import DataDistributionType

'''
广义线性模型的包装器（用于建模）
广义线性模型（GLM）是一种扩展传统线性回归的统计方法，通过引入链接函数和灵活的分布假设，解决因变量非正态分布或离散的问题
GLM突破了普通线性回归对因变量正态分布的限制。其核心假设是：自变量与因变量关系可通过线性组合表达，但响应变量可服从二项分布、泊松分布等非正态分布类型
普通线性回归要求因变量服从正态分布且方差恒定，仅使用恒等链接函数（即E(Y)=η）。而GLM允许因变量服从指数族分布，链接函数可自由选择.
当响应变量为二元变量时，GLM退化为逻辑回归；
当响应变量为计数数据时，GLM转化为泊松回归；
普通线性回归是GLM在正态分布假设下的特例

GLM数学模型的三要素:
线性预测器（η）：即自变量的线性组合，η=β₀+β₁X₁+…+βₖXₖ，反映变量间的线性关系。
链接函数（g）：连接η与因变量期望值E(Y)，满足g(E(Y))=η。例如泊松回归使用对数函数，逻辑回归使用logit函数。
响应变量分布：根据数据类型选择分布，如二分类变量对应二项分布，计数数据对应泊松分布(非负整数（计数数据）‌)，连续非正态数据可使用伽马分布(数据类型‌：正实数（连续数据）‌)。
注意：
GLM对分布假设敏感，错误设定导致偏差；链接函数选择依赖经验，部分场景需反复试验；无法直接处理非线性关系（需引入多项式项或广义加性模型扩展）。
当前研究聚焦于高维数据适应性（如结合Lasso正则化防止过拟合）和复杂链接函数设计（如深度神经网络构建非线性链接)。同时，贝叶斯广义线性模型通过引入先验分布提升小样本数据下的稳定性，成为重要发展方向。
'''


class GLMWrap(DSMBuildModel):
    """
    构造广义线性回归模型进行建模
    prop_name:预测的属性名称
    csv_file:样点和协变量信息所在的文件
    category_vars:协变量当中的类别型变量
    data_distribution_type:响应变量数据分布类型
    """

    def __init__(self, prop_name: str, category_vars: dict, data_distribution_type: DataDistributionType):
        super().__init__(prop_name, category_vars)
        self.data_distribution_type = data_distribution_type

    '''
    计算gamma分布的AIC和BIC指标
    n_samples:样本数量
    n_features:特征数量
    '''

    def computing_aic_bic_of_gamma(self, n_samples, n_features, y_gt, y_pred):
        deviance = mean_gamma_deviance(y_gt, y_pred) * n_samples  # 总偏差
        log_likelihood = -deviance / 2  # 对数似然近似计算

        # 计算AIC和BIC
        n_params = n_features + 1  # 特征数+截距项
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_samples) * n_params - 2 * log_likelihood
        return aic, bic

    '''
    生成GLM模型
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

        alpha_dist_min = 1e-3
        alpha_dist_max = 1
        power_dist_min = 1.01
        power_dist_max = 1.99
        param_dist = {
            'alpha': loguniform(alpha_dist_min, alpha_dist_max),
            'fit_intercept': [True, False],
            'solver': ['lbfgs', 'newton-cholesky'],  # 优化算法
            'power': uniform(power_dist_min,
                             power_dist_max - power_dist_min) if self.data_distribution_type == DataDistributionType.CompoundPossionGamma else (
                self.data_distribution_type.value, self.data_distribution_type.value),  # 1<p<2
        }
        if not algorithms_config.USE_BAYES_OPTI:  # 随机搜索
            # 参数搜索
            kfold = KFold(n_splits=algorithms_config.KFOLD, shuffle=True)
            model = TweedieRegressor(max_iter=algorithms_config.GLM_MAX_ITER)
            param_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                              n_iter=algorithms_config.RANDOM_ITER_TIMES,
                                              cv=custom_cv_split(algorithms_config.LIMESODA_FOLDS_FILE_PATH),
                                              random_state=algorithms_config.RANDOM_STATE,
                                              scoring=algorithms_config.RS_SCOROLING,
                                              refit=algorithms_config.RS_REFIT,
                                              # 负对数似然:neg_mean_poisson_deviance' 或 'neg_gamma_deviance'
                                              verbose=algorithms_config.VERBOSE)
            try:
                param_search.fit(train_X_encoder, train_y)
            except Exception as e:
                print(f'随机搜索建模失败，错误信息：{e}')
                return False

            print("---------------------{}的最优模型(广义线性回归)----------------------------".format(
                self.prop_name))
            best_GLM_params = param_search.best_params_
            best_GLM_score = param_search.best_score_
            # 输出最优参数
            print("最佳参数:", best_GLM_params)
            print("最佳score: ", best_GLM_score)
            print("KFold: {}, 随机搜索次数：{}".format(algorithms_config.KFOLD, algorithms_config.RANDOM_ITER_TIMES))
            best_GLM = param_search.best_estimator_
            if algorithms_config.RS_REFIT == 'r2':
                train_r2 = np.max(param_search.cv_results_["mean_test_r2"])
            else:
                y_train_pred = best_GLM.predict(train_X_encoder)
                train_r2 = r2_score(train_y, y_train_pred)  # 在训练集上重新计算一下全量样本的R2
        else:
            def tweedie_evaluate(alpha, fit_intercept, solver, power):
                tweedie_regressor = TweedieRegressor(
                    power=float(power),
                    alpha=float(alpha),
                    fit_intercept=True if round(fit_intercept) == 1 else False,
                    solver=param_dist['solver'][round(solver)],
                    max_iter=algorithms_config.GLM_MAX_ITER,
                )
                if algorithms_config.USE_LIMESODA_KFOLD:  # 如果采用limesoda的计算方式:每一次分折预测后不在测试集上计算R2，而是所有折计算完毕后，将所有折的预测值和真实值进行R2计算
                    limesoda_split = LimeSodaSplit(self.limesoda_sample_file,
                                                   folds_file_path=algorithms_config.LIMESODA_FOLDS_FILE_PATH,
                                                   feature_name=self.prop_name)
                    y_true_all = []
                    y_pred_all = []
                    for i in range(algorithms_config.KFOLD):
                        X_train, X_val, y_train, y_val = limesoda_split.get_fold_n(i + 1)
                        tweedie_regressor.fit(X_train, y_train)
                        y_pred = tweedie_regressor.predict(X_val)
                        y_true_all.extend(y_val.values)
                        y_pred_all.extend(y_pred)
                    y_true_all = np.array(y_true_all)
                    y_pred_all = np.array(y_pred_all)
                    score = r2_score(y_true_all, y_pred_all)
                else:  # 采用分折的平均R2来计算
                    scorer = make_scorer(
                        adjusted_r2,
                        n_features=len(train_X.columns),
                        greater_is_better=True
                    )
                    if power > 1.0 and power <= 2.0:
                        train_y[train_y == 0] = 1e-6  # 很重要，对于gamma分布或复合泊松-伽马分布，如果因变量存在0值，则建模过程报错，需要加一个很小的值

                    score = np.mean(
                        cross_val_score(tweedie_regressor, train_X_encoder, train_y, cv=algorithms_config.KFOLD,
                                        scoring=scorer if algorithms_config.USE_ADJUSTED_R2 else 'r2', n_jobs=-1))
                return score

            pbounds = {"alpha": (alpha_dist_min, alpha_dist_max),
                       "fit_intercept": (1, 2),
                       "solver": (0, len(param_dist['solver']) - 1),
                       'power': (power_dist_min,
                                 power_dist_max) if self.data_distribution_type == DataDistributionType.CompoundPossionGamma else (
                           self.data_distribution_type.value, self.data_distribution_type.value),
                       }
            # 创建贝叶斯优化对象
            optimizer = BayesianOptimization(
                f=tweedie_evaluate,
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
            best_GLM_params = optimizer.max['params']
            best_GLM_params['fit_intercept'] = True if round(best_GLM_params['fit_intercept']) == 1 else False
            best_GLM_params['solver'] = param_dist['solver'][round(best_GLM_params['solver'])]
            best_GLM = TweedieRegressor(**best_GLM_params, max_iter=algorithms_config.GLM_MAX_ITER)

            # 训练
            best_GLM.fit(train_X_encoder, train_y)
            train_r2 = optimizer.max['target']

        t2 = time.time()
        print(f"优化用时：{t2 - t1:.1f}秒")

        # 对训练数据进行预测
        if test_X is None:
            train_predictions = best_GLM.predict(train_X_encoder)
            save_regressor_result(algorithms_id, 'glm', best_GLM, best_GLM_params, train_r2,
                                  train_y.to_numpy(), train_predictions, train_X.columns, zscore_normalize,
                                  mean_encoder)
        else:
            test_X_encoder = mean_encoder.transform(test_X).values  # Mean Encoding所需代码
            test_predictions = best_GLM.predict(test_X_encoder)
            save_regressor_result(algorithms_id, 'glm', best_GLM, best_GLM_params, train_r2,
                                  test_y.to_numpy(), test_predictions, train_X.columns, zscore_normalize, mean_encoder)
        return True
