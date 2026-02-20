import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, GridSearchCV
from bayes_opt import BayesianOptimization
from boruta import BorutaPy
import config

# '''
# 使用xgboosting回归模型进行特征重要性分析
# '''
#
#
# def analyze_by_xgbr(X: pd.DataFrame, y: pd.Series):
#     # 定义目标函数
#     def xgb_evaluate(n_estimators, max_depth, subsample, colsample_bytree, reg_lambda, gamma, learning_rate,
#                      min_child_weight):
#         KFOLD = 5
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 / KFOLD, random_state=config.RANDOM_STATE)
#
#         model = XGBRegressor(
#             n_estimators=int(n_estimators),
#             max_depth=int(max_depth),
#             subsample=subsample,
#             colsample_bytree=colsample_bytree,
#             reg_lambda=reg_lambda,
#             gamma=gamma,
#             learning_rate=learning_rate,
#             min_child_weight=min_child_weight,
#             enable_categorical=True, # 非常重要
#             random_state=config.RANDOM_STATE
#         )
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_val)
#
#         return r2_score(y_val, y_pred)
#         # return -root_mean_squared_error(y_val, y_pred)
#
#     # 定义参数分布
#     pbounds = {"n_estimators": (50, 500),
#                "max_depth": (3, 10),
#                "subsample": (0.6, 1),
#                "colsample_bytree": (0.6, 1),
#                "reg_lambda": (1, 32),
#                "gamma": (0, 1),
#                "learning_rate": (0.01, 0.5),
#                "min_child_weight": (3, 10),
#                }
#     # 创建贝叶斯优化对象
#     optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=pbounds, random_state=config.RANDOM_STATE, verbose=1)
#
#     # 进行贝叶斯优化
#     optimizer.maximize(init_points=10, n_iter=200, )
#
#     # 使用最优参数创建最优模型
#     best_XGBR_params = optimizer.max['params']
#     best_XGBR_params['max_depth'] = int(best_XGBR_params['max_depth'])
#     best_XGBR_params['n_estimators'] = int(best_XGBR_params['n_estimators'])
#     best_XGBR_params['reg_lambda'] = int(best_XGBR_params['reg_lambda'])
#     best_XGBR_params['min_child_weight'] = int(best_XGBR_params['min_child_weight'])
#     best_XGBR = XGBRegressor(**best_XGBR_params, enable_categorical=True, random_state=config.RANDOM_STATE)
#
#     # 训练
#     best_XGBR.fit(X, y)
#     importances = best_XGBR.feature_importances_  # 获取特征重要性
#     importance_data = {}
#     for i in range(len(importances)):
#         importance_data[X.columns[i]] = float(importances[i])
#     items = importance_data.items()
#     sorted_items = sorted(items, key=lambda x: x[1], reverse=True)  # 倒序排列
#     sorted_dict = dict(sorted_items)
#     return sorted_dict

"""
建立xgboost建立回归模型以获取解释变量的特征重要性

Args:
    structure_data_file (str): 结构化数据文件。
    predict_variable (str): 预测变量。
    categorical_variables (list[str]): 类别型变量列表。

Returns:
    排序后的特征重要性列表。
"""


def analyze_features_importance(train_dataset: pd.DataFrame, train_labels: pd.Series) -> dict:
    # 定义参数分布
    param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
                  'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                  'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
                  'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1],
                  'reg_lambda': [1, 2, 4, 8, 16, 32, 64],
                  'gamma': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                  'learning_rate': [0.01,0.05, 0.1, 0.3, 0.5],
                  'min_child_weight': [3, 4, 5, 6, 7, 8],
                  }
    # 参数搜索
    xgbr = XGBRegressor(random_state=config.RANDOM_STATE, enable_categorical=True)
    # 使用 KFold 进行交叉验证
    kfold = KFold(n_splits=5, shuffle=True)
    RANDOM_SEARCH = True
    if RANDOM_SEARCH:
        param_search = RandomizedSearchCV(xgbr, param_distributions=param_grid, n_iter=50, cv=kfold,
                                          scoring='r2', verbose=1)
    else:
        param_search = GridSearchCV(xgbr, param_grid=param_grid, cv=kfold, scoring='r2', verbose=1)
    param_search.fit(train_dataset, train_labels)

    # 使用最优参数创建最优模型
    best_XGBR = param_search.best_estimator_
    # 通过的feature_importances_属性，我们来查看模型的特征重要性：
    importances = best_XGBR.feature_importances_  # 获取特征重要性

    importance_data = {}
    for i in range(len(importances)):
        importance_data[train_dataset.columns[i]] = float(importances[i])
    items = importance_data.items()
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)  # 倒序排列
    sorted_dict = dict(sorted_items)
    return sorted_dict


'''
采用XGboost回归方法分析协变量的特征重要性
'''


def select_important_covars_by_xgboost(X: pd.DataFrame, y: pd.Series, based_cols: list[str] = None) -> list[str]:
    if based_cols is not None:
        # 剔除不在备选列中的解释变量
        X_new = X.copy(deep=True)
        for col in X.columns:
            if col not in based_cols:
                X_new.pop(col)
    else:
        X_new = X
    features_importance = analyze_features_importance(X_new, y)
    print(features_importance)
    left_cols = []
    for key, value in features_importance.items():
        if value > 0:
            left_cols.append(key)  # 特征被选择
    return left_cols

# '''
# 选择重要的环境协变量
# '''
# def select_important_covars_by_boruta(X: pd.DataFrame, y: pd.Series, filter_cols: list[str]) -> list[str]:
#     # 剔除不在备选列中的解释变量
#     X_new = X.copy(deep=True)
#     for col in X.columns:
#         if col not in filter_cols:
#             X_new.pop(col)
#     # 2. 初始化Boruta选择器
#     model = XGBRegressor(
#         n_estimators=int(300),
#         max_depth=int(9),
#         subsample=0.9,
#         colsample_bytree=0.9,
#         reg_lambda=2,
#         gamma=0.1,
#         learning_rate=0.01,
#         min_child_weight=8,
#         enable_categorical=True,  # 非常重要
#         random_state=config.RANDOM_STATE
#     )
#     feat_selector = BorutaPy(
#         estimator=model,
#         n_estimators='auto',
#         verbose=0,
#         alpha=0.05,
#         random_state=config.RANDOM_STATE
#     )
#
#     # 3. 执行特征选择
#     feat_selector.fit(X_new, y)
#
#     # results = pd.DataFrame({
#     #     'Feature': filter_cols,
#     #     'Selected': feat_selector.support_,
#     #     'Ranking': feat_selector.ranking_,  # 特征重要性排名
#     #     'WeakSelected': feat_selector.support_weak_
#     # })
#
#     left_cols = []
#     for i in range(len(filter_cols)):
#         if feat_selector.support_[i]:
#             left_cols.append(filter_cols[i])  # 特征被选择
#             continue
#         if feat_selector.support_weak_[i]:
#             left_cols.append(filter_cols[i])  # 弱特征被选择
#     return left_cols
