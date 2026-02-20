import datetime
from natsort import natsorted
from typing import List
from mcp.server.fastmcp import FastMCP
import multiprocessing
import os
import shutil
import sys
import json
import pickle
import numpy as np
import pandas as pd
import random
from PIL import Image
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor,XGBRFRegressor,XGBClassifier,XGBRFClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer



DEBUG_STATE = True
RANDOM_SEARCH = True
RANDOM_STATE= 42
REG_LAMBDA = 1
GAMMA = 1
KFOLD = 3
RANDOM_ITER_TIMES = 100



"""
建立xgboost建立回归模型以获取解释变量的特征重要性

Args:
    structure_data_file (str): 结构化数据文件。
    predict_variable (str): 预测变量。
    categorical_variables (list[str]): 类别型变量列表。

Returns:
    排序后的特征重要性列表。
"""


def analyze_features_importance(structure_data_file: str, prediction_variable: str,
                                interpretation_variables: list[str],
                                categorical_variables: list[str])->dict:
    train_dataset, train_labels, categorical_features_info  = prepare_dataset(structure_data_file, prediction_variable,
                                                                              interpretation_variables,
                                                                              categorical_variables)
    # 依据搜索策略来分两个量级控制参数范围，一方面降低过拟合，另外一方面使得预测不至于太差
    n_estimators = [50, 100, 150, 200, 250, 300]
    max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
    subsample = [0.5, 0.6, 0.7, 0.8, 0.9]
    col_sample = [0.6, 0.7, 0.8, 0.9, 1]
    reg_lambda = [1, 2, 4, 8, 16, 32, 64]
    gamma = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    learning_rate = [0.05, 0.1, 0.3, 0.5]

    # 定义参数分布
    if DEBUG_STATE:
        param_grid = {'n_estimators': [150], 'max_depth': [5], 'subsample': [0.8], 'colsample_bytree': [0.8],
                      'learning_rate': [0.05]}
    else:
        param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'subsample': subsample,
                      'colsample_bytree': col_sample,
                      'reg_lambda': reg_lambda, 'gamma': gamma, 'learning_rate': learning_rate}
    # 参数搜索
    xgbr = XGBRegressor(random_state=RANDOM_STATE, enable_categorical=True, min_child_weight=2)
    # 使用 KFold 进行交叉验证
    kfold = KFold(n_splits=KFOLD, shuffle=True)
    if RANDOM_SEARCH:
        param_search = RandomizedSearchCV(xgbr, param_distributions=param_grid, n_iter=RANDOM_ITER_TIMES, cv=kfold,
                                          scoring='r2', verbose=2)
    else:
        param_search = GridSearchCV(xgbr, param_grid=param_grid, cv=kfold, scoring='r2', verbose=2)
    param_search.fit(train_dataset, train_labels)

    # 使用最优参数创建最优模型
    best_XGBR = param_search.best_estimator_
    # 通过的feature_importances_属性，我们来查看模型的特征重要性：
    importances = best_XGBR.feature_importances_  # 获取特征重要性

    importance_data = {}
    for i in range(len(importances)):
        importance_data[train_dataset.columns[i]] = round(float(importances[i]),4)
    items = importance_data.items()
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)  # 倒序排列
    sorted_dict = dict(sorted_items)
    return sorted_dict

'''
保存模型相关的类别变量的类别信息，保存下来的json文件将被用于预测将列由数值转变为类别
'''
def save_category_file(build_model_id:str, categorical_features_info:dict):
    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, '{}-category.json'.format(build_model_id)), 'w', encoding='utf-8') as file:
        json.dump(categorical_features_info, file, ensure_ascii=False)

"""
建立极限梯度提升树的回归模型

Args:
    structure_data_file (str): 结构化数据文件。
    predict_variable (str): 预测变量。
    categorical_variables (list[str]): 类别型变量列表。

Returns:
    是否成功开启了建模过程。
"""


def build_XGBR(build_model_id, structure_data_file: str, predict_variable: str, categorical_variables: list[str]):
    train_dataset, train_labels, categorical_features_info  = prepare_dataset(structure_data_file, predict_variable, categorical_variables)

    # 保存模型相关的类别变量的类别信息，用于预测环节
    save_category_file(build_model_id, categorical_features_info)

    # 依据搜索策略来分两个量级控制参数范围，一方面降低过拟合，另外一方面使得预测不至于太差
    n_estimators = [50, 100, 150, 200, 250, 300]
    max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
    subsample = [0.5, 0.6, 0.7, 0.8, 0.9]
    col_sample = [0.6, 0.7, 0.8, 0.9, 1]
    reg_lambda = [1, 2, 4, 8, 16, 32, 64]
    gamma = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    learning_rate = [0.05, 0.1, 0.3, 0.5]

    # 定义参数分布
    if DEBUG_STATE:
        param_grid = {'n_estimators': [150], 'max_depth': [5], 'subsample': [0.8], 'colsample_bytree': [0.8],
                      'learning_rate': [0.05]}
    else:
        param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'subsample': subsample,
                      'colsample_bytree': col_sample,
                      'reg_lambda': reg_lambda, 'gamma': gamma, 'learning_rate': learning_rate}
    # 参数搜索
    xgbr = XGBRegressor(random_state=RANDOM_STATE, enable_categorical=True, min_child_weight=2)
    # 使用 KFold 进行交叉验证
    kfold = KFold(n_splits=KFOLD, shuffle=True)
    if RANDOM_SEARCH:
        param_search = RandomizedSearchCV(xgbr, param_distributions=param_grid, n_iter=RANDOM_ITER_TIMES, cv=kfold,
                                          scoring='r2', verbose=2)
    else:
        param_search = GridSearchCV(xgbr, param_grid=param_grid, cv=kfold, scoring='r2', verbose=2)
    param_search.fit(train_dataset, train_labels)

    print("---------------------{}的最优模型(XGBoost回归)----------------------------".format(predict_variable))
    # 输出最优参数
    print("最佳参数:", param_search.best_params_)
    print("最佳R2 score: ", param_search.best_score_)
    print("KFold: {}, 随机搜索次数：{}".format(KFOLD, RANDOM_ITER_TIMES))
    # 使用最优参数创建最优模型
    best_XGBR = param_search.best_estimator_

    # 对训练数据进行预测
    test_predictions = best_XGBR.mapping(train_dataset)
    model_params = save_regressor_result(build_model_id, predict_variable, 'XGBoost回归', best_XGBR, param_search.best_params_,
                                         param_search.best_score_, train_labels, test_predictions,
                                         train_dataset.columns)
    return True


"""
建立极限梯度提升随机森林的回归模型

Args:
    structure_data_file (str): 结构化数据文件。
    predict_variable (str): 预测变量。
    categorical_variables (list[str]): 类别型变量列表。

Returns:
    是否成功开启了建模过程。
"""

def build_XGBRFR(build_model_id, structure_data_file: str, predict_variable: str, categorical_variables: list[str]):
    train_dataset, train_labels, categorical_features_info = prepare_dataset(structure_data_file, predict_variable, categorical_variables)
    # 保存模型相关的类别变量的类别信息，用于预测环节
    save_category_file(build_model_id, categorical_features_info)

    # 依据搜索策略来分两个量级控制参数范围，一方面降低过拟合，另外一方面使得预测不至于太差
    n_estimators = [50, 100, 150, 200, 250, 300]
    max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
    subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    col_sample = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    reg_lambda = [1, 2, 4, 8, 16, 32]
    gamma = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    # 定义参数分布
    if DEBUG_STATE:
        # param_grid = {'n_estimators': [500], 'max_depth': [8], 'subsample':[0.8], 'colsample_bytree': [0.8]}
        param_grid = {'n_estimators': [50], 'max_depth': [8], 'subsample': [0.8], 'colsample_bytree': [0.6]}
    else:
        param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'subsample': subsample,
                      'colsample_bytree': col_sample, 'reg_lambda': reg_lambda, 'gamma': gamma}
    # 参数搜索
    # 此处min_child_weight=1,与XGBRegressor不同
    kfold = KFold(n_splits=KFOLD, shuffle=True)
    xgbrfr = XGBRFRegressor(booster='gbtree', random_state=RANDOM_STATE, enable_categorical=True)
    if RANDOM_SEARCH:
        param_search = RandomizedSearchCV(xgbrfr, param_distributions=param_grid, n_iter=RANDOM_ITER_TIMES, cv=kfold,
                                          scoring='r2', verbose=2)
    else:
        param_search = GridSearchCV(xgbrfr, param_grid=param_grid, cv=kfold, scoring='r2', verbose=2)
    param_search.fit(train_dataset, train_labels)

    print("---------------------{}的最优模型(XGBoost Random Forest回归)----------------------------".format(
        predict_variable))
    # 输出最优参数
    print("最佳参数:", param_search.best_params_)
    print("最佳R2 score: ", param_search.best_score_)
    print("KFold: {}, 随机搜索次数：{}".format(KFOLD, RANDOM_ITER_TIMES))

    best_XGBRFR = param_search.best_estimator_
    # 对训练数据进行预测
    train_predictions = best_XGBRFR.mapping(train_dataset)
    model_params = save_regressor_result(build_model_id, predict_variable, 'XGBoost随机森林回归', best_XGBRFR, param_search.best_params_,
                                         param_search.best_score_, train_labels, train_predictions,
                                         train_dataset.columns)
    return True # {"model_name": "XGBoost随机森林回归", "R2": model_params["R2"], "RMSE": model_params["RMSE"]}

"""
建立支持向量机的回归模型

Args:
    structure_data_file (str): 结构化数据文件。
    predict_variable (str): 预测变量。
    categorical_variables (list[str]): 类别型变量列表。

Returns:
    是否成功开启了建模过程。
"""


def build_SVR(build_model_id, structure_data_file: str, predict_variable: str, categorical_variables: list[str]) -> dict[
    str, str | float]:
    train_dataset, train_labels, categorical_features_info = prepare_dataset(structure_data_file, predict_variable, categorical_variables)
    # 保存模型相关的类别变量的类别信息，用于预测环节
    save_category_file(build_model_id, categorical_features_info)
    # 定义参数网格
    if DEBUG_STATE:
        param_grid = {'kernel': ['rbf'], 'C': [1], 'gamma': [0.1], 'epsilon': [0.8]}  # only for debug
    else:
        param_grid = {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000],
                      'epsilon': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1], 'gamma': [0.0001, 0.001, 0.01, 0.1]}
        # param_grid = [{'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1], 'epsilon': [0.01, 0.05, 0.1, 0.2]},
        #               {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
        #               {'kernel': ['poly'], 'C': [0.1, 1, 10, 100, 1000], 'degree' : [1, 2, 3, 4, 5], 'gamma': [0.0001, 0.001, 0.01, 0.1]}]
    # 进行网格搜索
    grid_search = GridSearchCV(estimator=SVR(max_iter=10000000), param_grid=param_grid, cv=5, scoring='r2', verbose=2)
    grid_search.fit(train_dataset, train_labels)

    print("---------------------最优模型----------------------------")
    # 输出最佳参数
    print("最佳参数:", grid_search.best_params_)
    print("最佳R2 score: ", grid_search.best_score_)

    best_svr = grid_search.best_estimator_

    # 重新训练一个制图的模型，而不是保存评价性能的模型，所以需要根据搜索到的最优参数，重新对Label数据进行夹持到8位空间中，再进行训练一次
    # cato_model = SVR(**grid_search.best_params_)
    # cato_model.fit(train_dataset, cato_train_labels)

    # 对训练数据进行预测
    train_predictions = best_svr.mapping(train_dataset)
    model_params = save_regressor_result(build_model_id, predict_variable, '支持向量机回归', best_svr, grid_search.best_params_,
                                         grid_search.best_score_, train_labels, train_predictions,
                                         (train_dataset, train_labels))
    return True # {"model_name": "支持向量机回归", "R2": model_params["R2"], "RMSE": model_params["RMSE"]}


'''
将.csv中的所有数据作为数据集
'''
def prepare_dataset(structure_data_file:str, prediction_variable:str, interpretation_variables:list[str], categorical_features:list[str]):
    #导入数据
    df = pd.read_csv(structure_data_file, usecols=interpretation_variables + [prediction_variable]) # 读取所有的解释变量列和预测变量列
    train_dataset = df.sample(frac=1, random_state = RANDOM_STATE).reset_index(drop=True) # 注意，必须要shuffle，否则效果很差，因为.csv中的数据是原始排序的,具有区域性

    #获取训练集的目标变量Y
    train_labels = train_dataset.pop(prediction_variable)

    # 如果训练集涉及无序类别变量，将类别变量的类型从数值转换为类别
    categorical_features_info = {}
    for key in categorical_features:
        unique_values = train_dataset[key].unique()
        train_dataset[key] = pd.Categorical(train_dataset[key], categories=unique_values)

        # 还需记录下每个类别变量的具体类别值
        unique_value = natsorted(unique_values.astype(str)) # 转换为字符串、取唯一值，并按照字符串自然顺序排序
        categorical_features_info[key] = ','.join(unique_value)

    # 对连续型的列进行z-score标准化处理
    for key in train_dataset.columns:
        if key not in categorical_features: # 仅针对连续型环境变量处理
            train_dataset[key] = (train_dataset[key] - train_dataset[key].mean()) / train_dataset[key].std()

    return train_dataset, train_labels, categorical_features_info

def save_regressor_result(build_model_id:str, prop_name:str, algorithms:str, best_model, best_param,
                          best_score, data_labels, data_predictions, importance_param)->dict[str,str|float]:
    # 计算均方误差 (MSE) 、均方根误差（RMSE）和决定系数 (R2)
    MSE = mean_squared_error(data_labels, data_predictions)
    RMSE = MSE ** 0.5
    R2 = r2_score(data_labels, data_predictions)
    print("均方误差 (MSE):", round(MSE,4))
    print('均方根误差(RMSE): ', round(RMSE,4))
    print("决定系数 (R2):", round(R2,4))

    if algorithms == '多层感知机' or algorithms == '支持向量机回归':
        # SHAP属性重要性
        # explainer = shap.KernelExplainer(model.predict, importance_param[0])  # 使用样本数据以减少计算时间
        # shap_values = explainer.shap_values(shap.sample(importance_param[0], 50))
        # shap.summary_plot(shap_values)
        # 置换特征重要性
        result = permutation_importance(best_model, importance_param[0], importance_param[1], n_repeats=20, random_state=RANDOM_STATE)
        sorted_idx = result.importances_mean.argsort() # 升序排列
        # fig, ax = plt.subplots()
        # ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=importance_param[0].columns[sorted_idx])
        # ax.set_title("Permutation Importances")
        # fig.tight_layout()
        # plt.show()
        # print(sorted_idx)
        # print(result)
        importance_list = []
        for i in range(len(sorted_idx)-1, -1, -1): # 倒序，记录下特征重要性均值
            importance_dict = {}
            importance_dict[importance_param[0].columns[i]] = result.importances_mean[sorted_idx[i]]
            importance_list.append(importance_dict)
        importances_json = json.dumps(importance_list, ensure_ascii=False)
    else:
        # 通过的feature_importances_属性，我们来查看模型的特征重要性：
        importances = best_model.feature_importances_  # 获取特征重要性

        importance_data = {}
        for i in range(len(importances)):
            importance_data[importance_param[i]]= float(importances[i])
        items = importance_data.items()
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True) # 倒序排列
        sorted_dict = dict(sorted_items)
        importances_json = json.dumps(sorted_dict, ensure_ascii=False)

    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 保存用来制图的模型
    with open(os.path.join(model_dir, "{}-{}-{}.pkl".format(build_model_id, prop_name, algorithms)), 'wb') as file:
        pickle.dump(best_model, file)

    param_str = ''
    for key, value in best_param.items():
        param_str += key
        param_str += ':'
        param_str += str(value)
        param_str += ','
    param_str = param_str[:-1]
    # 保存R2和模型参数
    data = {'best_score': best_score,'R2': R2, 'RMSE': RMSE, 'ModelParams': param_str, 'importance':importances_json}
    with open(os.path.join(model_dir, '{}-{}-{}.json'.format(build_model_id, prop_name, algorithms)), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
    return {"MSE":MSE,"RMSE":RMSE,"R2":R2,
            "模型信息文件":os.path.join(model_dir, '{}-{}-{}.json'.format(build_model_id, prop_name, algorithms))}
