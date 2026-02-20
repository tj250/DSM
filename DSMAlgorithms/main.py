import os
import sys
import json
import uuid

import algorithms_config
from DSMAlgorithms.misc.abandon_mgwr_bm import MGWRWrap, MGWRModel
from DSMAlgorithms.mlp import MLPWrap, MLPModel
from DSMAlgorithms.plsr import PLSRWrap, PLSRModel
from DSMAlgorithms.glm import GLMWrap, GLMModel
from DSMAlgorithms.knr import KNRWrap, KNRModel
from DSMAlgorithms.elastic_net import ElasticNetWrap
from DSMAlgorithms.rk import RegressionKrigeWrap
from DSMAlgorithms.rfr import RandomForestRegressionWrapper
from DSMAlgorithms.ck import CoKrigeWrap
from DSMAlgorithms.xgbr import XGBRWrap
from DSMAlgorithms.xgbrfr import XGBRFRWrap
from DSMAlgorithms.svr import SVRWrap
from DSMAlgorithms.base.dsm_models import (XGBRModel,XGBRFRModel,SVRModel,CoKrigeModel,RegressionKrigeModel,
                                           RandomForestRegressionModel,ElasticNetModel,ElasticNetModel)
from DSMAlgorithms.stacking import StackingWrap
from data.data_dealer import prepare_dataset
from DSMAlgorithms.base.dsm_base_model import AlgorithmsType
from DSMAlgorithms.base.base_data_structure import DataDistributionType


if __name__ == "__main__":
    print('进入程序-----------------------------------------')
    if os.path.exists(os.path.join(os.getcwd(), 'debug.txt')):
        DEBUG_STATE = True  # Debug状态下算法的搜素空间很小，可以快速完成搜索
    else:
        DEBUG_STATE = False
    if os.path.exists(os.path.join(os.getcwd(), 'gridsearch.txt')):
        RANDOM_SEARCH = False  # 随机搜索次数较少，可以快速完成搜索
    else:
        RANDOM_SEARCH = True
    print('参数列表:', str(sys.argv))
    if DEBUG_STATE:
        print('正处于调试模式！！！')
        print('正处于调试模式！！！')
        print('正处于调试模式！！！')
    if RANDOM_SEARCH:
        print('随机搜索模式')
    else:
        print('格网搜索模式')

    # 装载类别变量信息
    category_json_file = os.path.join(os.getcwd(), "category.json")
    with open(category_json_file, 'r', encoding='utf-8') as file:
        category_data = json.load(file)

    if sys.argv[1].lower() == 'en':  # 调试时的命令行参数：en pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        en = ElasticNetWrap(sys.argv[2], category_data)
        en.build(r'D:\PycharmProjects\DSMAlgorithms\models\en1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'en_predict':  # 调试时的命令行参数：en_predict pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        en = ElasticNetModel.init_from_file(AlgorithmsType.EN, r'D:\PycharmProjects\DSMAlgorithms\models\en1.pkl')
        en.predict(train_X)
    elif sys.argv[1].lower() == 'plsr':  # 调试时的命令行参数：plsr pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        plsr = PLSRWrap(sys.argv[2], category_data)
        plsr.build(r'D:\PycharmProjects\DSMAlgorithms\models\pls1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'plsr_predict':  # 调试时的命令行参数：plsr_predict pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        plsr = PLSRModel(r'D:\PycharmProjects\DSMAlgorithms\models\pls1.pkl')
        plsr.predict(train_X)
    elif sys.argv[1].lower() == 'glm':  # 调试时的命令行参数：glr pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        glr = GLMWrap(sys.argv[2], category_data, DataDistributionType.Normal)
        glr.build(r'D:\PycharmProjects\DSMAlgorithms\models\glr1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'glr_predict':  # 调试时的命令行参数：glr_predict pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        glr = GLMModel(r'D:\PycharmProjects\DSMAlgorithms\models\glr1.pkl')
        glr.predict(train_X)
    elif sys.argv[1].lower() == 'mlp':  # 调试时的命令行参数：mlp pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        mlp = MLPWrap(sys.argv[2], category_data)
        mlp.build(r'D:\PycharmProjects\DSMAlgorithms\models\mlp1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'mlp_predict':  # 调试时的命令行参数：mlp_predict pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        mlp = MLPModel(r'D:\PycharmProjects\DSMAlgorithms\models\mlp1.pkl')
        mlp.predict(train_X)
    elif sys.argv[1].lower() == 'knr':  # 调试时的命令行参数：knr pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        knr = KNRWrap(sys.argv[2], category_data)
        knr.build(r'D:\PycharmProjects\DSMAlgorithms\models\knr1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'knr_predict':  # 调试时的命令行参数：knr_predict pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        knr = KNRModel(r'D:\PycharmProjects\DSMAlgorithms\models\knr1.pkl')
        knr.predict(train_X)
    elif sys.argv[1].lower() == 'svr':  # 调试时的命令行参数：svr pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        svr = SVRWrap(sys.argv[2], category_data)
        svr.build(r'D:\PycharmProjects\DSMAlgorithms\models\svr1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'svr_predict':  # 调试时的命令行参数：svr_predict pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        svr = SVRModel(r'D:\PycharmProjects\DSMAlgorithms\models\svr1.pkl')
        svr.predict(train_X)
    elif sys.argv[1].lower() == 'rfr':  # 调试时的命令行参数：rfr pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        rfr = RandomForestRegressionWrapper(sys.argv[2], category_data)
        rfr.build(r'D:\PycharmProjects\DSMAlgorithms\models\rfr1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'rfr_predict':  # 调试时的命令行参数：rfr_predict pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        rfr = RandomForestRegressionModel(r'D:\PycharmProjects\DSMAlgorithms\models\rfr1.pkl')
        rfr.predict(train_X)
    elif sys.argv[1].lower() == 'xgbr':  # 调试时的命令行参数：xgbr pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        xgbr = XGBRWrap(sys.argv[2], category_data)
        xgbr.build(r'D:\PycharmProjects\DSMAlgorithms\models\xgbr1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'xgbr_predict':  # 调试时的命令行参数：xgbr_predict pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        xgbr = XGBRModel(r'D:\PycharmProjects\DSMAlgorithms\models\xgbr1.pkl')
        xgbr.predict(train_X)
    elif sys.argv[1].lower() == 'xgbrfr':  # 调试时的命令行参数：xgbrfr pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        xgbrfr = XGBRFRWrap(sys.argv[2], category_data)
        xgbrfr.build(r'D:\PycharmProjects\DSMAlgorithms\models\xgbrfr1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'xgbrfr_predict':  # 调试时的命令行参数：xgbrfr_predict pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], False)
        xgbrfr = XGBRFRModel(r'D:\PycharmProjects\DSMAlgorithms\models\xgbrfr1.pkl')
        xgbrfr.predict(train_X)
    elif sys.argv[1].lower() == 'stacking':  # 调试时的命令行参数：stacking pH pHEnv2.csv 1
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y, zscore_normalize = prepare_dataset(sys.argv[2], sys.argv[3], category_data, True, False)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        stacking = StackingWrap(sys.argv[2], category_data, [AlgorithmsType.XGBR, AlgorithmsType.RFR], algorithms_config.RANDOM_ITER_TIMES)
        stacking.build("9d93d003-8415-11f0-96ff-0433c205c543", train_X, train_y, test_X, test_y, zscore_normalize)
    elif sys.argv[1].lower() == 'stacking_predict':  # 调试时的命令行参数：stacking_predict pH pHEnv2.csv
        train_X, train_y, test_X, test_y = prepare_dataset(sys.argv[2], sys.argv[3], category_data, True, False)
        cokrige_predict = CoKrigeModel(r'9d93d003-8415-11f0-96ff-0433c205c543')
        result = cokrige_predict.predict(train_X)
    elif sys.argv[1].lower() == 'cokrige':  # 调试时的命令行参数：cokrige pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y, zscore_normalize = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], True)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        cokrige = CoKrigeWrap(sys.argv[2], category_data)
        cokrige.build(r'D:\PycharmProjects\DSMAlgorithms\models\cokrige1.pkl', train_X, train_y, test_X, test_y, zscore_normalize)
    elif sys.argv[1].lower() == 'cokrige_predict':  # 调试时的命令行参数：cokrige_predict pH pHEnv2.csv
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], True)
        cokrige_predict = CoKrigeModel(r'D:\PycharmProjects\DSMAlgorithms\models\cokrige1.pkl')
        result = cokrige_predict.predict(train_X)
    elif sys.argv[1].lower() == 'reg_krige':  # 调试时的命令行参数：reg_krige pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y, zscore_normalize = prepare_dataset(sys.argv[2], sys.argv[3], category_data,True)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        reg_krige = RegressionKrigeWrap(sys.argv[2], category_data)
        reg_krige.build(str(uuid.uuid1()), train_X, train_y, test_X, test_y, zscore_normalize)
    elif sys.argv[1].lower() == 'reg_krige_predict':  # 调试时的命令行参数：reg_krige_predict pH pHEnv2.csv
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], True)
        reg_krige_predict = RegressionKrigeModel(r'D:\PycharmProjects\DSMAlgorithms\models\reg_krige1.pkl')
        result = reg_krige_predict.predict(train_X)
    elif sys.argv[1].lower() == 'mgwr':  # 调试时的命令行参数：mgwr pH pHEnv2.csv
        # 分割数据为训练集和验证集
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], True)
        if test_X is None:  # 不做分割
            test_X = train_X
            test_y = train_y
        mgwr = MGWRWrap(sys.argv[2], category_data)
        mgwr.build(r'D:\PycharmProjects\DSMAlgorithms\models\mgwr1.pkl', train_X, train_y, test_X, test_y)
    elif sys.argv[1].lower() == 'mgwr_predict':  # 调试时的命令行参数：mgwr_predict pH pHEnv2.csv
        train_X, train_y, test_X, test_y = prepare_dataset_after_argument(sys.argv[2], sys.argv[3], True)
        mgwr_predict = MGWRModel.init_from_file(r'D:\PycharmProjects\DSMAlgorithms\models\mgwr1.pkl')
        result = mgwr_predict.predict(train_X)
    else:
        print('错误的参数')
        sys.exit(-100)