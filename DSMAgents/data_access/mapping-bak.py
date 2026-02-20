import os.path
import glob
from osgeo import gdal
import numpy as np
import shutil
import pandas as pd
from DSMAlgorithms.base.base_data_structure import AlgorithmsType
from DSMAlgorithms.base.dsm_models import (XGBRModel, RandomForestRegressionModel, SVRModel, KNRModel, PLSRModel,
                                           GLMModel,
                                           MGWRModel, ElasticNetModel, CoKrigeModel, RegressionKrigeModel, MLPModel,
                                           StackingModel)
import config
from data_access.build_model import BuildModelDataAccess
from agents.utils.views import print_agent_output, print_caller_parameters
from agents.data_structure.base_data_structure import RegressionParams, MappingMetrics, EvaluatingMetrics
import psycopg2, os
import pickle
from data_access.db import PostgresAccess
from sklearn.metrics import r2_score, root_mean_squared_error
import uuid
from DSMAlgorithms import Uncertainty
from DSMAlgorithms import prepare_model_train_dataset
from eda.analysis import EsdaAnalysisResult
from agents.utils.views import UncertaintyType
from data_access.uncertainty_map import UncertaintyMappingDataAccess

DATA_OUTPUT_ROOT_DIR = r'E:\data\DSM_Test'  # 存放制图结果的根目录

'''
根据算法ID获取相应的制图结果文件路径
'''


def get_raster_map_file(algorithms_id: str):
    task_id = BuildModelDataAccess.retrive_taskID_of_algorithms(algorithms_id)
    task_dir = os.path.join(DATA_OUTPUT_ROOT_DIR, task_id)
    return os.path.join(task_dir, algorithms_id + '.tif')  # 目标文件的完整路径


'''
执行栅格预测

栅格文件统一生成在固定目录下，其根目录为DATA_OUTPUT_ROOT_DIR，在该目录下，按照如下结构生成目标栅格文件
DATA_OUTPUT_ROOT_DIR/[TaskID]/[algorithms_id].*
'''


def mapping(task_id: str, algorithms_id: str, mapping_area_file: str,
            covariates_path: str, build_model_params: RegressionParams):
    print_caller_parameters()
    if not os.path.exists(DATA_OUTPUT_ROOT_DIR):
        os.mkdir(DATA_OUTPUT_ROOT_DIR)
    task_dir = os.path.join(DATA_OUTPUT_ROOT_DIR, task_id)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    target_prediction_file_name = os.path.join(task_dir, algorithms_id)  # 目标文件的完整路径

    # 构建预测模型。注意：预测模型必须采用经过重写的模型，而不能采用原始的回归器，重新的模型会在predict方法中，根据模型特性，对数据进行如下处理后再进行预测：
    # 1、平均编码，或xgboost的类别化
    # 2、z-score/min-max
    # 3、删除坐标列（针对无需geometry的算法）
    algorithms_type, model, mean_encoder, transform_info = BuildModelDataAccess.retrive_model(algorithms_id)
    if isinstance(algorithms_type, list):
        prediction_model = StackingModel(model, transform_info, mean_encoder)
    else:
        if algorithms_type == AlgorithmsType.RK:
            prediction_model = RegressionKrigeModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.CK:
            prediction_model = CoKrigeModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.MGWR:
            prediction_model = MGWRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.EN:
            prediction_model = ElasticNetModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.GLM:
            prediction_model = GLMModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.PLSR:
            prediction_model = PLSRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.KNR:
            prediction_model = KNRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.SVR:
            prediction_model = SVRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.MLP:
            prediction_model = MLPModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.RFR:
            prediction_model = RandomForestRegressionModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.XGBR:
            prediction_model = XGBRModel(model, transform_info)  # xgboost库的模型可以自己处理类别变量，无需平均编码

    # 是否需要坐标值作为预测的输入项
    need_coordinates_for_prediction = False
    if isinstance(algorithms_type,
                  list) or algorithms_type == AlgorithmsType.MGWR or algorithms_type == AlgorithmsType.RK or algorithms_type == AlgorithmsType.CK:  # 对于几种特定算法，需要坐标作为输入
        need_coordinates_for_prediction = True

    TEMPLATE_TIFF_FILE, file_extent_name = os.path.splitext(mapping_area_file)
    CARTOGRAPHY_AREA_FILE_DIRECTORY = os.path.dirname(mapping_area_file)  # 制图区域文件所在的目录

    # 打开模板tiff图片
    template_dataset = gdal.Open(mapping_area_file)  # 必须位于当前目录
    template_band = template_dataset.GetRasterBand(1)
    template_tif_nodata_value = template_band.GetNoDataValue()  # 获取模板图像中表示NoData的数值
    if template_tif_nodata_value is None:
        print('模板图像未设置NoData值')
        return None

    # 将模板TIF中的有值区域行列号进行序列化，以加快后续处理速度
    cached_row_col_file = os.path.join(task_dir, 'cached_row_col.pkl')
    if not os.path.isfile(cached_row_col_file):
        template_tif_data = template_band.ReadAsArray()
        pixels_row_col = []
        for row in range(template_dataset.RasterYSize):  # 扫描每一行
            if config.DEBUG and row > 10:  # only for debug
                continue
            for col in range(template_dataset.RasterXSize):  # 扫描每一列
                # 检查模板图像中的当前像素值是否为nodata
                if template_tif_data[row, col] == template_tif_nodata_value:
                    continue
                pixels_row_col.append((row, col))  # 记录下相应的像素的行列位置
        with open(cached_row_col_file, 'wb') as file:
            pickle.dump(pixels_row_col, file)  # 将数据序列化并写入文件
        print('成功缓存模板TIF有值区域数据')

    # 获取tiff图片的像素值
    target_tif_data = np.zeros([template_dataset.RasterYSize, template_dataset.RasterXSize], np.dtype('f4'))  # 生成空数组

    # 将模板文件进行复制，形成一个副本栅格图文件
    data_dir = os.path.join(task_dir, 'template_data')
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    pattern = os.path.join(data_dir, f"{TEMPLATE_TIFF_FILE}*")
    all_matches = glob.glob(pattern)
    # 过滤出文件（排除目录）
    template_file_list = [f for f in all_matches if os.path.isfile(f)]
    for template_file in template_file_list:
        if template_file.endswith('.pkl'):
            continue
        template_file_name = os.path.basename(template_file)
        _, file_extension = os.path.splitext(template_file_name)
        # 将模板文件作为预测的模板文件，后续打开该文件进行写入
        shutil.copy2(os.path.join(CARTOGRAPHY_AREA_FILE_DIRECTORY, template_file_name),
                     os.path.join(task_dir, target_prediction_file_name + file_extension))

    target_tiff_file = target_prediction_file_name + ".tif"
    target_dataset = gdal.Open(target_tiff_file, gdal.GA_Update)  # 必须位于当前目录
    band_target = target_dataset.GetRasterBand(1)

    template_tif_data = template_band.ReadAsArray()
    template_geo_transform = template_dataset.GetGeoTransform()  # 获取模板TIF影像的地理变换

    # 遍历每一个环境变量的tiff图片
    # 1、如果环境变量有效区域尚未被缓存过，则打开原始TIF进行读取,并将每个环境变量有效区域内的值序列化至文件保存，以便下次快速装载
    for i in range(len(build_model_params.interpretation_variables)):  # 遍历每个环境变量
        current_env_name = build_model_params.interpretation_variables[i]
        env_cached_file = os.path.join(task_dir, current_env_name + '.pkl')
        if os.path.exists(env_cached_file):
            print('发现缓存的环境变量：{}'.format(current_env_name))
            continue
        # 检测坐标的缓存文件是否存在,如果不存在，则新建坐标缓存文件
        coords_cached_file_exists = os.path.exists(os.path.join(task_dir, "coords.pkl"))
        coord_x_list = []  # 存储每个模板像素的X坐标
        coord_y_list = []  # 存储每个模板像素的Y坐标

        current_env_tif_data_list = []  # 记录当前环境变量图层中每个有值像素的值
        # 1、打开当前环境变量TIF图层，准备必要的信息
        env_var_file = get_env_var_file_name(covariates_path, build_model_params.interpretation_variables[i])
        if env_var_file is None:
            print('环境变量（{}）对应的栅格文件不存在'.format(build_model_params.interpretation_variables[i]))
            return None
        env_dataset = gdal.Open(env_var_file)
        env_geo_transform = env_dataset.GetGeoTransform()
        env_no_data_value = env_dataset.GetRasterBand(1).GetNoDataValue()
        if env_no_data_value is None:
            print('环境变量图像（{}）未设置NoData值'.format(build_model_params.interpretation_variables[i]))
            return None
        env_tif_full_data = env_dataset.GetRasterBand(1).ReadAsArray()

        min_value, max_value, _, _ = env_dataset.GetRasterBand(1).GetStatistics(False, True)

        print('开始处理环境变量有值区域数据:{}'.format(build_model_params.interpretation_variables[i]))
        # 2、遍历模板中的每个有效像素区域，确定对应的环境变量图层的每个像素值
        for row in range(template_dataset.RasterYSize):  # 扫描每一行
            if config.DEBUG and row > 10:  # only for debug
                continue
            for col in range(template_dataset.RasterXSize):  # 扫描每一列
                pixel_value = template_tif_data[row, col]
                # 检查模板图像中的当前像素值是否为nodata
                if pixel_value == template_tif_nodata_value:
                    continue  # 如果像素值为NoData，则跳过处理

                # 将模板像素对应于当前环境变量的像素
                env_pixel_coord = pixel_2_world_2_pixel(col, row, template_geo_transform, env_geo_transform)
                # 如果计算出的像素坐标超过了当前环境变量栅格图层的像素范围，则记录
                if env_pixel_coord[0] >= env_dataset.RasterXSize or env_pixel_coord[1] >= env_dataset.RasterYSize:
                    print(
                        f"模板图像中row:{row},col:{col}处的像素未能找到对应的环境变量像素点，环境变量图为：{build_model_params.interpretation_variables[i]}")
                    continue

                # 获取环境变量栅格图中指定像素位置的值
                pixel_value = env_tif_full_data[env_pixel_coord[1], env_pixel_coord[0]]
                # 检测环境变量图层指定像素的值是否为有意义的值
                if env_no_data_value == pixel_value:
                    # print("环境变量图中row:{},col:{}处值为NoData：{}".format(env_pixel_coord[1], env_pixel_coord[0], env_var_tiffs[i]))
                    # return False  # 像素点无有效值，则跳过，因为无法取值用于预测
                    pixel_value = min_value  # 用最小值作为nodata区域的值

                current_env_tif_data_list.append(pixel_value)  # 记录环境变量像素的值

                # 对于诸如rk,rk,mgwr这样的模型，还需要坐标作为输入
                if not coords_cached_file_exists:  # 仅在第一个环境变量遍历时，记录X和Y坐标
                    template_pixel_coord_x, template_pixel_coord_y = pixel_2_world(col, row, template_geo_transform)
                    coord_x_list.append(template_pixel_coord_x)
                    coord_y_list.append(template_pixel_coord_y)
            if row % 1000 == 0:
                print("收集完第{}行像素的输入环境变量值".format(row))

        # 在处理第一个环境变量的时候，记录下坐标（回归克里金，协同克里金，MGWR等需要）,对于每一个属性，需要有一个坐标缓存文件
        if not coords_cached_file_exists:
            with open(os.path.join(task_dir, "coords.pkl"), 'wb') as file:
                pickle.dump([coord_x_list, coord_y_list], file)  # 将数据序列化并写入文件
            print('成功缓存坐标数据')
        with open(env_cached_file, 'wb') as file:
            pickle.dump(current_env_tif_data_list, file)  # 将数据序列化并写入文件
        print('成功缓存环境变量有值区域数据:{}'.format(env_cached_file))

    # 2、检测是否有缓存的堆叠环境变量X
    find_cached_envs_arr = False
    search_envs = ','.join(build_model_params.interpretation_variables)
    meta_file_name = 'stack_envs_meta.pkl'
    cached_stack_env_arr_file = os.path.join(task_dir, meta_file_name)
    if os.path.exists(cached_stack_env_arr_file):
        with open(cached_stack_env_arr_file, 'rb') as file:
            env_name_dict = pickle.load(file)
            if search_envs in env_name_dict:
                find_cached_envs_arr = True
    if find_cached_envs_arr:  # 存在匹配的缓存下来的X
        df_pixel_X = pd.read_pickle(os.path.join(task_dir, env_name_dict[search_envs]))
        print('发现缓存的用于预测的X，从磁盘缓存加载成功')
    else:   # 不存在，则新建X的缓存
        env_tif_data = []  # 存储环境变量栅格图的有效数据区域的数据值,第一维为环境变量索引，第二维为相应环境变量图层的数据值数组
        for i in range(len(build_model_params.interpretation_variables)):  # 遍历每个环境变量
            current_env_name = build_model_params.interpretation_variables[i]
            env_cached_file = os.path.join(task_dir, current_env_name + '.pkl')
            with open(env_cached_file, 'rb') as file:
                env_tif_data.append(pickle.load(file))
        # 4、拼接缓存的环境变量有值数据，以像素为单位建立X（dataframe的每一行是一个像素）
        # 出于性能考虑，预先创建一个足够大的numpy数组，数组中的每一行存储需要进行预测的各个环境变量的属性值，即要参与预测的X
        print('开始拼接环境变量有值数据...')
        np_tiff_data = np.column_stack(tuple(env_tif_data))

        print('开始处理类别变量...')
        # 5、将本次计算中涉及的类别型的列设置为Category类型
        df_pixel_X = pd.DataFrame(np_tiff_data, columns=build_model_params.interpretation_variables)
        # 如果本次预测的输入特征涉及无序类别变量，则将数据中的相应列转换为类别变量后，再传入预测模型
        for key, value in build_model_params.categorical_vars_detail.items():
            if key in df_pixel_X.columns:
                df_pixel_X[key] = pd.Categorical(df_pixel_X[key], categories=list(map(int, value.split(','))))

        # 6、将dataframe进行缓存，供后续使用
        if os.path.exists(cached_stack_env_arr_file):  # 如果存在，先读取
            with open(cached_stack_env_arr_file, 'rb') as file:
                env_name_dict = pickle.load(file)
        else:
            env_name_dict = {}
        env_name_dict[search_envs] = str(uuid.uuid4()) + '.pkl'  # 用uuid作为缓存的文件名，用字典存储元数据
        # dataframe缓存下来--持久化至磁盘
        df_pixel_X.to_pickle(os.path.join(task_dir, env_name_dict[search_envs]))
        # 元数据持久化到磁盘
        with open(cached_stack_env_arr_file, 'wb') as file:
            pickle.dump(env_name_dict, file)  # 将数据序列化并写入文件
        print('合成用于预测的X，并持久化至磁盘成功')

    # 最后记录一下X坐标和Y坐标（回归克里金，协同克里金，MGWR等需要）
    if need_coordinates_for_prediction:
        df_coordinates = pd.read_pickle(os.path.join(task_dir, "coords.pkl"))
        # env_tif_data.append(df_coordinates[0])
        # env_tif_data.append(df_coordinates[1])
        df_pixel_X['coordinates_x'] = df_coordinates[0]
        df_pixel_X['coordinates_y'] = df_coordinates[1]

    # print('开始拼接环境变量有值数据...')
    # np_tiff_data = np.column_stack(tuple(env_tif_data))

    # 将numpy数组转换为dataframe，并赋予列名,而且，对于需要坐标值的列，设置坐标值列的名称
    # cols = build_model_params.interpretation_variables
    # if need_coordinates_for_prediction:
    #     cols = cols + ['coordinates_x', 'coordinates_y']
    # df_pixel_X = pd.DataFrame(np_tiff_data, columns=cols)



    print('开始处理类别变量...')
    # 5、将本次计算中涉及的类别型的列设置为Category类型
    # 如果本次预测的输入特征涉及无序类别变量，则将数据中的相应列转换为类别变量后，再传入预测模型
    for key, value in build_model_params.categorical_vars_detail.items():
        if key in df_pixel_X.columns:
            df_pixel_X[key] = pd.Categorical(df_pixel_X[key], categories=list(map(int, value.split(','))))

    print("开始预测...")
    # 列排序--确保预测传入的X和建模时的X中的列顺序保持一致。
    # 机制：建模时的数据集在prepare_dataset方法的最后做了排序，此处也采用列名称的字符序列排序，并且对于需要带坐标的列时，坐标值为最后两列
    df_pixel_X_sorted = df_pixel_X.sort_index(axis=1)
    if need_coordinates_for_prediction:
        new_order = list(df_pixel_X_sorted.columns)
        new_order.remove(config.CSV_GEOM_COL_X)  # 先删除
        new_order.remove(config.CSV_GEOM_COL_Y)
        new_order.append(config.CSV_GEOM_COL_X)  # 确保坐标列为最后两列，预测方法依赖这一设定
        new_order.append(config.CSV_GEOM_COL_Y)
        df_pixel_X_sorted = df_pixel_X_sorted.reindex(columns=new_order)  # 确保坐标值列放在最后

    # 对于多重共线性敏感的算法，删除建模时不使用的环境协变量列
    if (algorithms_type == AlgorithmsType.EN or algorithms_type == AlgorithmsType.PLSR or algorithms_type == AlgorithmsType.RK
            or algorithms_type == AlgorithmsType.GLM or algorithms_type == AlgorithmsType.SVR
            or (isinstance(algorithms_type,
                           list) and (AlgorithmsType.EN in algorithms_type or AlgorithmsType.PLSR in algorithms_type
                                      or AlgorithmsType.RK in algorithms_type or AlgorithmsType.GLM in algorithms_type or AlgorithmsType.SVR in algorithms_type))):
        used_covariate = BuildModelDataAccess.retrive_used_covariate(algorithms_id)
        columns = df_pixel_X_sorted.columns
        for col_name in columns:
            if col_name != config.CSV_GEOM_COL_X and col_name != config.CSV_GEOM_COL_Y and col_name not in used_covariate:
                df_pixel_X_sorted.pop(col_name)
    # 预测
    predict_result = prediction_model.predict(df_pixel_X_sorted)

    prediction_metrics = MappingMetrics()
    prediction_metrics.algorithms_id = algorithms_id
    prediction_metrics.min_value = float(np.min(predict_result))
    prediction_metrics.max_value = float(np.max(predict_result))
    prediction_metrics.mean_value = float(np.mean(predict_result))
    prediction_metrics.algorithms_type = algorithms_type
    print('预测TIF像素的最小值：{:.4f},最大值：{:.4f},均值：{:.4f}'.format(prediction_metrics.min_value,
                                                                        prediction_metrics.max_value,
                                                                        prediction_metrics.mean_value))
    # 写入TIFF
    print("回归预测结束，开始写入目标TIF...")
    # 将预测结果赋值给numpy数组中的第row行的第row_pixel_index_list[i]列
    with open(cached_row_col_file, 'rb') as file:
        pixel_index_list = pickle.load(file)  # 对应于每个像素在行中的索引位置

    for i in range(predict_result.shape[0]):  # 逐个像素赋值,也可以使用 len(pixel_index_list)
        target_tif_data[pixel_index_list[i][0], pixel_index_list[i][1]] = predict_result[i]
    band_target.WriteArray(target_tif_data)
    band_target.ComputeBandStats()
    band_target.FlushCache()
    target_dataset = None  # 关闭目标Tiff
    print_agent_output("生成：" + target_tiff_file, agent="EVALUATION")
    return prediction_metrics


'''
生成不确定性分布图
'''


def create_uncertainty_map(task_id: str, algorithms_id: str, sample_file: str, mapping_area_file: str,
                           covariates_path: str, uncertainty_metrics_type: list[UncertaintyType],
                           build_model_params: RegressionParams, esda_result: EsdaAnalysisResult):
    print_caller_parameters()
    task_dir = os.path.join(DATA_OUTPUT_ROOT_DIR, task_id)
    # 构建预测模型。注意：预测模型必须采用经过重写的模型，而不能采用原始的回归器，重新的模型会在predict方法中，根据模型特性，对数据进行如下处理后再进行预测：
    # 1、平均编码，或xgboost的类别化
    # 2、z-score/min-max
    # 3、删除坐标列（针对无需geometry的算法）
    algorithms_type, model, mean_encoder, transform_info = BuildModelDataAccess.retrive_model(algorithms_id)
    if isinstance(algorithms_type, list):
        prediction_model = StackingModel(model, transform_info, mean_encoder)
    else:
        if algorithms_type == AlgorithmsType.RK:
            prediction_model = RegressionKrigeModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.CK:
            prediction_model = CoKrigeModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.MGWR:
            prediction_model = MGWRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.EN:
            prediction_model = ElasticNetModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.GLM:
            prediction_model = GLMModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.PLSR:
            prediction_model = PLSRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.KNR:
            prediction_model = KNRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.SVR:
            prediction_model = SVRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.MLP:
            prediction_model = MLPModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.RFR:
            prediction_model = RandomForestRegressionModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.XGBR:
            prediction_model = XGBRModel(model, transform_info)  # xgboost库的模型可以自己处理类别变量，无需平均编码

    # 是否需要坐标值作为预测的输入项
    need_coordinates_for_prediction = False
    if isinstance(algorithms_type,
                  list) or algorithms_type == AlgorithmsType.MGWR or algorithms_type == AlgorithmsType.RK or algorithms_type == AlgorithmsType.CK:  # 对于几种特定算法，需要坐标作为输入
        need_coordinates_for_prediction = True

    TEMPLATE_TIFF_FILE, file_extent_name = os.path.splitext(mapping_area_file)
    CARTOGRAPHY_AREA_FILE_DIRECTORY = os.path.dirname(mapping_area_file)  # 制图区域文件所在的目录

    # 打开模板tiff图片
    template_dataset = gdal.Open(mapping_area_file)  # 必须位于当前目录
    template_band = template_dataset.GetRasterBand(1)
    template_tif_nodata_value = template_band.GetNoDataValue()  # 获取模板图像中表示NoData的数值
    if template_tif_nodata_value is None:
        print('模板图像未设置NoData值')
        return None

    # 将模板TIF中的有值区域行列号进行序列化，以加快后续处理速度
    cached_row_col_file = os.path.join(task_dir, 'cached_row_col.pkl')
    if not os.path.isfile(cached_row_col_file):
        template_tif_data = template_band.ReadAsArray()
        pixels_row_col = []
        for row in range(template_dataset.RasterYSize):  # 扫描每一行
            if config.DEBUG and row > 10:  # only for debug
                continue
            for col in range(template_dataset.RasterXSize):  # 扫描每一列
                # 检查模板图像中的当前像素值是否为nodata
                if template_tif_data[row, col] == template_tif_nodata_value:
                    continue
                pixels_row_col.append((row, col))  # 记录下相应的像素的行列位置
        with open(cached_row_col_file, 'wb') as file:
            pickle.dump(pixels_row_col, file)  # 将数据序列化并写入文件
        print('成功缓存模板TIF有值区域数据')

    # 获取tiff图片的像素值
    target_tif_data = np.zeros([template_dataset.RasterYSize, template_dataset.RasterXSize], np.dtype('f4'))  # 生成空数组

    # 将模板文件进行复制，形成一个副本栅格图文件
    data_dir = os.path.join(task_dir, 'template_data')
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    pattern = os.path.join(data_dir, f"{TEMPLATE_TIFF_FILE}*")
    all_matches = glob.glob(pattern)
    # 过滤出文件（排除目录）
    template_file_list = [f for f in all_matches if os.path.isfile(f)]

    template_tif_data = template_band.ReadAsArray()
    template_geo_transform = template_dataset.GetGeoTransform()  # 获取模板TIF影像的地理变换
    # 遍历每一个环境变量的tiff图片
    env_tif_data = []  # 存储环境变量栅格图的有效数据区域的数据值,第一维为环境变量索引，第二维为相应环境变量图层的数据值数组
    coord_x_list = []  # 存储每个模板像素的X坐标
    coord_y_list = []  # 存储每个模板像素的Y坐标
    # 1、如果环境变量有效区域尚未被缓存过，则打开原始TIF进行读取,并将每个环境变量有效区域内的值序列化至文件保存，以便下次快速装载
    for i in range(len(build_model_params.interpretation_variables)):  # 遍历每个环境变量
        current_env_name = build_model_params.interpretation_variables[i]
        env_cached_file = os.path.join(task_dir, current_env_name + '.pkl')
        if os.path.exists(env_cached_file):
            print('发现缓存的环境变量：{}'.format(current_env_name))
            continue
        # 检测坐标的缓存文件是否存在,如果不存在，则新建坐标缓存文件
        coords_cached_file_exists = os.path.exists(os.path.join(task_dir, "coords.pkl"))
        coord_x_list = []  # 存储每个模板像素的X坐标
        coord_y_list = []  # 存储每个模板像素的Y坐标

        current_env_tif_data_list = []  # 记录当前环境变量图层中每个有值像素的值
        # 1、打开当前环境变量TIF图层，准备必要的信息
        env_var_file = get_env_var_file_name(covariates_path, build_model_params.interpretation_variables[i])
        if env_var_file is None:
            print('环境变量（{}）对应的栅格文件不存在'.format(build_model_params.interpretation_variables[i]))
            return None
        env_dataset = gdal.Open(env_var_file)
        env_geo_transform = env_dataset.GetGeoTransform()
        env_no_data_value = env_dataset.GetRasterBand(1).GetNoDataValue()
        if env_no_data_value is None:
            print('环境变量图像（{}）未设置NoData值'.format(build_model_params.interpretation_variables[i]))
            return None
        env_tif_full_data = env_dataset.GetRasterBand(1).ReadAsArray()

        min_value, max_value, _, _ = env_dataset.GetRasterBand(1).GetStatistics(False, True)

        print('开始处理环境变量有值区域数据:{}'.format(build_model_params.interpretation_variables[i]))
        # 2、遍历模板中的每个有效像素区域，确定对应的环境变量图层的每个像素值
        for row in range(template_dataset.RasterYSize):  # 扫描每一行
            if config.DEBUG and row > 10:  # only for debug
                continue
            for col in range(template_dataset.RasterXSize):  # 扫描每一列
                pixel_value = template_tif_data[row, col]
                # 检查模板图像中的当前像素值是否为nodata
                if pixel_value == template_tif_nodata_value:
                    continue  # 如果像素值为NoData，则跳过处理

                # 将模板像素对应于当前环境变量的像素
                env_pixel_coord = pixel_2_world_2_pixel(col, row, template_geo_transform, env_geo_transform)
                # 如果计算出的像素坐标超过了当前环境变量栅格图层的像素范围，则记录
                if env_pixel_coord[0] >= env_dataset.RasterXSize or env_pixel_coord[1] >= env_dataset.RasterYSize:
                    print(
                        f"模板图像中row:{row},col:{col}处的像素未能找到对应的环境变量像素点，环境变量图为：{build_model_params.interpretation_variables[i]}")
                    continue

                # 获取环境变量栅格图中指定像素位置的值
                pixel_value = env_tif_full_data[env_pixel_coord[1], env_pixel_coord[0]]
                # 检测环境变量图层指定像素的值是否为有意义的值
                if env_no_data_value == pixel_value:
                    # print("环境变量图中row:{},col:{}处值为NoData：{}".format(env_pixel_coord[1], env_pixel_coord[0], env_var_tiffs[i]))
                    # return False  # 像素点无有效值，则跳过，因为无法取值用于预测
                    pixel_value = min_value  # 用最小值作为nodata区域的值

                current_env_tif_data_list.append(pixel_value)  # 记录环境变量像素的值

                # 对于诸如rk,rk,mgwr这样的模型，还需要坐标作为输入
                if not coords_cached_file_exists:  # 仅在第一个环境变量遍历时，记录X和Y坐标
                    template_pixel_coord_x, template_pixel_coord_y = pixel_2_world(col, row, template_geo_transform)
                    coord_x_list.append(template_pixel_coord_x)
                    coord_y_list.append(template_pixel_coord_y)

            if row % 1000 == 0:
                print("收集完第{}行像素的输入环境变量值".format(row))

        # 在处理第一个环境变量的时候，记录下坐标（回归克里金，协同克里金，MGWR等需要）,对于每一个属性，需要有一个坐标缓存文件
        if not coords_cached_file_exists:
            with open(os.path.join(task_dir, "coords.pkl"), 'wb') as file:
                pickle.dump([coord_x_list, coord_y_list], file)  # 将数据序列化并写入文件
            print('成功缓存坐标数据')
        with open(env_cached_file, 'wb') as file:
            pickle.dump(current_env_tif_data_list, file)  # 将数据序列化并写入文件
        print('成功缓存环境变量有值区域数据:{}'.format(env_cached_file))


    env_tif_data = []  # 存储环境变量栅格图的有效数据区域的数据值,第一维为环境变量索引，第二维为相应环境变量图层的数据值数组
    for i in range(len(build_model_params.interpretation_variables)):  # 遍历每个环境变量
        current_env_name = build_model_params.interpretation_variables[i]
        env_cached_file = os.path.join(task_dir, current_env_name + '.pkl')
        with open(env_cached_file, 'rb') as file:
            env_tif_data.append(pickle.load(file))

    # 最后记录一下X坐标和Y坐标（回归克里金，协同克里金，MGWR等需要）
    if need_coordinates_for_prediction:
        df_coordinates = pd.read_pickle(os.path.join(task_dir, "coords.pkl"))
        env_tif_data.append(df_coordinates[0])
        env_tif_data.append(df_coordinates[1])

    print('开始拼接环境变量有值数据...')
    np_tiff_data = np.column_stack(tuple(env_tif_data))

    # 将numpy数组转换为dataframe，并赋予列名,而且，对于需要坐标值的列，设置坐标值列的名称
    cols = build_model_params.interpretation_variables
    if need_coordinates_for_prediction:
        cols = cols + ['coordinates_x', 'coordinates_y']
    df_pixel_X = pd.DataFrame(np_tiff_data, columns=cols)

    print('开始处理类别变量...')
    # 5、将本次计算中涉及的类别型的列设置为Category类型
    # 如果本次预测的输入特征涉及无序类别变量，则将数据中的相应列转换为类别变量后，再传入预测模型
    for key, value in build_model_params.categorical_vars_detail.items():
        if key in df_pixel_X.columns:
            df_pixel_X[key] = pd.Categorical(df_pixel_X[key], categories=list(map(int, value.split(','))))

    print("开始预测...")
    # 列排序--确保预测传入的X和建模时的X中的列顺序保持一致。
    # 机制：建模时的数据集在prepare_dataset方法的最后做了排序，此处也采用列名称的字符序列排序，并且对于需要带坐标的列时，坐标值为最后两列
    df_pixel_X_sorted = df_pixel_X.sort_index(axis=1)
    if need_coordinates_for_prediction:
        new_order = list(df_pixel_X_sorted.columns)
        new_order.remove(config.CSV_GEOM_COL_X)  # 先删除
        new_order.remove(config.CSV_GEOM_COL_Y)
        new_order.append(config.CSV_GEOM_COL_X)  # 确保坐标列为最后两列，预测方法依赖这一设定
        new_order.append(config.CSV_GEOM_COL_Y)
        df_pixel_X_sorted = df_pixel_X_sorted.reindex(columns=new_order)  # 确保坐标值列放在最后

    # 对于多重共线性敏感的算法，删除建模时不使用的环境协变量列
    if (
            algorithms_type == AlgorithmsType.EN or algorithms_type == AlgorithmsType.PLSR or algorithms_type == AlgorithmsType.RK
            or algorithms_type == AlgorithmsType.GLM or algorithms_type == AlgorithmsType.SVR
            or (isinstance(algorithms_type, list) and (
            AlgorithmsType.EN in algorithms_type or AlgorithmsType.PLSR in algorithms_type
            or AlgorithmsType.RK in algorithms_type or AlgorithmsType.GLM in algorithms_type or AlgorithmsType.SVR in algorithms_type))):
        used_covariate = BuildModelDataAccess.retrive_used_covariate(algorithms_id)
        columns = df_pixel_X_sorted.columns
        for col_name in columns:
            if col_name != config.CSV_GEOM_COL_X and col_name != config.CSV_GEOM_COL_Y and col_name not in used_covariate:
                df_pixel_X_sorted.pop(col_name)
    # 预测
    uncertainty_analysis = Uncertainty(prediction_model)  # 构建不确定性分析器
    # 准备训练的数据
    train_X, train_y, _, _, _ = prepare_model_train_dataset(False,
                                                            sample_file,
                                                            covariates_path,
                                                            42,
                                                            0,  # 注意，这里需使用全部样点作为bootstrap_prediction_interval的输入
                                                            algorithms_type,
                                                            build_model_params.prediction_variable,
                                                            build_model_params.categorical_vars_detail,
                                                            esda_result.left_cols_after_rfe,
                                                            esda_result.left_cols_after_fia,
                                                            esda_result.left_cols_after_rfe_fia,
                                                            esda_result.data_distribution)
    if need_coordinates_for_prediction:  # 如果预测时需要坐标列
        # 将原有的geometry列拆分为两列
        train_X[config.CSV_GEOM_COL_X] = train_X[config.DF_GEOM_COL].apply(lambda coord: coord.x)
        train_X[config.CSV_GEOM_COL_Y] = train_X[config.DF_GEOM_COL].apply(lambda coord: coord.y)
        train_X.pop(config.DF_GEOM_COL)

    evaluating_metrics = EvaluatingMetrics()
    evaluating_metrics.algorithms_id = algorithms_id
    evaluating_metrics.algorithms_type = algorithms_type
    # 进行有放回抽样的间隔预测,注意：由于算法的问题，可能会导致抽样中的拟合和预测失败
    if uncertainty_analysis.bootstrap_prediction_interval(train_X, train_y, df_pixel_X_sorted, bootstrap_times=5):
        with open(cached_row_col_file, 'rb') as file:
            pixel_index_list = pickle.load(file)  # 对应于每个像素在行/列中的索引位置
        # ---------------------开始遍历每一种指标类型，生成相应的分布图
        for metric_type in uncertainty_metrics_type:
            map_id = str(uuid.uuid4())  # 生成不确定性图的ID
            target_prediction_file_name = os.path.join(task_dir, map_id)  # 目标文件的完整路径
            for template_file in template_file_list:
                if template_file.endswith('.pkl'):
                    continue
                template_file_name = os.path.basename(template_file)
                _, file_extension = os.path.splitext(template_file_name)
                # 将模板文件作为预测的模板文件，后续打开该文件进行写入
                shutil.copy2(os.path.join(CARTOGRAPHY_AREA_FILE_DIRECTORY, template_file_name),
                             os.path.join(task_dir, target_prediction_file_name + file_extension))

            target_tiff_file = target_prediction_file_name + ".tif"
            target_dataset = gdal.Open(target_tiff_file, gdal.GA_Update)  # 必须位于当前目录
            band_target = target_dataset.GetRasterBand(1)
            if metric_type == UncertaintyType.STD:
                predict_result = uncertainty_analysis.get_std_error_dist()  # 标准差作为像素值
            elif metric_type == UncertaintyType.Percentile:
                percentile_dist = uncertainty_analysis.get_percentile_dist(alpha = 0.05)
                predict_result = percentile_dist[1] - percentile_dist[0]  # 5%分位数区间值作为像素值
            elif metric_type == UncertaintyType.Confidence95:
                half_width_dist = uncertainty_analysis.get_confidence_interval_half_width_dist()
                predict_result = half_width_dist[1] - half_width_dist[0]    # 95%置信区间半宽分布作为像素值
            elif metric_type == UncertaintyType.VariationCoefficient:
                predict_result = uncertainty_analysis.get_variation_coefficient_dist()  # 变异系数的值作为像素值
            # 写入TIFF
            print("开始写入目标TIF...")
            # 遍历每个有值的像素，将数值写入该像素
            for i in range(len(predict_result)):  # 逐个有值的像素赋值,也可以使用 len(pixel_index_list)
                target_tif_data[pixel_index_list[i][0], pixel_index_list[i][1]] = predict_result[i]
            band_target.WriteArray(target_tif_data)
            band_target.ComputeBandStats()
            band_target.FlushCache()
            target_dataset = None  # 关闭目标Tiff
            BuildModelDataAccess.record_uncertainty_mapping_result(algorithms_id, map_id, metric_type.name)
            print_agent_output("生成：" + target_tiff_file, agent="EVALUATION")

        # 打开土壤属性图TIF文件,计算预测图的最大值、最小值、均值等
        mapping_file = os.path.join(task_dir, algorithms_id + '.tif')  # 地图文件的完整路径
        soil_prop_dataset = gdal.Open(mapping_file)
        soil_prop_band = soil_prop_dataset.GetRasterBand(1)
        soil_prop_tif_nodata_value = soil_prop_band.GetNoDataValue()  # 获取模板图像中表示NoData的数值
        if soil_prop_tif_nodata_value is None:
            print('土壤属性图未设置NoData值')
            return None
        soil_prop_tif_data = soil_prop_band.ReadAsArray()  # 整个属性图中的数值（包括nodata区域）
        soil_prop_mapping_values = []  # 属性图中的有效值
        for i in range(len(pixel_index_list)):
            soil_prop_mapping_values.append(soil_prop_tif_data[pixel_index_list[i][0], pixel_index_list[i][1]])

        print(len(soil_prop_mapping_values))
        print(f"最小值：{np.min(np.array(soil_prop_mapping_values, dtype=float))}")
        print(f"均值：{np.mean(np.array(soil_prop_mapping_values, dtype=float))}")
        print(f"最大值：{np.max(np.array(soil_prop_mapping_values, dtype=float))}")
        evaluating_metrics.PICP = uncertainty_analysis.calculate_PICP(np.array(soil_prop_mapping_values, dtype=float), (1-0.9)/2)  # 计算90%的PICP
        evaluating_metrics.MPIW = uncertainty_analysis.calculate_MPIW()  # 计算MPIW
    return evaluating_metrics  # 注意，由于抽样可能会失败，所以返回结果中不一定包含PICP或MPIW的结算结果


'''
根据算法对应的某一类不确定制图结果文件路径
'''


def get_uncertainty_raster_map_file(algorithms_id: str, map_type:UncertaintyType):
    task_id = BuildModelDataAccess.retrive_taskID_of_algorithms(algorithms_id)
    task_dir = os.path.join(DATA_OUTPUT_ROOT_DIR, task_id)
    map_id = UncertaintyMappingDataAccess.query_uncertainty_map_id(algorithms_id, map_type)
    return os.path.join(task_dir, map_id+".tif")  # 目标文件的完整路径


'''
依据独立验证数据集上的R2，RMSE指标
'''


def compute_indepent_test_metrics(algorithms_id: str, sample_file: str, covariates_path: str,
                                  build_model_params: RegressionParams, esda_result: EsdaAnalysisResult):
    print_caller_parameters()
    algorithms_type, model, mean_encoder, transform_info = BuildModelDataAccess.retrive_model(algorithms_id)
    if isinstance(algorithms_type, list):
        prediction_model = StackingModel(model, transform_info, mean_encoder)
    else:
        if algorithms_type == AlgorithmsType.RK:
            prediction_model = RegressionKrigeModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.CK:
            prediction_model = CoKrigeModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.MGWR:
            prediction_model = MGWRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.EN:
            prediction_model = ElasticNetModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.GLM:
            prediction_model = GLMModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.PLSR:
            prediction_model = PLSRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.KNR:
            prediction_model = KNRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.SVR:
            prediction_model = SVRModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.MLP:
            prediction_model = MLPModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.RFR:
            prediction_model = RandomForestRegressionModel(model, transform_info, mean_encoder)
        elif algorithms_type == AlgorithmsType.XGBR:
            prediction_model = XGBRModel(model, transform_info)  # xgboost库的模型可以自己处理类别变量，无需平均编码
    # 准备训练的数据
    df_pixel_X_sorted, y, _, _, _ = prepare_model_train_dataset(False,
                                                                sample_file,
                                                                covariates_path,
                                                                42,
                                                                0,  # 不分割
                                                                algorithms_type.value,
                                                                build_model_params.prediction_variable,
                                                                build_model_params.categorical_vars_detail,
                                                                esda_result.left_cols_after_rfe,
                                                                esda_result.left_cols_after_fia,
                                                                esda_result.left_cols_after_rfe_fia,
                                                                esda_result.data_distribution)
    # print('开始处理类别变量...')
    # 将本次计算中涉及的类别型的列设置为Category类型
    # 如果本次预测的输入特征涉及无序类别变量，则将数据中的相应列转换为类别变量后，再传入预测模型
    # for key, value in build_model_params.categorical_vars_detail.items():
    #     if key in X.columns:
    #         X[key] = pd.Categorical(X[key], categories=list(map(int, value.split(','))))

    # print("开始预测...")
    # 列排序--确保预测传入的X和建模时的X中的列顺序保持一致。
    # 机制：建模时的数据集在prepare_dataset方法的最后做了排序，此处也采用列名称的字符序列排序，并且对于需要带坐标的列时，坐标值为最后两列
    # df_pixel_X_sorted = X.sort_index(axis=1)
    # 是否需要坐标值作为预测的输入项
    if isinstance(algorithms_type,
                  list) or algorithms_type == AlgorithmsType.MGWR or algorithms_type == AlgorithmsType.RK or algorithms_type == AlgorithmsType.CK:  # 对于几种特定算法，需要坐标作为输入
        # 将原有的geometry列拆分为两列
        df_pixel_X_sorted[config.CSV_GEOM_COL_X] = df_pixel_X_sorted[config.DF_GEOM_COL].apply(lambda coord: coord.x)
        df_pixel_X_sorted[config.CSV_GEOM_COL_Y] = df_pixel_X_sorted[config.DF_GEOM_COL].apply(lambda coord: coord.y)
        df_pixel_X_sorted.pop(config.DF_GEOM_COL)

    # 对于多重共线性敏感的算法，删除建模时不使用的环境协变量列
    # if (
    #         algorithms_type == AlgorithmsType.EN or algorithms_type == AlgorithmsType.PLSR or algorithms_type == AlgorithmsType.RK
    #         or algorithms_type == AlgorithmsType.GLM or algorithms_type == AlgorithmsType.SVR
    #         or (isinstance(algorithms_type,
    #                        list) and (AlgorithmsType.EN in algorithms_type or AlgorithmsType.PLSR in algorithms_type
    #                                   or AlgorithmsType.RK in algorithms_type or AlgorithmsType.GLM in algorithms_type or AlgorithmsType.SVR in algorithms_type))):
    #     used_covariate = BuildModelDataAccess.retrive_used_covariate(algorithms_id)
    #     columns = df_pixel_X_sorted.columns
    #     for col_name in columns:
    #         if col_name != config.CSV_GEOM_COL_X and col_name != config.CSV_GEOM_COL_Y and col_name not in used_covariate:
    #             df_pixel_X_sorted.pop(col_name)
    # 预测
    predict_result = prediction_model.predict(df_pixel_X_sorted)
    RMSE = root_mean_squared_error(predict_result, y)
    R2 = r2_score(predict_result, y)
    return RMSE, R2


'''
根据指定的环节变量文件所在目录以及环境变量名称获取环境变量文件的全路径
'''


def get_env_var_file_name(dir: str, env_var_name: str) -> str:
    if os.path.exists(os.path.join(dir, env_var_name + ".tif")):
        return os.path.join(dir, env_var_name + ".tif")
    if os.path.exists(os.path.join(dir, env_var_name + ".img")):
        return os.path.join(dir, env_var_name + ".img")
    if os.path.exists(os.path.join(dir, env_var_name + ".jpg")):
        return os.path.join(dir, env_var_name + ".jpg")
    if os.path.exists(os.path.join(dir, env_var_name + ".png")):
        return os.path.join(dir, env_var_name + ".png")
    if os.path.exists(os.path.join(dir, env_var_name + ".gif")):
        return os.path.join(dir, env_var_name + ".gif")
    return None


'''
将一个栅格图像上像素位置转换为另外一个栅格图上相应地理位置处的像素位置
'''


def pixel_2_world_2_pixel(x, y, gt, geotransform):
    pixel_x = int((gt[0] + x * gt[1] + y * gt[2] - geotransform[0]) / geotransform[1])  #
    pixel_y = int((gt[3] + x * gt[4] + y * gt[5] - geotransform[3]) / geotransform[5])  #
    return pixel_x, pixel_y


'''
计算栅格图上某一像素位置的地理坐标
'''


def pixel_2_world(x, y, gt):
    """
    根据地理转换参数和像素坐标计算地理坐标。
    参数:
    col (int): 像素列坐标-X坐标。
    row (int): 像素行坐标-Y坐标。
    gt (tuple): 地理转换参数，包含(a, b, c, d, e, f)。
    返回:
    tuple: 包含(x_geo, y_geo)的地理坐标。
    """
    x_geo = gt[0] + x * gt[1] + y * gt[2]
    y_geo = gt[3] + x * gt[4] + y * gt[5]
    return x_geo, y_geo


class MappingDataAccess:
    '''
    记录制图的指标信息(每个算法对应的栅格图的均值，最大值，最小值)
    '''

    @staticmethod
    def record_mapping_metrics(mapping_results: list[MappingMetrics]) -> bool:
        successful = True
        conn = None
        update_data_single = []
        update_data_stacking = []
        for result in mapping_results:
            if isinstance(result.algorithms_type, list):
                update_data_stacking.append(
                    (result.min_value, result.max_value, result.mean_value, result.algorithms_id))
            else:
                update_data_single.append(
                    (result.min_value, result.max_value, result.mean_value, result.algorithms_id))
        try:
            conn = PostgresAccess.get_db_conn()
            cursor = conn.cursor()
            if len(update_data_single) > 0:
                cursor.executemany(
                    '''update algorithms set mapping_min_val=%s,mapping_max_val=%s,mapping_avg_val=%s where algorithms_id=%s''',
                    update_data_single)
            if len(update_data_stacking) > 0:
                cursor.executemany(
                    '''update stacking_algorithms set mapping_min_val=%s,mapping_max_val=%s,mapping_avg_val=%s where stacking_algorithms_id=%s''',
                    update_data_stacking)
            conn.commit()
            cursor.close()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()  # 回滚事务
            print("更新数据时发生错误:", e)
            successful = False
        finally:
            if conn:
                conn.close()
        return successful

    '''
    获取制图的指标信息
    '''

    @staticmethod
    def retrive_mapping_metrics(task_id: str) -> list[dict]:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()

        # 1、先确定该任务最后一次建模的批次ID
        cursor.execute("select build_model_call_id from task where task_id=%s", (task_id,))
        rows = cursor.fetchall()
        build_model_id = rows[0][0]

        # 2、根据建模批次ID，确定单一算法相关的制图指标结果，注意：仅获取进行过制图的
        cursor.execute(
            "select algorithms_id, algorithms_name, model_parameters,mapping_min_val,mapping_max_val,mapping_avg_val from algorithms where build_model_id=%s and mapping_min_val is not null",
            (build_model_id,))
        rows2 = cursor.fetchall()
        all_model_metrics = []
        for row in rows2:
            regression_model_info = {}
            regression_model_info['algorithms_id'] = row[0]
            regression_model_info['algorithms_type'] = AlgorithmsType(row[1])
            model_parameters = pickle.loads(row[2])
            regression_model_info['R2'] = model_parameters["R2"]
            regression_model_info['RMSE'] = model_parameters["RMSE"]
            regression_model_info['mapping_min_value'] = row[3]
            regression_model_info['mapping_max_value'] = row[4]
            regression_model_info['mapping_mean_value'] = row[5]
            all_model_metrics.append(regression_model_info)

        # 3、确定是否执行过堆叠建模，如果有，则获取堆叠ID
        cursor.execute(
            "select stacking_id from stacking where build_model_id=%s order by end_dt desc limit 1",
            (build_model_id,))
        rows3 = cursor.fetchall()
        if len(rows3) > 0:  # 存在堆叠结果（必须存在）
            stacking_id = rows3[0][0]
            # 进一步根据堆叠ID查询所有相关的堆叠结果，注意：仅获取进行过制图的
            cursor.execute(
                "select stacking_algorithms_id, algorithms_name, model_parameters,mapping_min_val,mapping_max_val,mapping_avg_val from stacking_algorithms where stacking_id=%s and mapping_min_val is not null",
                (stacking_id,))
            rows4 = cursor.fetchall()
            for row in rows4:
                regression_model_info = {}
                regression_model_info['algorithms_id'] = row[0]
                regression_model_info['algorithms_type'] = [AlgorithmsType(item) for item in row[1].split('|')]
                model_parameters = pickle.loads(row[2])
                regression_model_info['R2'] = model_parameters["R2"]
                regression_model_info['RMSE'] = model_parameters["RMSE"]
                regression_model_info['mapping_min_value'] = row[3]
                regression_model_info['mapping_max_value'] = row[4]
                regression_model_info['mapping_mean_value'] = row[5]
                all_model_metrics.append(regression_model_info)
        cursor.close()
        conn.close()
        sorted_objects = sorted(all_model_metrics, key=lambda x: x['R2'], reverse=True)  # 按照R2对列表进行排序
        return sorted_objects
