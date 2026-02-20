import os.path
import os
import glob
from osgeo import gdal, gdalconst
import pickle
import numpy as np
import shutil
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler
import matplotlib.pyplot as plt  # 用于绘图
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
from PIL import Image
import uuid
from utility import print_with_color, print_caller_parameters

'''
执行栅格预测
'''


def create_raster(prediction_id:str, call_id:str, model_id:str, raster_template_file:str,
                  interpretation_variables_file_storage_directory:str, env_var_image_names:list[str]):
    print_caller_parameters()
    data_dir = os.path.join(os.getcwd(), 'data')        # 存放数据的目录（例如：用于预测的环境变量数据，矢量图数据）
    env_var_tif_dir = interpretation_variables_file_storage_directory # os.path.join(data_dir, 'env')
    model_dir = os.path.join(os.getcwd(), 'models')     # 存放机器学习模型相关文件的目录，建模结果会放在这个目录下
    result_dir = os.path.join(os.getcwd(), 'result')    # 存放制图结果文件的目录
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 加载模型并进行预测
    pattern = os.path.join(model_dir, f"{model_id}*.pkl")
    all_matches = glob.glob(pattern)
    # 过滤出文件（排除目录）
    model_file_list = [f for f in all_matches if os.path.isfile(f)]
    with open(os.path.join(model_dir, os.path.basename(model_file_list[0])), 'rb') as file:
        model = pickle.load(file)

    TEMPLATE_TIFF_FILE, file_extent_name = os.path.splitext(raster_template_file)

    # 打开模板tiff图片
    template_dataset = gdal.Open(raster_template_file)  # 必须位于当前目录
    template_band = template_dataset.GetRasterBand(1)
    template_tif_nodata_value = template_band.GetNoDataValue()  # 获取模板图像中表示NoData的数值
    if template_tif_nodata_value is None:
        print('模板图像未设置NoData值')
        return False

    # 将模板TIF中的有值区域行列号进行序列化，以加快后续处理速度
    cached_row_col_file = os.path.join(result_dir, call_id + '_row_col.pkl')
    if not os.path.isfile(cached_row_col_file):
        template_tif_data = template_band.ReadAsArray()
        pixels_row_col = []
        for row in range(template_dataset.RasterYSize):  # 扫描每一行
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
    target_prediction_file_name = prediction_id # 目标栅格文件的主文件名（不含扩展名）

    # 将模板文件进行复制，形成一个副本栅格图文件
    pattern = os.path.join(data_dir, f"{TEMPLATE_TIFF_FILE}*")
    all_matches = glob.glob(pattern)
    # 过滤出文件（排除目录）
    template_file_list = [f for f in all_matches if os.path.isfile(f)]
    for template_file in template_file_list:
        if template_file.endswith('.pkl'):
            continue
        template_file_name = os.path.basename(template_file)
        _, file_extension = os.path.splitext(template_file_name)
        shutil.copy2(os.path.join(data_dir, template_file_name),
                     os.path.join(result_dir, target_prediction_file_name + file_extension))

    target_tiff_file = os.path.join(result_dir, raster_template_file)
    target_dataset = gdal.Open(target_tiff_file, gdal.GA_Update)  # 必须位于当前目录
    band_target = target_dataset.GetRasterBand(1)

    # 获取无序类别型的列信息
    category_json_file = os.path.join(model_dir, '{}-category.json'.format(model_id))
    with open(category_json_file, 'r', encoding='utf-8') as file:
        category_data = json.load(file)

    template_tif_data = template_band.ReadAsArray()
    template_geo_transform = template_dataset.GetGeoTransform()  # 获取模板TIF影像的地理变换
    # 遍历每一个环境变量的tiff图片
    env_tif_data = []  # 存储环境变量栅格图的有效数据区域的数据值,第一维为环境变量索引，第二维为相应环境变量图层的数据值数组
    env_var_tif_dir = os.path.join(data_dir, "env")
    # 1、如果环境变量有效区域尚未被缓存过，则打开原始TIF进行读取,并将每个环境变量有效区域内的值序列化至文件保存，以便下次快速装载
    for i in range(len(env_var_image_names)):  # 遍历每个环境变量
        current_env_name = env_var_image_names[i]
        current_env_tif_data_list = []  # 记录当前环境变量图层中每个有值像素的值
        # 1、打开当前环境变量TIF图层，准备必要的信息
        env_var_file = get_env_var_file_name(env_var_tif_dir, env_var_image_names[i])
        if env_var_file is None:
            print('环境变量（{}）对应的栅格文件不存在'.format(env_var_image_names[i]))
            return False
        env_dataset = gdal.Open(env_var_file)
        env_geo_transform = env_dataset.GetGeoTransform()
        env_no_data_value = env_dataset.GetRasterBand(1).GetNoDataValue()
        if env_no_data_value is None:
            print('环境变量图像（{}）未设置NoData值'.format(env_var_image_names[i]))
            return False
        env_tif_full_data = env_dataset.GetRasterBand(1).ReadAsArray()

        # 获取当前环保变量像元的最小值，用以填充nodata区域
        # min_value = np.min(env_tif_full_data)

        # 读取标准差和均值，用于对环境变量数据进行z-score变换处理
        # prop_df = pd.read_csv(prop_name + 'Env2.csv')
        # prop_sample_mean = prop_df[current_env_name].mean()
        # prop_sample_std = prop_df[current_env_name].std()

        min_value, max_value, prop_sample_mean, prop_sample_std = env_dataset.GetRasterBand(1).GetStatistics(False, True)

        print('开始处理环境变量有值区域数据:{}'.format(env_var_image_names[i]))
        # 2、遍历模板中的每个有效像素区域，确定对应的环境变量图层的每个像素值
        for row in range(template_dataset.RasterYSize):  # 扫描每一行
            for col in range(template_dataset.RasterXSize):  # 扫描每一列
                pixel_value = template_tif_data[row, col]
                # 检查模板图像中的当前像素值是否为nodata
                if pixel_value == template_tif_nodata_value:
                    continue  # 如果像素值为NoData，则跳过处理

                # 将模板像素对应于当前环境变量的像素
                env_pixel_coord = pixel_2_world_2_pixel(col, row, template_geo_transform, env_geo_transform)
                # 如果计算出的像素坐标超过了当前环境变量栅格图层的像素范围，则记录
                if env_pixel_coord[0] >= env_dataset.RasterXSize or env_pixel_coord[1] >= env_dataset.RasterYSize:
                    print(f"模板图像中row:{row},col:{col}处的像素未能找到对应的环境变量像素点，环境变量图为：{env_var_image_names[i]}")
                    continue

                # 获取环境变量栅格图中指定像素位置的值
                pixel_value = env_tif_full_data[env_pixel_coord[1], env_pixel_coord[0]]
                # 检测环境变量图层指定像素的值是否为有意义的值
                if env_no_data_value == pixel_value:
                    # print("环境变量图中row:{},col:{}处值为NoData：{}".format(env_pixel_coord[1], env_pixel_coord[0], env_var_tiffs[i]))
                    # return False  # 像素点无有效值，则跳过，因为无法取值用于预测
                    pixel_value = min_value  # 用最小值作为nodata区域的值

                if current_env_name not in category_data:  # 如果不是类别环境变量，则需要进行z-score处理
                    pixel_value = (pixel_value - prop_sample_mean) / prop_sample_std
                current_env_tif_data_list.append(pixel_value)  # 记录环境变量像素的值

            if row % 1000 == 0:
                print("收集完第{}行像素的输入环境变量值".format(row))
        env_tif_data.append(current_env_tif_data_list)

    print('开始拼接环境变量有值数据...')
    np_tiff_data = np.column_stack(tuple(env_tif_data))

    print('开始处理类别变量...')
    # 5、将本次计算中涉及的类别型的列设置为Category类型
    df_pixel_X = pd.DataFrame(np_tiff_data, columns=env_var_image_names)
    # 如果本次预测的输入特征涉及无序类别变量，则将数据中的相应列转换为类别变量后，再传入预测模型
    for key, value in category_data.items():
        if key in df_pixel_X.columns:
            df_pixel_X[key] = pd.Categorical(df_pixel_X[key], categories=list(map(int, value.split(','))))

    # 用模型进行预测
    print("开始预测...")
    df_pixel_X_sorted = df_pixel_X.sort_index(axis=1) # 重要，必须和建模时的变量顺序一致
    predict_result = model.predict(df_pixel_X_sorted)
    print('预测TIF像素的最小值：{:.4f},最大值：{:.4f}'.format(np.min(predict_result), np.max(predict_result)))
    # 写入TIFF
    print("回归预测结束，开始写入目标TIF...")
    # 将预测结果赋值给numpy数组中的第row行的第row_pixel_index_list[i]列
    with open(cached_row_col_file, 'rb') as file:
        pixel_index_list = pickle.load(file)  # 对应于每个像素在行中的索引位置    for i in range(len(pixel_index_list)):  # 逐个像素赋值
    for i in range(len(pixel_index_list)):  # 逐个像素赋值
        target_tif_data[pixel_index_list[i][0], pixel_index_list[i][1]] = predict_result[i]
    band_target.WriteArray(target_tif_data)
    band_target.ComputeBandStats()
    band_target.FlushCache()
    target_dataset = None  # 关闭目标Tiff
    print_with_color("生成：" + target_tiff_file)
    return True

'''
根据指定的环节变量文件所在目录以及环境变量名称获取环境变量文件的全路径
'''
def get_env_var_file_name(dir:str, env_var_name:str) -> str:
    if os.path.exists(os.path.join(dir, env_var_name+".tif")):
        return os.path.join(dir, env_var_name+".tif")
    if os.path.exists(os.path.join(dir, env_var_name + ".img")):
        return os.path.join(dir, env_var_name + ".img")
    if os.path.exists(os.path.join(dir, env_var_name + ".jpg")):
        return os.path.join(dir, env_var_name + ".jpg")
    if os.path.exists(os.path.join(dir, env_var_name+".png")):
        return os.path.join(dir, env_var_name + ".png")
    if os.path.exists(os.path.join(dir, env_var_name+".gif")):
        return os.path.join(dir, env_var_name + ".gif")
    return None


'''
将一个栅格图像上像素位置转换为另外一个栅格图上相应地理位置处的像素位置
'''


def pixel_2_world_2_pixel(x, y, gt, geotransform):
    pixel_x = int((gt[0] + x * gt[1] + y * gt[2] - geotransform[0]) / geotransform[1])  #
    pixel_y = int((gt[3] + x * gt[4] + y * gt[5] - geotransform[3]) / geotransform[5])  #
    return pixel_x, pixel_y