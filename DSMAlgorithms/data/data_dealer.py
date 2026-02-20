import json
import os
import algorithms_config
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from data.data_argument import SpatialDataAugmentation
from data.mean_encoder import MeanEncoder
from data.outliers import DataExceptionTest
from DSMAlgorithms.base.base_data_structure import AlgorithmsType, DataDistributionType, \
    DataTransformType
from db_access.algorithms_parameters import AlgorithmsDataAccess

'''
由dataframe生成geo_dataframe,即：在dataframe中增加一个geometry列，并删除原有的自变量列，只保留预测列和geomtry列
'''


def create_geo_dataframe(df, prop_name):
    geometry = []
    for index, row in df.iterrows():
        geometry.append(
            Point(float(row[algorithms_config.CSV_GEOM_COL_X]), float(row[algorithms_config.CSV_GEOM_COL_Y])))
    points_gdf = gpd.GeoDataFrame(df[[prop_name]].copy(), geometry=geometry)  # 创建GeoDataFrame,只保留两列：预测的属性列和坐标值列
    return points_gdf


'''
对dataframe中列的值进行标准化和类别化处理
transorm_type:对数据进行变换的方式，包括：Nochange(不做任何变换)/z-score/normalize
'''


def dataframe_transform(categorical_data: dict, df: pd.DataFrame, prop_name: str, transform_type: DataTransformType):
    # 如果训练集涉及无序类别变量，将类别变量的类型从数值转换为类别
    cate_columns = []
    zscore_normalize = {'transform_type': transform_type.value}
    if categorical_data is not None:  # 有类别数据
        for key, value in categorical_data.items():
            if key in df.columns:
                df[key] = pd.Categorical(df[key], categories=list(map(int, value.split(','))))
                cate_columns.append(key)
    # 对连续型的列进行z-score/normalize标准化处理
    for key in df.columns:
        if key not in cate_columns and key != prop_name and key != algorithms_config.DF_GEOM_COL:  # 仅针对连续型环境变量处理,还需排除预测的目标变量
            if transform_type == DataTransformType.Normalize:  # 需要归一化
                col_min = df[key].min()
                col_max = df[key].max()
                df[key] = (df[key] - col_min) / (col_max - col_min + 1e-8)
                # 必须记录下来，在预测时使用
                zscore_normalize[key] = {'min': float(col_min), 'max': float(col_max)}
            elif transform_type == DataTransformType.ZScore:  # zscore标准化
                col_mean = df[key].mean()
                col_std = df[key].std()
                if col_std == 0:
                    col_std = 0.0000001
                df[key] = (df[key] - col_mean) / col_std
                # 必须记录下来，在预测时使用
                zscore_normalize[key] = {'mean': float(col_mean), 'std': float(col_std)}

    return df, zscore_normalize


'''
根据建模所采用的算法的类型，准备所需数据集
方法内部进行了如下处理：
1、检测是否需要包含geometry列
2、确定对解释变量进行数据变换的方式：不变换/z-score/normalize，并执行必要的数据变换
3、剔除多重共线性的列

注意：
当algorithms_type为STACKING时，algorithms_name为包含两个key-value的字典,例如{AlgorithmsType.CUSTOM:"支持向量回归(poly核)",AlgorithmsType.EN:"ElasticNet"}
当algorithms_type为非STACKING时，algorithms_name为包含单个key-value的字典,例如：
{"支持向量回归(poly核)":AlgorithmsType.CUSTOM}或者{"ElasticNet":AlgorithmsType.EN},前者为自定义的扩展模型，或者为内置的模型
'''


def prepare_model_train_dataset(use_data_argument: bool, sample_file: str, covariates_path: str, random_state: int,
                                test_dataset_size, algorithms_type:AlgorithmsType, algorithms_name: str|list[str],
                                prediction_variable: str, categorical_vars: dict[str, str],
                                left_cols1: list[str], left_cols2: list[str], left_cols3: list[str],
                                data_distribution_type: DataDistributionType = DataDistributionType.Unknown):
    if algorithms_type == AlgorithmsType.STACKING:  # 堆叠建模
        need_geometry = True  # 堆叠算法，均需包含geometry列
        # 确定数据变换方式:不做变换/进行Z-score标准化/进行normalize
        if (AlgorithmsType.MLP.value in algorithms_name or AlgorithmsType.PLSR.value in algorithms_name):  # MLP,PLSR采用min-max归一化
            transform_type = DataTransformType.Normalize
        elif (AlgorithmsType.EN.value in algorithms_name or AlgorithmsType.CK.value in algorithms_name
              or AlgorithmsType.GLM.value in algorithms_name or AlgorithmsType.MGWR.value in algorithms_name
              or AlgorithmsType.RK.value in algorithms_name or AlgorithmsType.SVR.value in algorithms_name
              or AlgorithmsType.KNR.value in algorithms_name):  # EN、CK、GLM,MGWR,RK,SVR，KNR,采用z-score归一化
            transform_type = DataTransformType.ZScore
        else:  # RF，XGBR,XGBRFR,无需进行任何变换(决策树、随机森林、朴素贝叶斯、XGBoost、LightGBM等)
            transform_type = DataTransformType.Nochange
    elif algorithms_type == AlgorithmsType.CUSTOM:   # 动态添加的自定义模型
        custom_model = AlgorithmsDataAccess.query_custom_model_info(algorithms_name)
        transform_type = DataTransformType(custom_model.data_transform)  # 自定义模型的数据变换类型
        need_geometry = custom_model.X_with_geometry == True  # 自定义模型的输入X中是否需要geometry
    else:  # 固有的单一模型建模
        # 确定模型建模时数据是否包含geometry列
        need_geometry = (algorithms_type == AlgorithmsType.MGWR or
                         algorithms_type == AlgorithmsType.CK or
                         algorithms_type == AlgorithmsType.RK)
        # 确定数据变换方式:不做变换/进行Z-score标准化/进行normalize
        if algorithms_type == AlgorithmsType.MLP or algorithms_type == AlgorithmsType.PLSR:  # MLP,PLSR采用min-max归一化
            transform_type = DataTransformType.Normalize
        elif (algorithms_type == AlgorithmsType.EN or algorithms_type == AlgorithmsType.CK
              or algorithms_type == AlgorithmsType.GLM or algorithms_type == AlgorithmsType.MGWR
              or algorithms_type == AlgorithmsType.RK or algorithms_type == AlgorithmsType.SVR
              or algorithms_type == AlgorithmsType.KNR):  # EN、CK、GLM,MGWR,RK,SVR，KNR,采用z-score归一化
            transform_type = DataTransformType.ZScore
        else:  # RF，XGBR,XGBRFR,无需进行任何变换(决策树、随机森林、朴素贝叶斯、XGBoost、LightGBM等)
            transform_type = DataTransformType.Nochange

    # 从tabular文件加载数据，并切分为训练集和测试集
    if use_data_argument:
        train_X, train_y, test_X, test_y, transform_info = prepare_dataset_after_argument(sample_file,
                                                                                          prediction_variable,
                                                                                          covariates_path,
                                                                                          categorical_vars,
                                                                                          left_cols1,
                                                                                          left_cols2,
                                                                                          left_cols3,
                                                                                          random_state,
                                                                                          test_dataset_size,
                                                                                          transform_type=transform_type,
                                                                                          is_normality=(
                                                                                                  data_distribution_type == DataDistributionType.Normal),
                                                                                          need_geometry=need_geometry)
    else:
        train_X, train_y, test_X, test_y, transform_info = prepare_dataset(sample_file,
                                                                           prediction_variable,
                                                                           categorical_vars,
                                                                           left_cols1,
                                                                           left_cols2,
                                                                           left_cols3,
                                                                           random_state,
                                                                           test_dataset_size,
                                                                           transform_type=transform_type,
                                                                           is_normality=(
                                                                                   data_distribution_type == DataDistributionType.Normal),
                                                                           need_geometry=need_geometry
                                                                           )
    # 针对受多重共线性影响大的算法（目前包括MGWR，RK,GLM，SVR），去除共线性列
    # 使用特征重要性筛选后的列建模,防止过拟合
    min_cols = 5  # 最少保留5个列
    if algorithms_config.DO_COVARS_ELLIMATION:  # 如果需要进行列筛选
        if algorithms_type == AlgorithmsType.STACKING:  # 多重共线性敏感算法，使用left_cols1
            if (AlgorithmsType.MGWR.value in algorithms_name or AlgorithmsType.RK.value in algorithms_name
                    or AlgorithmsType.GLM.value in algorithms_name or AlgorithmsType.CK.value in algorithms_name):  # MGWR，GLM，RK，CK
                if len(left_cols3) < min_cols:
                    left_cols = left_cols1  # 经过双重剔除后剩余的列过少时，仅采用经过多重共线性剔除后的列
                else:
                    left_cols = left_cols3  # 采用双重剔除后的列
            else:
                if len(left_cols2) < min_cols:
                    left_cols = None  # 经过特征重要性分析后剩余的列过少时，采用全部列
                else:
                    left_cols = left_cols2  # 采用经过特征重要性分析后剩余的列
            if left_cols is not None:  # 如果需要进行列剔除
                all_cols = train_X.columns
                for col_name in all_cols:
                    if col_name not in left_cols and col_name != algorithms_config.DF_GEOM_COL:
                        train_X.pop(col_name)
                if test_X is not None:
                    all_cols = test_X.columns
                    for col_name in all_cols:
                        if col_name not in left_cols and col_name != algorithms_config.DF_GEOM_COL:
                            test_X.pop(col_name)
        else:
            if (algorithms_type == AlgorithmsType.MGWR or algorithms_type == AlgorithmsType.RK
                    or algorithms_type == AlgorithmsType.GLM or algorithms_type == AlgorithmsType.CK or
                (algorithms_type == AlgorithmsType.CUSTOM and custom_model.can_deal_multicollinearity == False)):
                if len(left_cols3) < min_cols:  # 经过双重剔除后剩余的列过少时，仅采用经过多重共线性剔除后的列
                    left_cols = left_cols2 if len(left_cols1) == 0 else left_cols1  # 考虑到多重共线性剔除时left_cols1为空
                else:
                    left_cols = left_cols3  # 采用双重剔除后的列
            else:
                if len(left_cols2) < min_cols:
                    left_cols = None  # 经过特征重要性分析后剩余的列过少时，采用全部列
                else:
                    left_cols = left_cols2  # 采用经过特征重要性分析后剩余的列
            if left_cols is not None:  # 如果需要进行列剔除
                all_cols = train_X.columns
                for col_name in all_cols:
                    if col_name not in left_cols and col_name != algorithms_config.DF_GEOM_COL:
                        train_X.pop(col_name)
                if test_X is not None:
                    all_cols = test_X.columns
                    for col_name in all_cols:
                        if col_name not in left_cols and col_name != algorithms_config.DF_GEOM_COL:
                            test_X.pop(col_name)
    position = os.path.basename(sample_file).rfind('_')
    ds_name = os.path.basename(sample_file)[:position]
    algorithms_config.LIMESODA_FOLDS_FILE_PATH = r'D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data'+'\\'+ds_name
    algorithms_config.LIMESODA_FOLDS_FILE_PATH += '\\'
    algorithms_config.LIMESODA_FOLDS_FILE_PATH += ds_name
    algorithms_config.LIMESODA_FOLDS_FILE_PATH += '_folds.csv'
    return train_X, train_y, test_X, test_y, transform_info


'''
将数据分割为训练集和测试集
structure_data_file：结构化数据文件的路径
prop_name：待建模的土壤属性名称
categorical_data：类别型变量数据
left_cols:具有特征重要性的列
random_state：数据采样的随机状态
test_dataset_size：测试集的分割比例,范围为(0-1),0表示不分割
transform_type：对建模的解释变量数据进行变换的方式
is_normality：响应变量是否符合正态分布（异常值剔除时需要），是则采用3σ准则检测，否则采用四分位距法
need_geometry:是否需要生成Geometry列

注：在方法内部，对数据进行了如下处理：
1、剔除异常值
2、仅留下具有特征重要性的列
3、shuffle
4、按需进行了增加geometry列（同时删除原有csv中的coordinates_x和coordinates_y）
5、类别数据的类别化处理（使用pandas.Categorical）
6、不变换/标准化/归一化（三选一）
7、按字符序对列进行了排序，以确保建模和预测时传入的列能够保持一致

应在此方法的外部执行其它必要的处理：
1、对类别型的列进行平均编码（部分算法不需要，如xgboost）
'''


def prepare_dataset(structure_data_file: str, prop_name: str, categorical_data: dict[str, str], left_cols1: list[str],
                    left_cols2: list[str], left_cols3: list[str], random_state: int, test_dataset_size: float,
                    transform_type: DataTransformType, is_normality, need_geometry=False):
    use_split = test_dataset_size > 0
    # 导入数据
    df = pd.read_csv(structure_data_file)
    # 仅保留经过特征重要性筛选后的列
    if algorithms_config.DO_COVARS_ELLIMATION:
        all_cols = df.columns
        for col in all_cols:
            if (col != prop_name and col != algorithms_config.CSV_GEOM_COL_X and col != algorithms_config.CSV_GEOM_COL_Y
                    and (col not in left_cols1 and col not in left_cols2)):
                df.pop(col)
    # 注意，必须要shuffle，否则效果很差，因为.csv中的数据可能是原始排序的,具有区域性
    train_dataset = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # train_dataset = df
    # 在数据需要进行Normalize时，剔除异常值,因为min-max归一化对异常值敏感
    if transform_type == DataTransformType.Normalize and algorithms_config.DO_COVARS_ELLIMATION: # 在和limesoda数据集比较时，不剔除异常值
        outliers, lower_bound, upper_bound = DataExceptionTest.exception_test(list(train_dataset[prop_name]),
                                                                              is_normality)
        if len(outliers) > 0:  # 使用过滤掉异常值后的数据进行分析和检测
            train_dataset = train_dataset[
                (train_dataset[prop_name] >= lower_bound) & (train_dataset[prop_name] <= upper_bound)]

    # 由坐标值列生成geometry列
    if need_geometry and algorithms_config.CSV_GEOM_COL_X in train_dataset.columns:
        geometry = []
        for index, row in train_dataset.iterrows():
            geometry.append(
                Point(float(row[algorithms_config.CSV_GEOM_COL_X]), float(row[algorithms_config.CSV_GEOM_COL_Y])))
        gdf = gpd.GeoDataFrame(train_dataset[train_dataset.columns].copy(),
                               geometry=geometry)  # 创建GeoDataFrame-增加了geometry列
    else:
        gdf = train_dataset.copy()

    # 删除掉数据中的X和Y坐标值列
    if algorithms_config.CSV_GEOM_COL_X in gdf.columns:
        gdf.drop(algorithms_config.CSV_GEOM_COL_X, axis=1, inplace=True)
        gdf.drop(algorithms_config.CSV_GEOM_COL_Y, axis=1, inplace=True)

    gdf, transform_info = dataframe_transform(categorical_data, gdf, prop_name, transform_type)
    # 强制将64位数值转换为32位，以降低计算内存消耗
    # float_cols = gdf.select_dtypes(include=['float64']).columns
    # gdf[float_cols] = gdf[float_cols].astype('float32')
    # int_cols = gdf.select_dtypes(include=['int64']).columns
    # gdf[int_cols] = gdf[int_cols].astype('int32')
    if use_split:  # 分割为训练和测试
        # 分位数分割
        bins = pd.qcut(gdf[prop_name], q=algorithms_config.KFOLD, labels=False)
        train_X, test_X = train_test_split(gdf, test_size=test_dataset_size,
                                           stratify=bins)
        # 按比例随机分割
        # train_X, test_X = train_test_split(gdf, test_size=algorithms_config.TEST_DATASET_SIZE,
        #                                    random_state=random_state)  # 按照8：2比例固定的随机切分
    else:  # 不做分割
        train_X = gdf
        test_X = None

    # 获取训练集和测试集的目标变量Y
    train_y = train_X.pop(prop_name)
    if use_split:
        test_y = test_X.pop(prop_name)
    else:
        test_y = None

    # 对数据集中的环境协变量列进行排序，以确保报错的模型中是按照特定顺序存储的，在预测时，也会按同样规则排序，才能做出正确预测
    train_X_sorted = train_X.sort_index(axis=1)
    if use_split:
        test_X_sorted = test_X.sort_index(axis=1)
    else:
        test_X_sorted = None
    return train_X_sorted, train_y, test_X_sorted, test_y, transform_info


'''
进行数据增强后再分割为训练集和测试集

注：在方法内部，对数据进行了如下处理：
1、剔除异常值
2、仅留下具有特征重要性的列
3、shuffle
4、按需进行了增加geometry列（同时删除原有csv中的coordinates_x和coordinates_y）
5、类别数据的类别化处理（使用pandas.Categorical）
6、不变换/标准化/归一化（三选一）
7、按字符序对列进行了排序，以确保建模和预测时传入的列能够保持一致

应在此方法的外部执行其它必要的处理：
1、对类别型的列进行平均编码（部分算法不需要，如xgboost）
'''


def prepare_dataset_after_argument(structure_data_file: str, prop_name: str, covariates_path: str,
                                   categorical_data: dict[str, str], left_cols1: list[str], left_cols2: list[str],
                                   left_cols3: list[str], random_state: int, test_dataset_size: float,
                                   transform_type: DataTransformType, is_normality, need_geometry=False):
    use_split = test_dataset_size > 0
    # 导入数据
    df = pd.read_csv(structure_data_file)  # csv中包含了坐标点的列，列名称固定为：coordinate_x,coordinate_y
    # 仅保留经过特征重要性筛选后的列
    # if algorithms_config.DO_COVARS_ELLIMATION:
    #     all_cols = df.columns
    #     for col in all_cols:
    #         if (col != prop_name and col != algorithms_config.CSV_GEOM_COL_X and col != algorithms_config.CSV_GEOM_COL_Y and
    #                 (col not in left_cols1 or col not in left_cols2)):
    #             df.pop(col)
    # 在数据需要进行Normalize时，剔除异常值,因为min-max归一化对异常值敏感
    if transform_type == DataTransformType.Normalize:
        outliers, lower_bound, upper_bound = DataExceptionTest.exception_test(list(df[prop_name]), is_normality)
        if len(outliers) > 0:  # 使用过滤掉异常值后的数据进行分析和检测
            df = df[(df[prop_name] >= lower_bound) & (df[prop_name] <= upper_bound)]
    # 创建GeoDataFrame,只保留两列：预测的属性列和坐标值列
    points_gdf = create_geo_dataframe(df, prop_name)

    # 进行数据增强-增加虚拟样点，points_gdf列不变，行增加
    augmenter = SpatialDataAugmentation(points_gdf, prop_name)
    points_gdf = augmenter.augment_data(augmentation_ratio=algorithms_config.AUGUMENTATION_RATION,
                                        max_distance=100)

    # 从环境变量TIF中提取出新增点的环境变量值
    # 获取栅格文件列表
    raster_folder = os.path.join(
        algorithms_config.DOCKER_DATA_PATH if algorithms_config.DOCKER_MODE else algorithms_config.LOCAL_DATA_PATH,
        covariates_path)
    raster_files = []
    df.pop(prop_name)
    for column_name in df.columns:
        if column_name == algorithms_config.CSV_GEOM_COL_X or column_name == algorithms_config.CSV_GEOM_COL_Y:
            continue
        raster_files.append(os.path.join(raster_folder, column_name + ".tif"))
    print(f"确定 {len(raster_files)} 个栅格文件")

    # 提取栅格值
    raster_values = {}
    # 并行处理栅格文件
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     results = list(tqdm(
    #         executor.map(augmenter.extract_from_raster, raster_files),
    #         total=len(raster_files),
    #         desc="提取栅格值"
    #     ))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(augmenter.extract_from_raster, raster_files))
    # 整理结果
    for column_name, values in results:
        raster_values[column_name] = values

    # 创建结果DataFrame
    raster_df = pd.DataFrame(raster_values)

    # 合并数据
    if not need_geometry:  # 不需要保持geometry列，例如机器学习方法，而mgwr之类的方法通常需要
        del points_gdf[algorithms_config.DF_GEOM_COL]
    points_gdf = points_gdf.reset_index(drop=True)
    raster_df = raster_df.reset_index(drop=True)
    result_df = pd.concat([raster_df, points_gdf], axis=1)  # 自变量列与预测列(+Geometry列)合并

    # 打散顺序，不再和原始的.csv中的顺序保持一致，避免原始.csv中记录顺序可能存在相邻记录聚集的问题
    augument_dataset = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 如果训练集涉及无序类别变量，将类别变量的类型从数值转换为类别,并且连续类型列做Z-score处理
    augument_dataset, transform_info = dataframe_transform(categorical_data, augument_dataset,
                                                           prop_name,
                                                           transform_type)

    # 强制将64位数值转换为32位，以降低计算内存消耗
    float_cols = augument_dataset.select_dtypes(include=['float64']).columns
    augument_dataset[float_cols] = augument_dataset[float_cols].astype('float32')
    int_cols = augument_dataset.select_dtypes(include=['int64']).columns
    augument_dataset[int_cols] = augument_dataset[int_cols].astype('int32')

    if use_split:  # 将增强后的数据分割为训练和测试
        # 分位数分割
        bins = pd.qcut(augument_dataset[prop_name], q=algorithms_config.KFOLD, labels=False)
        train_X, test_X = train_test_split(augument_dataset, test_size=algorithms_config.TEST_DATASET_SIZE,
                                           stratify=bins)
        # 按比例随机分割
        # train_X, test_X = train_test_split(augument_dataset, test_size=algorithms_config.TEST_DATASET_SIZE,
        #                                    random_state=random_state)
    else:  # 不做分割
        train_X = augument_dataset
        test_X = None

    # 获取训练集的目标变量Y
    train_y = train_X.pop(prop_name)
    if use_split:
        test_y = test_X.pop(prop_name)
    else:
        test_y = None

    # 对数据集中的环境协变量列进行排序，以确保报错的模型中是按照特定顺序存储的，在预测时，也会按同样规则排序，才能做出正确预测
    train_X_sorted = train_X.sort_index(axis=1)
    if use_split:
        test_X_sorted = test_X.sort_index(axis=1)
    else:
        test_X_sorted = None
    return train_X_sorted, train_y, test_X_sorted, test_y, transform_info


'''
将.csv中的所有数据作为数据集
当前目录下必须存在category.json文件，该文件中存储着所有的类别变量的取值
split_scale:训练集和测试集的切割比例

先分为为训练集和测试集，然后仅针对训练集进行数据增强(仅针对训练集进行增强，如果涉及分割，则分割的测试集不增强)
'''


def prepare_dataset_only_augment_train(prop_name: str, structure_data_file: str, covariates_path: str,
                                       random_state: int):
    use_split = algorithms_config.TEST_DATASET_SIZE > 0  # 根据测试集的比例来确定是否需要分割出测试集
    # 导入数据
    df = pd.read_csv(structure_data_file)  # csv中包含了坐标点的列：coordinates
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')

    if use_split:  # 将数据分割为训练和测试
        df_train, df_test = train_test_split(df, test_size=algorithms_config.TEST_DATASET_SIZE,
                                             random_state=random_state)  # 按照8：2比例固定的随机切分
    else:  # 不做分割
        df_train = df
        df_test = None

    coordinates = list(df_train.pop('coordinates'))  # 从csv中取出样点坐标
    if use_split:
        df_test.pop('coordinates')

    geometry = []
    for xy in coordinates:
        coord = xy.split(' ')
        geometry.append(Point(float(coord[0][1:]), float(coord[1][:-1])))
    points_gdf = gpd.GeoDataFrame(df_train[[prop_name]].copy(), geometry=geometry)  # 创建GeoDataFrame,只保留两列：预测的属性列和坐标值列
    augmenter = SpatialDataAugmentation(points_gdf, prop_name)  # 进行数据增强
    points_gdf = augmenter.augment_data(augmentation_ratio=algorithms_config.AUGUMENTATION_RATION,
                                        max_distance=100)  # 数据增强的结果仅仅是生成了预测的因变量值
    # 从环境变量TIF中提取出新增点的环境变量值
    df_train.pop(prop_name)
    if use_split:
        test_labels = df_test.pop(prop_name)
    else:
        test_labels = None
    # 获取栅格文件列表
    raster_folder = os.path.join(
        algorithms_config.DOCKER_DATA_PATH if algorithms_config.DOCKER_MODE else algorithms_config.LOCAL_DATA_PATH,
        covariates_path)

    raster_files = []
    for column_name in df_train.columns:
        raster_files.append(os.path.join(raster_folder, column_name + ".tif"))
    print(f"确定 {len(raster_files)} 个栅格文件")

    # 提取栅格值
    raster_values = {}
    # 并行处理栅格文件
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     results = list(tqdm(
    #         executor.map(augmenter.extract_from_raster, raster_files),
    #         total=len(raster_files),
    #         desc="提取栅格值"
    #     ))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(augmenter.extract_from_raster, raster_files))
    # 整理结果
    for column_name, values in results:
        raster_values[column_name] = values

    # 创建结果DataFrame
    raster_df = pd.DataFrame(raster_values)

    # 合并数据
    points_gdf_no_geom = points_gdf.drop(algorithms_config.DF_GEOM_COL, axis=1).reset_index(drop=True)
    raster_df = raster_df.reset_index(drop=True)
    result_df = pd.concat([points_gdf_no_geom, raster_df], axis=1)

    train_dataset = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 获取训练集的目标变量Y
    train_labels = train_dataset.pop(prop_name)

    # 如果训练集涉及无序类别变量，将类别变量的类型从数值转换为类别
    category_json_file = os.path.join(os.getcwd(), "category.json")
    cate_columns = []
    with open(category_json_file, 'r', encoding='utf-8') as file:
        category_data = json.load(file)
        for key, value in category_data.items():
            if key in train_dataset.columns:
                train_dataset[key] = pd.Categorical(train_dataset[key], categories=list(map(int, value.split(','))))
                if use_split:
                    df_test[key] = pd.Categorical(df_test[key], categories=list(map(int, value.split(','))))
                cate_columns.append(key)
    # 对连续型的列进行z-score标准化处理
    for key in train_dataset.columns:
        if key not in cate_columns:  # 仅针对连续型环境变量处理
            col_mean = df[key].mean()
            col_std = df[key].std()
            train_dataset[key] = (train_dataset[key] - col_mean) / col_std
            if use_split:
                df_test[key] = (df_test[key] - col_mean) / col_std
    return train_dataset, train_labels, df_test, test_labels


'''
对数据集中的定类变量进行平均编码
category_vars:类别变量字典
'''


def mean_encode_dataset(categorical_vars: dict, X: pd.DataFrame, y: pd.Series):
    if categorical_vars is None:  # 如果无类别型变量
        return X, None
    col_names = []
    for key, value in categorical_vars.items():  # 遍历每一个定类变量,字典中的变量名称必须和dataframe中的列名称保持一致
        if key in X.columns:
            col_names.append(key)
    mean_encoder = MeanEncoder(col_names, target_type='regression')
    if len(col_names) > 0:  # 如果有定类变量需要处理
        X = mean_encoder.fit_transform(X, y)
    return X, mean_encoder
