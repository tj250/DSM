import os
import os.path
import algorithms_config
from data.data_dealer import prepare_model_train_dataset
from DSMAlgorithms.base.base_data_structure import AlgorithmsType, algorithms_dict, DataDistributionType
from DSMAlgorithms.mgwr import MGWRWrap
from DSMAlgorithms.mlp import MLPWrap
from DSMAlgorithms.plsr import PLSRWrap
from DSMAlgorithms.glm import GLMWrap
from DSMAlgorithms.knr import KNRWrap
from DSMAlgorithms.elastic_net import ElasticNetWrap
from DSMAlgorithms.rk import RegressionKrigeWrap
from DSMAlgorithms.rfr import RandomForestRegressionWrapper
from DSMAlgorithms.ck import CoKrigeWrap
from DSMAlgorithms.xgbr import XGBRWrap
from DSMAlgorithms.svr import SVRWrap
from DSMAlgorithms.stacking import StackingWrap
from DSMAlgorithms.custom_model import CustomModelWrap
from db_access.task_dealer import TaskDataAccess
from db_access.algorithms_parameters import AlgorithmsDataAccess
from utility import load_module_from_file,get_sklearn_style_class

'''
执行单一算法的建模
algorithm_id:算法ID
algorithms_type:算法类型
prediction_variable:预测变量名称
file_name:样点数据文件
covariates_dir_name:协变量所在目录
categorical_vars:类别型变量信息列表
left_cols1:多重共线性剔除后剩余保留的列，在ESDA阶段确定
left_cols2:特征重要性分析后剩余保留的列，在ESDA阶段确定
left_cols3:经过多重共线性剔除和特征重要性分析后剩余保留的列，在ESDA阶段确定
data_distribution_type:可选，由于当使用GLM时，需要明确的数据分布类型
'''


def execute_algorithm(algorithm_id: str, algorithms_name: str, prediction_variable: str, file_name: str,
                      covariates_dir_name: str, categorical_vars: dict[str, str], left_cols1:list[str], left_cols2:list[str],
                      left_cols3:list[str], data_distribution_type: DataDistributionType = DataDistributionType.Unknown) -> bool:
    sample_file = os.path.join(algorithms_config.DOCKER_DATA_PATH if algorithms_config.DOCKER_MODE else algorithms_config.LOCAL_DATA_PATH, file_name)
    covariates_path = os.path.join(algorithms_config.DOCKER_DATA_PATH if algorithms_config.DOCKER_MODE else algorithms_config.LOCAL_DATA_PATH,
                                   covariates_dir_name)
    algorithms_type_names = [member.value for member in AlgorithmsType]
    if algorithms_name in algorithms_type_names:
        algorithms_type = AlgorithmsType(algorithms_name)
    else:
        algorithms_type = AlgorithmsType.CUSTOM
    train_X, train_y, test_X, test_y, transform_info = prepare_model_train_dataset(algorithms_config.USE_DATA_AUGMENTATION,
                                                                                   sample_file,
                                                                                   covariates_path,
                                                                                   algorithms_config.RANDOM_STATE,
                                                                                   algorithms_config.TEST_DATASET_SIZE,
                                                                                   algorithms_type,
                                                                                   algorithms_name,
                                                                                   prediction_variable,
                                                                                   categorical_vars,
                                                                                   left_cols1,
                                                                                   left_cols2,
                                                                                   left_cols3,
                                                                                   data_distribution_type)
    if algorithms_type == AlgorithmsType.EN:
        algorithms_wrap = ElasticNetWrap(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.GLM:
        algorithms_wrap = GLMWrap(prediction_variable, categorical_vars, data_distribution_type)
    elif algorithms_type == AlgorithmsType.PLSR:
        algorithms_wrap = PLSRWrap(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.MLP:
        algorithms_wrap = MLPWrap(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.KNR:
        algorithms_wrap = KNRWrap(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.SVR:
        algorithms_wrap = SVRWrap(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.RFR:
        algorithms_wrap = RandomForestRegressionWrapper(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.XGBR:
        algorithms_wrap = XGBRWrap(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.CK:
        algorithms_wrap = CoKrigeWrap(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.RK:
        algorithms_wrap = RegressionKrigeWrap(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.MGWR:
        algorithms_wrap = MGWRWrap(prediction_variable, categorical_vars)
    elif algorithms_type == AlgorithmsType.CUSTOM:
        custom_model_info = AlgorithmsDataAccess.query_custom_model_info(algorithms_name)
        py_file = os.path.join(os.getcwd(), algorithms_name + ".py")
        AlgorithmsDataAccess.save_as_pyfile(custom_model_info.class_code, py_file, True)  # 将代码保持为文件
        loaded_module = load_module_from_file(py_file) # 从.py文件装载python模块
        model_class = get_sklearn_style_class(loaded_module)  # 从模块中解析出第一个包含fit和method方法的类名称
        algorithms_pbounds = AlgorithmsDataAccess.retrival_custom_model_pbounds(algorithms_name)  # 获取自定义模型的关键字参数的边界
        algorithms_nested_args_types = AlgorithmsDataAccess.retrival_custom_model_nested_args_types(algorithms_name)
        # 将算类名称,类对象构造器的位置参数和关键字参数传递给包装类
        algorithms_wrap = CustomModelWrap(model_class, algorithms_name, custom_model_info.special_args,
                                          custom_model_info.dyn_eval_args, custom_model_info.nested_args_name,
                                          algorithms_nested_args_types, custom_model_info.special_enum_conversion_args,
                                          custom_model_info.enum_conversion_args, custom_model_info.complex_lamda_args,
                                          algorithms_pbounds, prediction_variable, categorical_vars,
                                          custom_model_info.X_with_geometry)
    else:
        print(f"发现未能支持的算法：{algorithms_dict[algorithms_type]}")
        return False
    if algorithms_config.USE_LIMESODA_KFOLD:  # 在计算limesoda指标时，设置相应的样点文件名
        algorithms_wrap.set_limesoda_sample_file(sample_file)
    successful = algorithms_wrap.build(algorithm_id, train_X, train_y, test_X, test_y, transform_info)
    if not successful:  # 未成功完成建模时，设置建模失败
        TaskDataAccess.set_build_model_failed(algorithm_id)

    if algorithms_type == AlgorithmsType.CUSTOM:
        print(f"算法执行结束，模型ID：{algorithm_id}, 自定义的算法名称：{algorithms_name}")
    else:
        print(f"算法执行结束，模型ID：{algorithm_id}, 算法名称：{algorithms_dict[algorithms_type]}")
    return successful


'''
执行多算法的堆叠
stacking_algorithms_id:堆叠ID
stacking_algorithms:堆叠的算法类型列表
prediction_variable:预测变量名称
file_name:样点数据文件
covariates_dir_name:协变量所在目录
categorical_vars:类别型变量信息列表
left_cols1:多重共线性剔除后剩余保留的列，在ESDA阶段确定
left_cols2:特征重要性分析后剩余保留的列，在ESDA阶段确定
left_cols3:经过多重共线性剔除和特征重要性分析后剩余保留的列，在ESDA阶段确定
data_distribution_type:当使用GLM时，需要明确的数据分布类型
'''


def stacking_models(stacking_algorithms_id: str, stacking_algorithms: list[str], prediction_variable: str,
                    file_name: str, covariates_dir_name: str, categorical_vars: dict[str, str], left_cols1:list[str],
                    left_cols2:list[str], left_cols3:list[str], data_distribution_type: DataDistributionType) -> bool:
    sample_file = os.path.join(algorithms_config.DOCKER_DATA_PATH if algorithms_config.DOCKER_MODE else algorithms_config.LOCAL_DATA_PATH, file_name)
    covariates_path = os.path.join(algorithms_config.DOCKER_DATA_PATH if algorithms_config.DOCKER_MODE else algorithms_config.LOCAL_DATA_PATH,
                                   covariates_dir_name)
    # if algorithms_name in AlgorithmsType.__members__:
    #     algorithms_type = AlgorithmsType(algorithms_name)
    # else:
    #     algorithms_type = AlgorithmsType.CUSTOM
    # 准备堆叠模型训练所需的数据集
    train_X, train_y, test_X, test_y, transform_info = prepare_model_train_dataset(algorithms_config.USE_DATA_AUGMENTATION,
                                                                                   sample_file,
                                                                                   covariates_path,
                                                                                   algorithms_config.RANDOM_STATE,
                                                                                   algorithms_config.TEST_DATASET_SIZE,
                                                                                   AlgorithmsType.STACKING,
                                                                                   stacking_algorithms,
                                                                                   prediction_variable,
                                                                                   categorical_vars,
                                                                                   left_cols1,
                                                                                   left_cols2,
                                                                                   left_cols3,
                                                                                   data_distribution_type)

    # 开始建模
    stacking = StackingWrap(prediction_variable, categorical_vars, stacking_algorithms)
    if algorithms_config.USE_LIMESODA_KFOLD:  # 在计算limesoda指标时，设置相应的样点文件名
        stacking.set_limesoda_sample_file(sample_file)
    successful = stacking.build(stacking_algorithms_id, train_X, train_y, test_X, test_y, transform_info,
                                data_distribution_type)
    print("建模的环境变量数量:"+str(len(train_X.columns)))
    if not successful:  # 未成功完成建模时，设置堆叠失败
        TaskDataAccess.set_stacking_failed(stacking_algorithms_id)

    print(f"模型堆叠执行结束，堆叠的算法ID：{stacking_algorithms_id}")
    return successful
