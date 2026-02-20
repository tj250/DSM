import json
import pickle
import os
from DSMAlgorithms.base.dsm_base_model import AlgorithmsType
from db_access.postgres_db import PostgresAccess

'''
扩充的模型数据结构
'''


class CustomModelData():
    custom_model_id: str = ''  # 扩充模型的ID
    custom_model_name: str = ''  # 扩充模型的名称
    description: str = ''  # 模型的描述
    class_code: str = ''  # 符合sklearn风格的算法类的代码片段
    X_with_geometry: bool = None  # 输入变量中是否需要包含空间坐标信息
    data_transform: int = None  # 数据变换类型，0：不变换,1:z-score标准化,2:normalize
    can_deal_high_dims: bool = None  # 是否可以处理高维数据
    can_deal_small_samples: bool = None  # 是否可以处理小样本数据
    can_deal_unknown_distribution: bool = None  # 是否可处理未知分布的数据
    can_deal_multicollinearity: bool = None  # 是否可以处理多重共线性数据
    can_deal_heterogeneity: bool = None  # 是否可以针对空间异质性建模
    special_args: dict = None  # 构造方法的位置参数列表
    special_enum_conversion_args:dict = None  # 特殊参数中的的枚举转换信息，可以有多个,例如：{"algorithm":{"1":"brute","2":"kd_tree"}}
    dyn_eval_args: str = None  # 构造方法的位置参数中需要进行动态评估的参数名称的列表，用,分隔
    enum_conversion_args:dict = None  # 可调参数中的枚举转换信息，可以有多个,例如：{"variogram_model":{"1":"gaussian","2":"spherical","3":"exponential"}}
    complex_lamda_args:dict = None   # 可调参数中需要通过lamda表达式处理的内容，可以有多个，例如：{"anisotropy_scaling":"lamda x:(x,x)","anisotropy_angle":"lamda x:(x,x,x)"}
    nested_args_name:str = None     # 在动态eval参数中可嵌套的参数的名称
    can_stacking:bool = None        # 是否可用于进行模型堆叠

class AlgorithmsDataAccess:
    '''
    从数据库中获取参数边界(用于BayesianOptimization得pbounds参数)
    algorithms_name:参数名称
    '''

    @staticmethod
    def retrival_pbounds(algorithms_types: list[AlgorithmsType]) -> dict[str, tuple]:
        pbounds = {}
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        for algorithm_type in algorithms_types:
            cursor.execute(
                "select param_name,min_value,max_value from params_search_range where algorithms_name=%s order by params_search_range_id",
                (algorithm_type.name.lower(),))
            rows = cursor.fetchall()
            for row in rows:
                pbounds[row[0]] = (row[1], row[2])
        cursor.close()
        conn.close()
        return pbounds

    '''
    从数据库中获取参数边界(用于BayesianOptimization得pbounds参数)
    algorithms_name:算法名称
    '''

    @staticmethod
    def retrival_custom_model_pbounds(algorithms_name: str) -> dict[str, tuple]:
        pbounds = {}
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select param_name,min_value,max_value from params_search_range where algorithms_name=%s order by params_search_range_id",
            (algorithms_name,))
        rows = cursor.fetchall()
        for row in rows:
            pbounds[row[0]] = (row[1], row[2])
        cursor.close()
        conn.close()
        return pbounds

    @staticmethod
    def retrival_all_params_pbounds() -> dict[str, tuple]:
        pbounds = {}
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select algorithms_name,param_name,min_value,max_value from params_search_range order by params_search_range_id")
        rows = cursor.fetchall()
        for row in rows:
            pbounds[row[0] + '_' + row[1]] = (row[2], row[3])
        cursor.close()
        conn.close()
        return pbounds

    '''
    根据自定义模型的名称获取模型信息
    '''

    @staticmethod
    def query_custom_model_info(custom_model_name: str):
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select custom_model_id,description,class_code,x_with_geometry,can_deal_high_dims,"
                       "can_deal_small_samples,can_deal_unknown_distribution,can_deal_multicollinearity,"
                       "can_deal_heterogeneity,data_transform,special_args,special_enum_conversion_args,dyn_eval_args,enum_conversion_args,"
                       "complex_lamda_args,nested_args_name,can_stacking from model_library where custom_model_name=%s",
                       (custom_model_name,))
        row = cursor.fetchall()[0]
        model_info = CustomModelData()
        model_info.custom_model_id = row[0]
        model_info.custom_model_name = custom_model_name
        model_info.description = row[1]
        model_info.class_code = pickle.loads(row[2])
        model_info.X_with_geometry = row[3]
        model_info.can_deal_high_dims = row[4]
        model_info.can_deal_small_samples = row[5]
        model_info.can_deal_unknown_distribution = row[6]
        model_info.can_deal_multicollinearity = row[7]
        model_info.can_deal_heterogeneity = row[8]
        model_info.data_transform = row[9]
        model_info.special_args = json.loads(row[10])
        model_info.special_enum_conversion_args = None if row[11] is None else json.loads(row[11])
        model_info.dyn_eval_args = row[12]
        model_info.enum_conversion_args = None if row[13] is None else json.loads(row[13])
        model_info.complex_lamda_args = None if row[14] is None else json.loads(row[14])
        model_info.nested_args_name = row[15]
        model_info.can_stacking = row[16]
        cursor.close()
        conn.close()
        return model_info

    '''
    获取自定义模型的python代码
    '''

    @staticmethod
    def retrival_custom_model_code_file(custom_model_name: str):
        py_file = os.path.join(os.getcwd(), custom_model_name + ".py")
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select class_code from model_library where custom_model_name=%s", (custom_model_name,))
        rows = cursor.fetchall()
        AlgorithmsDataAccess.save_as_pyfile(pickle.loads(rows[0][0]), py_file, True)
        cursor.close()
        conn.close()
        return py_file

    '''
    获取自定义模型的可嵌套参数的类型
    '''

    @staticmethod
    def retrival_custom_model_nested_args_types(custom_model_name: str):
        param_dict = {}
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select param_name,param_type from params_search_range where algorithms_name=%s",(custom_model_name,))
        rows = cursor.fetchall()
        for row in rows:
            if row[0].startswith("nested_"):
                param_dict[row[0]] = row[1]
        cursor.close()
        conn.close()
        return param_dict


    '''
    将一段代码保存为一个文件
    '''

    @staticmethod
    def save_as_pyfile(code_str, filename, overwrite=False):
        """
        将代码字符串保存为Python文件
        :param code_str: 要保存的代码字符串
        :param filename: 目标文件名（带.py后缀）
        :param overwrite: 是否覆盖已存在文件
        :return: 文件保存路径
        """
        if not filename.endswith('.py'):
            filename += '.py'

        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"文件 {filename} 已存在")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code_str)

        return os.path.abspath(filename)
