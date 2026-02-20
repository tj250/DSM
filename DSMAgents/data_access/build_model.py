import datetime, psycopg2, json, os
import pickle, dill
from dataclasses import dataclass
from agents.utils.views import BMActiveState
from data_access.db import PostgresAccess
from data_access.task import TaskInfo
import time
import uuid
from agents.data_structure.base_data_structure import RegressionModelInfo
from DSMAlgorithms import AlgorithmsType, algorithms_dict
from DSMAlgorithms import DataDistributionType
from DSMAlgorithms import CustomModelData

'''
建模信息数据结构
'''


@dataclass
class BuildModelInfo():
    task_id: str = ''  # 任务ID
    thread_id: str = ''  # 线程ID
    current_state: BMActiveState = BMActiveState.Beginning  # 建模过程的当前状态
    parameters: dict[str,] = None  # 建模结果
    start_dt: datetime.datetime = None  # 建模开始时间
    end_dt: datetime.datetime = None  # 建模结束时间


'''
任务数据模型：访问sqlite数据库，执行与任务相关的各类操作
'''


class BuildModelDataAccess:
    '''
    获取扩充算法信息
    '''

    @staticmethod
    def get_extend_models() -> list[CustomModelData]:
        models = []
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select custom_model_id,custom_model_name,description,class_code,x_with_geometry,can_deal_high_dims,"
            "can_deal_small_samples,can_deal_unknown_distribution,can_deal_multicollinearity,can_deal_heterogeneity,"
            "data_transform,special_args,special_enum_conversion_args,dyn_eval_args,enum_conversion_args,"
            "complex_lamda_args,can_stacking from model_library order by custom_model_name")
        rows = cursor.fetchall()
        for row in rows:
            model_info = CustomModelData()
            model_info.custom_model_id = row[0]
            model_info.custom_model_name = row[1]
            model_info.description = row[2]
            model_info.class_code = '' if row[3] is None else pickle.loads(row[3])
            model_info.X_with_geometry = row[4]
            model_info.can_deal_high_dims = row[5]
            model_info.can_deal_small_samples = row[6]
            model_info.can_deal_unknown_distribution = row[7]
            model_info.can_deal_multicollinearity = row[8]
            model_info.can_deal_heterogeneity = row[9]
            model_info.data_transform = row[10]
            model_info.special_args = json.loads(row[11])
            model_info.special_enum_conversion_args = None if row[12] is None else json.loads(row[12])
            model_info.dyn_eval_args = row[13]
            model_info.enum_conversion_args = None if row[14] is None else json.loads(row[14])
            model_info.complex_lamda_args = None if row[15] is None else json.loads(row[15])
            model_info.can_stacking = row[16]
            models.append(model_info)
        cursor.close()
        conn.close()
        return models

    @staticmethod
    def create_custom_model(custom_model:CustomModelData):
        successful = True
        conn = None
        try:
            conn = PostgresAccess.get_db_conn()
            cursor = conn.cursor()
            cursor.execute(
                "insert into model_library(custom_model_id, custom_model_name, description,class_code, x_with_geometry,"
                "can_deal_high_dims,can_deal_small_samples,can_deal_unknown_distribution,can_deal_multicollinearity,"
                "can_deal_heterogeneity,data_transform,special_args,special_enum_conversion_args,dyn_eval_args,"
                "enum_conversion_args,complex_lamda_args,can_stacking) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                "%s,%s,%s)",
                (custom_model.custom_model_id, custom_model.custom_model_name, custom_model.description,
                 pickle.dumps(custom_model.class_code), custom_model.X_with_geometry, custom_model.can_deal_high_dims,
                 custom_model.can_deal_small_samples, custom_model.can_deal_unknown_distribution,
                 custom_model.can_deal_multicollinearity, custom_model.can_deal_heterogeneity,
                 custom_model.data_transform, json.dumps(custom_model.special_args),
                 None if custom_model.special_enum_conversion_args is None else json.dumps(custom_model.special_enum_conversion_args),
                 custom_model.dyn_eval_args,
                 None if custom_model.enum_conversion_args is None else json.dumps(custom_model.enum_conversion_args),
                 None if custom_model.complex_lamda_args is None else json.dumps(custom_model.complex_lamda_args),
                 custom_model.can_stacking))
            conn.commit()
            cursor.close()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()  # 回滚事务
            print("向model_library插入数据时发生错误:", e)
            successful = False
        finally:
            if conn:
                conn.close()
        return successful

    @staticmethod
    def update_custom_model(custom_model:CustomModelData):
        successful = True
        conn = None
        try:
            conn = PostgresAccess.get_db_conn()
            cursor = conn.cursor()
            cursor.execute(
                "update model_library set custom_model_name=%s, description=%s,class_code=%s,x_with_geometry=%s,"
                "can_deal_high_dims=%s,can_deal_small_samples=%s,can_deal_unknown_distribution=%s,"
                "can_deal_multicollinearity=%s,can_deal_heterogeneity=%s,data_transform=%s,special_args=%s,"
                "special_enum_conversion_args=%s,dyn_eval_args=%s,enum_conversion_args=%s,complex_lamda_args=%s,"
                "can_stacking=%s where custom_model_id=%s",
                (custom_model.custom_model_name, custom_model.description,
                 pickle.dumps(custom_model.class_code), custom_model.X_with_geometry, custom_model.can_deal_high_dims,
                 custom_model.can_deal_small_samples, custom_model.can_deal_unknown_distribution,
                 custom_model.can_deal_multicollinearity, custom_model.can_deal_heterogeneity,
                 custom_model.data_transform, json.dumps(custom_model.special_args),
                 None if custom_model.special_enum_conversion_args is None else json.dumps(custom_model.special_enum_conversion_args),
                 custom_model.dyn_eval_args,
                 None if custom_model.enum_conversion_args is None else json.dumps(custom_model.enum_conversion_args),
                 None if custom_model.complex_lamda_args is None else json.dumps(custom_model.complex_lamda_args),
                 custom_model.can_stacking,custom_model.custom_model_id,))
            conn.commit()
            cursor.close()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()  # 回滚事务
            print("更新model_library中的记录时发生错误:", e)
            successful = False
        finally:
            if conn:
                conn.close()
        return successful

    '''
    查询任务相关的建模ID
    '''

    @staticmethod
    def query_build_model_id(task_id: str) -> str:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select build_model_id from build_model where task_id=%s", (task_id,))
        rows = cursor.fetchall()
        if len(rows) == 0:
            build_model_id = None
        else:
            build_model_id = rows[0][0]
        cursor.close()
        conn.close()
        return build_model_id

    '''
    查询建模的执行状态
    '''

    @staticmethod
    def query_build_model_execution_state(thread_id: str) -> BMActiveState:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select current_state from build_model where thread_id=%s", (thread_id,))
        rows = cursor.fetchall()
        active_state = BMActiveState(rows[0][0])
        cursor.close()
        conn.close()
        return active_state

    '''
    记录任务的建模批次信息（在开始远程调用分布式算法集群前，数据库中先写入记录,每一种算法会写入一条记录）
    同时，还要更新与建模相关的调整后数据文件及类别数据文件
    '''

    @staticmethod
    def record_task_batch(task_info: TaskInfo, algorithms: dict[str, str], prediction_variable: str,
                          categorical_vars_info: dict[str, str], data_distribution: DataDistributionType,
                          left_cols1: list[str], left_cols2: list[str], left_cols3: list[str]):
        successful = True
        new_build_model_id = str(uuid.uuid1())
        insert_data = []
        for algorithm, algorithm_id in algorithms.items():
            now_dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            insert_data.append((algorithm_id, new_build_model_id, algorithm, now_dt, now_dt))
        conn = None
        try:
            conn = PostgresAccess.get_db_conn()
            cursor = conn.cursor()
            # 1、向build_model表中插入一条新的建模批次信息
            cursor.execute(
                "insert into build_model(build_model_id, task_id, prediction_variable, categorical_vars_info, \
                sample_file,covariates_directory,mapping_area_file,create_dt,data_distribution_type, left_cols1, \
                left_cols2,left_cols3) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (new_build_model_id, task_info.task_id, prediction_variable,
                 None if categorical_vars_info is None else json.dumps(categorical_vars_info, ensure_ascii=False),
                 task_info.sample_file, task_info.covariates_path, task_info.mapping_area_file,
                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                 float(data_distribution.value), ','.join(left_cols1), ','.join(left_cols2),
                 ','.join(left_cols3),))  # 建模数据源
            # 2、向algorithms表插入如果建模批次相关的算法信息
            cursor.executemany('''insert into algorithms(algorithms_id, build_model_id, algorithms_name, start_dt,last_active_dt) values(%s,%s,%s,%s,%s)
                ''', insert_data)
            # 3、将task表中的建模批次更新为最新的批次
            cursor.execute("update task set build_model_call_id=%s where task_id=%s",
                           (new_build_model_id, task_info.task_id,))  # 更新建模数据信息
            conn.commit()
            cursor.close()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()  # 回滚事务
            print("插入数据时发生错误:", e)
            successful = False
        finally:
            if conn:
                conn.close()
        return successful, new_build_model_id

    '''
    记录任务的堆叠信息
    '''

    @staticmethod
    def record_task_stacking(task_info: TaskInfo, stacking_algorithms: dict[str, str]) -> bool:
        successful = True
        stacking_id = str(uuid.uuid1())
        now_dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        insert_data = []
        for stacking_algorithms_name, stacking_algorithms_id in stacking_algorithms.items():
            insert_data.append((stacking_algorithms_id, stacking_id, stacking_algorithms_name, now_dt, now_dt))
        conn = None
        try:
            conn = PostgresAccess.get_db_conn()
            cursor = conn.cursor()
            # 1、向stacking表中插入一条新的建模批次信息
            cursor.execute(
                "insert into stacking(stacking_id, build_model_id, last_active_dt,start_dt) values(%s,%s,%s,%s)",
                (stacking_id, task_info.last_build_model_id, now_dt, now_dt,))  # 建模数据源
            # 2、向algorithms表插入如果建模批次相关的算法信息
            cursor.executemany('''insert into stacking_algorithms(stacking_algorithms_id, stacking_id, algorithms_name, \
            last_active_dt, start_dt) values(%s,%s,%s,%s,%s)''', insert_data)
            conn.commit()
            cursor.close()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()  # 回滚事务
            print("插入数据时发生错误:", e)
            successful = False
        finally:
            if conn:
                conn.close()
        return successful

    '''
    获取是否存在悬挂的单一算法建模（即是否存在尚未结束的单个算法建模）
    '''

    @staticmethod
    def exists_task_batch(build_model_id: str):
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select count(*) from algorithms where build_model_id=%s", (build_model_id,))
        rows = cursor.fetchall()
        exist = rows[0][0] > 0
        cursor.close()
        conn.close()
        return exist

    '''
    获取某一批次单模型建模的进度是否已经结束
    '''

    @staticmethod
    def task_batch_is_finished(build_model_id: str):
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select end_dt from build_model where build_model_id=%s", (build_model_id,))
        rows = cursor.fetchall()
        all_finished = rows[0][0] is not None  # 如果存在为空的结束日期，则代表任务尚未完全结束
        cursor.close()
        conn.close()
        return all_finished

    '''
    获取是否存在未结束的堆叠
    '''

    @staticmethod
    def exists_stacking(build_model_id: str):
        exists = False
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        # 查询建模批次相关的堆叠ID
        cursor.execute(
            "select stacking_id from stacking where build_model_id=%s order by start_dt desc limit 1",
            (build_model_id,))
        rows = cursor.fetchall()
        if len(rows) > 0:  # 存在堆叠结果
            stacking_id = rows[0][0]
            cursor.execute("select count(*) from stacking_algorithms where stacking_id=%s",
                           (stacking_id,))
            rows2 = cursor.fetchall()
            exists = rows2[0][0] > 0  # 如果存在为空的结束日期，则代表任务尚未完全结束
        else:
            exists = False
        cursor.close()
        conn.close()
        return exists

    '''
    获取某一批次单模型建模的进度是否已经结束
    '''

    @staticmethod
    def stacking_is_finished(build_model_id: str):
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        # 查询建模批次相关的最后一次堆叠的结束时间是否为空
        cursor.execute(
            "select end_dt from stacking where build_model_id=%s order by start_dt desc limit 1",
            (build_model_id,))
        rows = cursor.fetchall()
        all_finished = rows[0][0] is not None
        cursor.close()
        conn.close()
        return all_finished

    '''
    获取某一建模批次所设计相关模型的性能指标参数
    '''

    @staticmethod
    def get_model_metrics(build_model_id: str, with_stacking=False) -> list:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select algorithms_id, algorithms_name, model_parameters from algorithms where build_model_id=%s and successful=%s",
            (build_model_id, True,))  # 取出成功的建模记录，失败的忽略
        rows = cursor.fetchall()
        all_model_metrics = []
        algorithms_type_names = [member.value for member in AlgorithmsType]
        for row in rows:
            regression_model_info = RegressionModelInfo()
            regression_model_info.algorithms_id = row[0]
            regression_model_info.algorithms_name = row[1]
            if row[1] in algorithms_type_names:  # 内置模型
                regression_model_info.algorithms_type = AlgorithmsType(row[1])
            else:
                regression_model_info.algorithms_type = AlgorithmsType.CUSTOM
            model_parameters = pickle.loads(row[2])
            regression_model_info.CV_best_score = model_parameters["best_score"]
            regression_model_info.R2 = model_parameters["R2"]
            regression_model_info.RMSE = model_parameters["RMSE"]
            regression_model_info.stacking = False
            all_model_metrics.append(regression_model_info)

        if with_stacking:  # 如果还需查询堆叠信息
            # 首先查询建模批次相关的最后一次堆叠ID（start_dt最晚时候的）
            cursor.execute(
                "select stacking_id from stacking where build_model_id=%s order by start_dt desc limit 1",
                (build_model_id,))
            rows2 = cursor.fetchall()
            if len(rows2) > 0:  # 存在堆叠结果（必须存在）
                stacking_id = rows2[0][0]
                # 进一步根据堆叠ID查询所有相关的堆叠结果
                cursor.execute(
                    "select stacking_algorithms_id, algorithms_name, model_parameters from stacking_algorithms where stacking_id=%s and successful=%s",
                    (stacking_id, True,))
                rows3 = cursor.fetchall()
                algorithms_type_names = [member.value for member in AlgorithmsType]
                for row in rows3:
                    regression_model_info = RegressionModelInfo()
                    regression_model_info.algorithms_id = row[0]
                    regression_model_info.algorithms_name = row[1]
                    regression_model_info.algorithms_type = [AlgorithmsType(name) if name in algorithms_type_names else AlgorithmsType.CUSTOM for name in row[1].split('|')]
                    model_parameters = pickle.loads(row[2])
                    regression_model_info.CV_best_score = model_parameters["best_score"]
                    regression_model_info.R2 = model_parameters["R2"]
                    regression_model_info.RMSE = model_parameters["RMSE"]
                    regression_model_info.stacking = True
                    all_model_metrics.append(regression_model_info)
        cursor.close()
        conn.close()
        sorted_objects = sorted(all_model_metrics, key=lambda x: x.CV_best_score, reverse=True)  # 按照R2对列表进行排序
        return sorted_objects

    '''
    检测某一模型是否适合进行堆叠建模
    '''
    def check_model_suitable_for_stacking(algorithm_name:str)->bool:
        if algorithm_name == AlgorithmsType.RK.value: # 回归克里金
            return False
        can_stacking = True
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select * from model_library where custom_model_name=%s and can_stacking=false",
            (algorithm_name,))
        rows = cursor.fetchall()
        if len(rows) > 0:  # 是不适合堆叠的算法
            can_stacking = False
        cursor.close()
        conn.close()
        return can_stacking

    '''
    从数据库中读取模型
    '''

    @staticmethod
    def retrive_model(algorithms_id: str) -> tuple:
        model = None
        mean_encoder = None
        origin_algorithms_name = None
        zscore_normalize = {}
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select model_content, mean_encoder, algorithms_name, zscore_normalize from algorithms where algorithms_id=%s",
            (algorithms_id,))
        rows = cursor.fetchall()
        if len(rows) > 0:  # 是单一算法的建模
            model = dill.loads(rows[0][0])
            if rows[0][1] is not None:
                mean_encoder = pickle.loads(rows[0][1])
            origin_algorithms_name = rows[0][2]
            zscore_normalize = pickle.loads(rows[0][3])
        else:  # 是堆叠建模
            cursor.execute(
                "select model_content, mean_encoder,algorithms_name,zscore_normalize from stacking_algorithms where stacking_algorithms_id=%s",
                (algorithms_id,))
            rows2 = cursor.fetchall()
            if len(rows2) > 0:  # 存在堆叠结果（必须存在）
                model = dill.loads(rows2[0][0])
                if rows2[0][1] is not None:
                    mean_encoder = pickle.loads(rows2[0][1])
                origin_algorithms_name = rows2[0][2]
                zscore_normalize = pickle.loads(rows2[0][3])
        cursor.close()
        conn.close()
        algorithmsType = None  # 算法的类型
        algorithms_info = {}    # 算法名称（字典对象，key为算法类型，value为算法可描述的名称）
        if "|" not in origin_algorithms_name:  # 单一算法建模
            if any(origin_algorithms_name in v.value for v in algorithms_dict.keys()): # 内置固有的算法
                algorithmsType = AlgorithmsType(origin_algorithms_name)
                algorithms_info = {algorithms_dict.get(algorithmsType):algorithmsType}
            else:  # 自定义算法
                algorithmsType = AlgorithmsType.CUSTOM
                algorithms_info = {origin_algorithms_name:algorithmsType}
        else: # 堆叠算法
            algorithmsType = AlgorithmsType.STACKING
            for algo_name in origin_algorithms_name.split('|'): # 遍历原始算法的名称
                if any(algo_name in v.value for v in algorithms_dict.keys()):  # 如果该名称可以在字典中找到
                    algorithms_info[algo_name]=AlgorithmsType(algo_name)
                else:
                    algorithms_info[algo_name] = AlgorithmsType.CUSTOM

        return algorithmsType, algorithms_info, model, mean_encoder, zscore_normalize

    '''
    获取算法对应的建模时的协变量（因为部分算法对多重共线性敏感，这些算法并不会使用所有的协变量建模）
    '''

    def retrive_used_covariate(algorithms_id: str) -> list[str]:
        covariate = None
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select left_cols1 from build_model where build_model_id in (select build_model_id from algorithms where algorithms_id=%s) or build_model_id in \
        (select build_model_id from stacking where stacking_id in (select stacking_id from stacking_algorithms where stacking_algorithms_id=%s))",
                       (algorithms_id, algorithms_id,))
        rows = cursor.fetchall()
        covariate = rows[0][0].split(',')  # 必须有一条记录
        cursor.close()
        conn.close()
        return covariate

    '''
    根据模型ID获取模型的任务ID信息
    '''

    @staticmethod
    def retrive_taskID_of_algorithms(algorithms_id: str) -> str:
        task_id = None
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select task_id from build_model where build_model_id in (select build_model_id from algorithms \
        where algorithms_id=%s)", (algorithms_id,))
        rows = cursor.fetchall()
        if len(rows) > 0:  # 是单一算法的建模
            task_id = rows[0][0]
        else:  # 是堆叠建模
            cursor.execute(
                "select task_id from build_model where build_model_id in (select build_model_id from stacking \
                where stacking_id in (select stacking_id from stacking_algorithms where stacking_algorithms_id=%s))",
                (algorithms_id,))
            rows2 = cursor.fetchall()
            if len(rows2) > 0:  # 存在堆叠结果（必须存在）
                task_id = rows2[0][0]
        cursor.close()
        conn.close()
        return task_id

    '''
    获取某一建模算法的性能指标参数
    '''

    @staticmethod
    def get_algorithms_metrics(algorithms_id: str) -> dict:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select model_parameters from algorithms where algorithms_id=%s", (algorithms_id,))
        rows = cursor.fetchall()
        if len(rows) == 0:
            cursor.execute("select model_parameters from stacking_algorithms where stacking_algorithms_id=%s",
                           (algorithms_id,))
            rows2 = cursor.fetchall()
            model_parameters = pickle.loads(rows2[0][0])
        else:
            model_parameters = pickle.loads(rows[0][0])
        cursor.close()
        conn.close()
        return pickle.loads(model_parameters)

    '''
    记录不确定制图的结果
    '''

    @staticmethod
    def record_uncertainty_mapping_result(algorithms_id: str, map_id: str, map_type: str) -> bool:
        successful = True
        conn = None
        try:
            conn = PostgresAccess.get_db_conn()
            cursor = conn.cursor()
            cursor.execute(
                "insert into map(map_id, algorithms_id, map_type) values(%s,%s,%s)",
                (map_id, algorithms_id, map_type,))
            conn.commit()
            cursor.close()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()  # 回滚事务
            print("插入数据时发生错误:", e)
            successful = False
        finally:
            if conn:
                conn.close()
        return successful
