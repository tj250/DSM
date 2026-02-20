import os.path
import math
import pickle,dill
import time, json
import algorithms_config
from db_access.postgres_db import PostgresAccess
from DSMAlgorithms.base.dsm_base_model import AlgorithmsType
from DSMAlgorithms.base.base_data_structure import DataDistributionType


'''
土壤属性制图建模任务数据访问：访问数据库，执行与任务相关的各类操作
'''


class TaskDataAccess:

    '''
    更新单模型的建模结果
    algorithms_id:算法ID
    model：模型内容
    zscore_normalize:数据的z-score信息或min-max归一化信息
    mean_encoder:平均编码的结果（如果使用了平均编码）
    model_params:模型的参数
    '''


    @staticmethod
    def save_model(algorithms_id: str, model, zscore_normalize: dict, mean_encoder:dict, model_params: dict) -> None:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("update algorithms set end_dt=%s, model_content=%s, mean_encoder=%s, zscore_normalize=%s, \
        model_parameters=%s, cv_best_score=%s, r2=%s, successful=%s where algorithms_id=%s",
            (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), dill.dumps(model),
             None if mean_encoder is None else pickle.dumps(mean_encoder), pickle.dumps(zscore_normalize),
             pickle.dumps(model_params), 0 if math.isnan(model_params['best_score']) else model_params['best_score'],
             model_params['R2'], True, algorithms_id,))
        conn.commit()
        cursor.execute("select build_model_id from algorithms where algorithms_id=%s", (algorithms_id,))
        rows = cursor.fetchall()
        build_model_id = rows[0][0]  # 获取建模算法相关的建模批次ID
        cursor.execute("select count(*) from algorithms where build_model_id=%s and end_dt is null", (build_model_id,))
        rows2 = cursor.fetchall()
        all_finished = rows2[0][0] == 0
        if all_finished:  # 如果均已结束，则设置build_model中的end_dt为结束
            cursor.execute(
                "update build_model set end_dt=%s where build_model_id=%s",
                (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), build_model_id,))
            conn.commit()
            # 获取建模相关的任务ID
            cursor.execute("select task_id from build_model where build_model_id=%s", (build_model_id,))
            rows3 = cursor.fetchall()
            task_id = rows3[0][0]  # 获取建模算法相关的任务ID
            # 获取建模相关的算法的最大r2
            cursor.execute("select max(cv_best_score) from algorithms where build_model_id=%s", (build_model_id,))
            rows4 = cursor.fetchall()
            max_r2 = rows4[0][0]
            # 更新任务表中所记录的最大的建模r2
            cursor.execute("update task set predict_max_r2=%s,do_stacking=%s where task_id=%s", (max_r2, False, task_id,))
            conn.commit()
        cursor.close()
        conn.close()

    '''
    更新堆叠模型的建模结果。
    stacking_algorithms_id:算法ID
    model：模型内容
    zscore_normalize:数据的z-score信息或min-max归一化信息
    mean_encoder:平均编码的结果（如果使用了平均编码）
    model_params:模型的参数
    '''

    @staticmethod
    def save_stacking_model(stacking_algorithms_id: str, model, zscore_normalize: dict, mean_encoder: dict, model_params: dict) -> None:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "update stacking_algorithms set end_dt=%s, model_content=%s, mean_encoder=%s,zscore_normalize=%s, \
            model_parameters=%s, cv_best_score=%s, r2=%s, successful=%s where stacking_algorithms_id=%s",
            (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), dill.dumps(model),
             None if mean_encoder is None else pickle.dumps(mean_encoder), pickle.dumps(zscore_normalize),
             pickle.dumps(model_params), 0 if math.isnan(model_params['best_score']) else model_params['best_score'],
             model_params['R2'], True, stacking_algorithms_id,))
        conn.commit()
        cursor.execute("select stacking_id from stacking_algorithms where stacking_algorithms_id=%s",
                       (stacking_algorithms_id,))
        rows = cursor.fetchall()
        stacking_id = rows[0][0]  # 获取堆叠算法相关的堆叠批次ID

        cursor.execute("select build_model_id from stacking where stacking_id=%s", (stacking_id,))
        rows2 = cursor.fetchall()
        build_model_id = rows2[0][0]  # 获取建模算法相关的建模批次ID

        cursor.execute("select count(*) from stacking_algorithms where stacking_id=%s and end_dt is null", (stacking_id,))
        rows3 = cursor.fetchall()
        all_finished = rows3[0][0] == 0
        if all_finished:  # 如果均已结束，则设置stacking中的end_dt为结束
            cursor.execute(
                "update stacking set end_dt=%s where stacking_id=%s",
                (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), stacking_id,))
            conn.commit()
            # 获取建模相关的任务ID
            cursor.execute("select task_id from build_model where build_model_id=%s", (build_model_id,))
            rows4 = cursor.fetchall()
            task_id = rows4[0][0]  # 获取建模算法相关的任务ID
            # 获取堆叠建模相关的算法的最大r2
            cursor.execute("select max(cv_best_score) from stacking_algorithms where stacking_id=%s", (stacking_id,))
            rows5 = cursor.fetchall()
            max_r2 = rows5[0][0]
            # 更新任务表中所记录的最大的堆叠建模r2
            cursor.execute("update task set stacking_predict_max_r2=%s,do_stacking=%s where task_id=%s",
                           (max_r2, True, task_id, ))
            conn.commit()
        cursor.close()
        conn.close()

    '''
    设置单个模型的建模过程失败
    '''
    @staticmethod
    def set_build_model_failed(algorithms_id: str):
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "update algorithms set end_dt=%s, successful=%s where algorithms_id=%s",
            (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), False, algorithms_id,))
        conn.commit()
        cursor.execute("select build_model_id from algorithms where algorithms_id=%s", (algorithms_id,))
        rows = cursor.fetchall()
        build_model_id = rows[0][0]  # 获取建模算法相关的建模批次ID
        cursor.execute("select count(*) from algorithms where build_model_id=%s and end_dt is null", (build_model_id,))
        rows2 = cursor.fetchall()
        all_finished = rows2[0][0] == 0
        if all_finished:  # 如果均已结束，则设置build_model中的end_dt为结束
            cursor.execute("update build_model set end_dt=%s where build_model_id=%s",
                (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), build_model_id,))
            conn.commit()
        cursor.close()
        conn.close()

    '''
    设置模型堆叠失败
    '''
    def set_stacking_failed(stacking_algorithms_id: str) -> None:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "update stacking_algorithms set end_dt=%s, successful=%s where stacking_algorithms_id=%s",
            (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), False, stacking_algorithms_id,))
        conn.commit()
        cursor.execute("select stacking_id from stacking_algorithms where stacking_algorithms_id=%s",
                       (stacking_algorithms_id,))
        rows = cursor.fetchall()
        stacking_id = rows[0][0]  # 获取堆叠算法相关的堆叠批次ID
        cursor.execute("select count(*) from stacking_algorithms where stacking_id=%s and end_dt is null", (stacking_id,))
        rows2 = cursor.fetchall()
        all_finished = rows2[0][0] == 0
        if all_finished:  # 如果均已结束，则设置stacking中的end_dt为结束
            cursor.execute(
                "update stacking set end_dt=%s where stacking_id=%s",
                (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), stacking_id,))
            conn.commit()
        cursor.close()
        conn.close()


    '''
    获取线程需要处理的下一个任务。
    '''

    @staticmethod
    def get_next_task():
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        task_type = None
        # 先从build_model中查询出尚未完成建模的第一条任务(当前时间和建模最后一次活动时间的差值超过2分钟)
        cursor.execute(
            "select algorithms_id, algorithms_name,build_model_id from algorithms \
            where end_dt is null and CURRENT_TIMESTAMP - last_active_dt > INTERVAL '%s minutes' order by start_dt ASC LIMIT 1",
            (algorithms_config.TASK_MAX_DURATION,))
        rows = cursor.fetchall()
        model_info = {}
        if len(rows) != 0:  # 存在未完成的建模记录
            model_info['algorithms_id'] = rows[0][0]
            model_info['algorithms_name'] = rows[0][1]  # 将算法名称作为算法类型
            build_model_id = rows[0][2]
            # 进一步，查询建模的目标属性
            cursor.execute(
                "select sample_file,categorical_vars_info,prediction_variable,covariates_directory, \
                data_distribution_type,left_cols1,left_cols2,left_cols3 from build_model where build_model_id=%s",
                (build_model_id,))
            rows2 = cursor.fetchall()
            model_info['file_name'] = os.path.basename(rows2[0][0])
            model_info['categorical_vars'] = {} if rows2[0][1] is None else json.loads(rows2[0][1])
            model_info['prediction_variable'] = rows2[0][2]
            model_info['covariates_directory'] = rows2[0][3]
            model_info['data_distribution_type'] = DataDistributionType(rows2[0][4])
            model_info['left_cols1'] = rows2[0][5].split(',') if rows2[0][5] != '' else []
            model_info['left_cols2'] = rows2[0][6].split(',') if rows2[0][6] != '' else []
            model_info['left_cols3'] = rows2[0][7].split(',') if rows2[0][7] != '' else []
            task_type = 1  # 单一建模类型的任务
        else:
            # 如果没有单一建模任务，则进一步看是否有堆叠模型任务尚未完成
            cursor.execute(
                "select stacking_algorithms_id, algorithms_name,stacking_id from stacking_algorithms \
                where end_dt is null and CURRENT_TIMESTAMP - last_active_dt > INTERVAL '%s minutes' order by start_dt ASC LIMIT 1",
                (algorithms_config.TASK_MAX_DURATION,))
            rows4 = cursor.fetchall()
            model_info = {}
            if len(rows4) != 0:  # 存在失败建模记录
                model_info['stacking_algorithms_id'] = rows4[0][0]
                model_info['algorithms_name'] = rows4[0][1].split('|')
                stacking_id = rows4[0][2]
                # 进一步，查询建模的目标属性
                cursor.execute("select sample_file,categorical_vars_info,prediction_variable,covariates_directory, \
                data_distribution_type,left_cols1,left_cols2,left_cols3 from build_model where build_model_id in \
                (select build_model_id from stacking where stacking_id=%s)", (stacking_id,))
                rows5 = cursor.fetchall()
                model_info['file_name'] = os.path.basename(rows5[0][0])
                model_info['categorical_vars'] = {} if rows5[0][1] is None else json.loads(rows5[0][1])
                model_info['prediction_variable'] = rows5[0][2]
                model_info['covariates_directory'] = rows5[0][3]
                model_info['data_distribution_type'] = DataDistributionType(rows5[0][4])
                model_info['left_cols1'] = rows5[0][5].split(',') if rows5[0][5] != '' else []
                model_info['left_cols2'] = rows5[0][6].split(',') if rows5[0][6] != '' else []
                model_info['left_cols3'] = rows5[0][7].split(',') if rows5[0][7] != '' else []
                task_type = 2  # 堆叠模型类型的任务

        cursor.close()
        conn.close()
        return model_info, task_type
