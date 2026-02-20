import datetime
import sqlite3
import uuid
import time, sys
from dataclasses import dataclass
import json

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

'''
访问sqlite数据库
'''


class DBAccess:
    @staticmethod
    def get_db_conn():
        return sqlite3.connect('./tools.db')


'''
建模结果指标信息
'''


@dataclass
class RegressionModelInfo():
    model_id: str = None                        # 建模ID
    model_name: str = None                      # 建模的模型名称
    call_id: str = None                         # 数据挖掘建模任务ID（来自客户端调用）
    input_params: str = None                    # 输入参数，用|分隔
    R2: float = sys.float_info.max              # 建模结果中的R2指标
    RMSE: float = sys.float_info.max            # 建模结果中的RMSE指标
    start_time: str = None                      # 建模开始时间
    finished_time: str = None                   # 建模结束时间
    finished:bool = False                       # 是否已完成建模

'''
回归预测信息
'''
@dataclass
class PredictionInfo():
    prediction_id: str = None                   # 预测任务ID
    model_id: str = None                        # 预测时所使用的模型的模型ID
    call_id: str = None                         # 预测调用ID（来自客户端调用）
    prediction_type:int = -1                    # 预测的类型，1：栅格数据预测，2-矢量数据预测，3-平面表格数据预测
    input_params: str = None                    # 输入参数，用|分隔
    start_time: str = None                      # 建模开始时间
    finished_time: str = None                   # 建模结束时间
    finished:bool = False                       # 是否已完成建模



'''
任务数据模型：访问sqlite数据库，执行与任务相关的各类操作
'''


class TaskDataAccess:
    '''
    向数据库插入一条建模任务
    '''

    @staticmethod
    def add_build_model_info(call_id: str, model_name: str, input_params: str):
        build_model_id = str(uuid.uuid1())
        conn = DBAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("insert into model_info(model_id,call_id,model_name,input_params, start_dt,finished) values (?,?,?,?,?,0)",
                       (build_model_id, call_id, model_name, input_params,
                        time.strftime(DATETIME_FORMAT, datetime.datetime.now().timetuple()),))
        conn.commit()
        cursor.close()
        conn.close()
        return build_model_id

    '''
    更新数据库中已有的建模任务,将其设置为结束
    '''

    @staticmethod
    def set_build_model_finished(build_model_id: str, metrics: str):
        conn = DBAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("update model_info set metrics=?, end_dt=?,finished=1 where model_id=?",
                       (metrics, time.strftime(DATETIME_FORMAT, datetime.datetime.now().timetuple()),
                        build_model_id,))
        conn.commit()
        cursor.close()
        conn.close()

    '''
    获取线程需要处理的下一个任务，任务可能为建模，也可能为预测。
    '''
    @staticmethod
    def get_next_task():
        conn = DBAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select model_id,model_name,input_params from model_info where finished=0 order by start_dt ASC LIMIT 1")
        rows = cursor.fetchall()
        if len(rows) != 0:
            one_model_parameter = RegressionModelInfo()
            one_model_parameter.model_id = rows[0][0]
            one_model_parameter.model_name = rows[0][1]
            one_model_parameter.input_params = rows[0][2]
            cursor.close()
            conn.close()
            return one_model_parameter

        cursor.execute(
            "select prediction_id,model_id,prediction_type, input_params from regression_info where finished=0 order by start_dt ASC LIMIT 1")
        rows = cursor.fetchall()
        if len(rows) != 0:
            regression_parameter = PredictionInfo()
            regression_parameter.prediction_id = rows[0][0]
            regression_parameter.model_id = rows[0][1]
            regression_parameter.prediction_type = rows[0][2]
            regression_parameter.input_params = rows[0][3]
            cursor.close()
            conn.close()
            return regression_parameter

        cursor.close()
        conn.close()
        return None

    '''
    确定指定调用ID下的相关建模任务是否均已结束
    '''
    @staticmethod
    def build_model_task_is_finished(call_id: str) -> bool:
        conn = DBAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select finished from model_info where call_id=?",
            (call_id,))
        rows = cursor.fetchall()
        finished = True
        for row in rows:
            if row[0] == 0:
                finished = False
                break
        cursor.close()
        conn.close()
        return finished

    '''
    从数据库中查询模型信息
    '''

    @staticmethod
    def get_model_info(model_id: str) -> tuple[bool,RegressionModelInfo]:
        conn = DBAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select model_id,call_id,model_name,input_params, metrics,start_dt,end_dt,finished from model_info where model_id=?",
            (model_id,))
        rows = cursor.fetchall()
        if len(rows) == 1:
            one_model_parameter = RegressionModelInfo()
            row = rows[0]
            one_model_parameter.model_id = row[0]
            one_model_parameter.call_id = row[1]
            one_model_parameter.model_name = row[2]
            one_model_parameter.input_params = row[3]
            if row[4] is not None:
                build_model_metrics = json.loads(row[4])
                one_model_parameter.RMSE = build_model_metrics["RMSE"]
                one_model_parameter.R2 = build_model_metrics["R2"]
            one_model_parameter.start_time = row[5]
            if row[6] is not None:
                one_model_parameter.finished_time = row[6]
            one_model_parameter.finished = (row[7] == 1)
            cursor.close()
            conn.close()
            return True, one_model_parameter
        else:
            cursor.close()
            conn.close()
            False, None


    '''
    从数据库中查询所有调用相关的建模结果
    '''

    @staticmethod
    def get_all_model_infos(call_id: str) -> list[RegressionModelInfo]:
        conn = DBAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select model_id,call_id,model_name,input_params, metrics,start_dt,end_dt,finished from model_info where call_id=?",
            (call_id,))
        rows = cursor.fetchall()
        all_model_infos = []
        for row in rows:
            one_model_parameter = RegressionModelInfo()
            one_model_parameter.model_id = row[0]
            one_model_parameter.call_id = row[1]
            one_model_parameter.model_name = row[2]
            one_model_parameter.input_params = row[3]
            if row[4] is not None:
                build_model_params = json.loads(row[4])
                one_model_parameter.RMSE = build_model_params["RMSE"]
                one_model_parameter.R2 = build_model_params["R2"]
            one_model_parameter.start_time = row[5]
            if row[6] is not None:
                one_model_parameter.finished_time = row[6]
            one_model_parameter.finished = (row[7] == 1)
            all_model_infos.append(one_model_parameter)
        cursor.close()
        conn.close()
        return all_model_infos

    '''
    向数据库插入一条回归预测任务
    '''

    @staticmethod
    def add_prediction_task_info(call_id: str, model_id: str, prediction_type:int, input_params: str):
        print(call_id)
        print(model_id)
        print(input_params)
        prediction_id = str(uuid.uuid1())
        conn = DBAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("insert into regression_info(prediction_id,call_id,model_id,prediction_type,input_params, start_dt,finished) values (?,?,?,?,?,?,0)",
                       (prediction_id, call_id, model_id, prediction_type,input_params,
                        time.strftime(DATETIME_FORMAT, datetime.datetime.now().timetuple()),))
        conn.commit()
        cursor.close()
        conn.close()
        return prediction_id

    '''
    设置某一预测任务结束
    '''

    @staticmethod
    def set_prediction_task_finished(prediction_id: str):
        conn = DBAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("update regression_info set end_dt=?,finished=1 where prediction_id=?",
                       (time.strftime(DATETIME_FORMAT, datetime.datetime.now().timetuple()),
                        prediction_id,))
        conn.commit()
        cursor.close()
        conn.close()


    '''
    指定的回归预测过程（可能包含多个预测）是否均已经结束
    '''
    @staticmethod
    def prediction_task_is_finished(task_call_id: str) -> bool:
        conn = DBAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select finished from regression_info where call_id=?",
            (task_call_id,))
        rows = cursor.fetchall()
        finished = True
        for row in rows:
            if row[0] == 0:
                finished = False
                break
        cursor.close()
        conn.close()
        return finished
