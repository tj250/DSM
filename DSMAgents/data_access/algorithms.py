import requests
import psycopg2
from db import PostgresAccess
import uuid
import datetime

# 目标URL和JSON数据
url = 'https://127.0.0.1:8389'

'''
调用容器内的方法
api_path：api的路径，如"/build_model"
request_params:以字典形式封装的请求参数
'''


def call_docker(api_path: str, request_params: dict)->bool:
    # 发送POST请求（自动设置Content-Type为application/json）
    response = requests.post(
        url + api_path,
        json=request_params,  # 使用json参数自动序列化
        headers={'User-Agent': 'MyApp/1.0'}  # 可选自定义请求头
    )

    # 或者手动序列化（适用于需要特殊JSON配置的情况）
    # response = requests.post(
    #     url,
    #     data=json.dumps(data, indent=2),  # 手动序列化
    #     headers={'Content-Type': 'application/json'}
    # )

    print(f'状态码: {response.status_code}')
    print(f'响应内容: {response.text}')
    return response.status_code == 200


'''
调用回归建模方法
request_params:以字典形式封装的请求参数
'''


def call_regression_model(request_params: dict):
    return call_docker('/call_model', request_params)

    '''
    土壤属性制图任务数据模型：访问sqlite数据库，执行与任务相关的各类操作
    '''

class AlgorithmsExecutor:

    '''
    创建一个新的算法搜索任务
    '''
    @staticmethod
    def create_search_task(self)->str:
        task_id = str(uuid.uuid4())  # 生成一次搜索的任务ID
        successful = True
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("insert into search_task(search_task_id,create_dt) values(%s,%s)", (task_id, datetime.now()))
            conn.commit()
        except psycopg2.Error as e:
            print("插入数据时发生错误:", e)
            successful = False
        finally:
            cursor.close()
            conn.close()

        return successful, task_id
