import datetime
import uuid
import time
import psycopg2
from dataclasses import dataclass
from agents.utils.views import ConversationMessageType, TaskStage, PredictionType
from data_access.db import PostgresAccess

'''
土壤属性制图任务信息数据结构
'''


@dataclass
class TaskInfo():
    task_id: str = ''                   # 任务ID
    task_name: str = ''                 # 任务名称
    description: str = ''               # 有关任务的详细描述
    soil_property: str = ''             # 待预测的土壤属性
    last_build_model_id: str = ''       # 最后一次的建模调用ID
    create_dt: datetime.datetime = datetime.datetime.now()  # 任务创建时间
    stage: TaskStage = TaskStage.Uncertain      # 任务所处阶段
    summary: str = ''                           # 任务总结，用LLM形成
    prediction_type = PredictionType.Raster     # 预测模式
    sample_file: str = ''               # 土壤样点数据文件（*.csv,*.shp等结构化数据）
    covariates_path: str = ''           # 环境协变量（解释变量）文件所在的目录，目录下存储一系列的栅格文件
    mapping_area_file: str = ''         # 制图区域文件，限定了预测的空间边界范围，可以是栅格文件(.tif,.img)或矢量文件（.shp）


'''
土壤属性制图任务会话历史数据结构
'''

@dataclass
class TaskConversation():
    conversation_id: str = ''  # 会话ID
    conversation_index: int = -1  # 会话记录的顺序
    conversation_type: ConversationMessageType = ConversationMessageType.Uncertain  # 会话类型
    content: str = ''  # 会话内容
    create_dt: datetime.datetime = datetime.datetime.now()  # 会话生成时间


'''
土壤属性制图任务数据模型：访问sqlite数据库，执行与任务相关的各类操作
'''

class TaskDataAccess:
    '''
    从数据库中查询所有已有任务
    '''

    @staticmethod
    def get_task_info(task_id: str) -> TaskInfo:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "select soil_property,task_name,create_dt,stage,summary,description,build_model_call_id,prediction_type, \
             bm_sample_file, bm_covariates_directory, bm_mapping_area_file from task where task_id=%s",
            (task_id,))
        rows = cursor.fetchall()
        task_info = TaskInfo()
        task_info.task_id = task_id
        task_info.soil_property = rows[0][0]
        task_info.task_name = rows[0][1]
        task_info.create_dt = rows[0][2]
        task_info.stage = TaskStage(rows[0][3])
        task_info.summary = rows[0][4]
        task_info.description = rows[0][5]
        task_info.last_build_model_id = rows[0][6]
        task_info.prediction_type = PredictionType(rows[0][7])
        task_info.sample_file = rows[0][8]
        task_info.covariates_path = rows[0][9]
        task_info.mapping_area_file = rows[0][10]

        if task_info.last_build_model_id is not None:  # 在存在建模批次的情况下，从build_model表中获取最新一个批次的建模信息
            cursor.execute(
                "select sample_file,covariates_directory,mapping_area_file from build_model where build_model_id=%s",
                (task_info.last_build_model_id,))
            rows = cursor.fetchall()
            task_info.sample_file = rows[0][0]
            task_info.covariates_path = rows[0][1]
            task_info.mapping_area_file = rows[0][2]

        cursor.close()
        conn.close()
        return task_info

    '''
    向数据库中添加一条新的任务
    当用户初次创建任务时，调用此方法
    其中bm_sample_file, bm_covariates_directory, bm_cartography_area_file信息只是初始建模的数据信息
    '''

    @staticmethod
    def create_task(task_info: TaskInfo) -> TaskInfo:
        conn = None
        try:
            conn = PostgresAccess.get_db_conn()
            cursor = conn.cursor()
            cursor.execute(
                "insert into task(task_id, task_name, soil_property, description, prediction_type, create_dt,stage, \
                bm_sample_file, bm_covariates_directory, bm_mapping_area_file) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (task_info.task_id, task_info.task_name, task_info.soil_property, task_info.description,
                 task_info.prediction_type.value, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                 TaskStage.AcceptTask.value, task_info.sample_file, task_info.covariates_path, task_info.mapping_area_file,))
            conn.commit()
            cursor.close()
            conn.close()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()  # 回滚事务
                print("插入数据时发生错误:", e)
        finally:
            if conn:
                conn.close()
        return task_info

    @staticmethod
    def update_task(task_info: TaskInfo)->bool:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            "update task set soil_property=%s, task_name=%s, description=%s, prediction_type=%s where task_id=%s",
            (task_info.soil_property.value, task_info.task_name, task_info.description,
             task_info.prediction_type.value, task_info.task_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return True

    '''
    更新任务的描述
    '''

    @staticmethod
    def update_task_desc(task_id: str, task_desc: str)->bool:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("update task set description=%s where task_id=%s", (task_desc, task_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return True

    '''
    更新任务的样点数据源
    '''

    @staticmethod
    def update_task_build_model_data_source(task_id: str, sample_file: str)->bool:
        task_info = TaskDataAccess.get_task_info(task_id)
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("update build_model set sample_file=%s where build_model_id=%s",
            (sample_file, task_info.last_build_model_id, ))  # 建模数据源
        conn.commit()
        cursor.close()
        conn.close()
        return True

    '''
    更新任务最新一次建模的预测数据设置
    '''

    @staticmethod
    def update_task_prediction_data_setting(task_id: str, covariates_directory: str, mapping_area_file: str):
        task_info = TaskDataAccess.get_task_info(task_id)  # 由task表来获取最新一次建模的ID
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("update build_model set covariates_directory=%s,mapping_area_file=%s where build_model_id=%s",
            (covariates_directory, mapping_area_file, task_info.last_build_model_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return True

    '''
    从数据库中查询所有已有任务
    '''

    @staticmethod
    def get_all_tasks() -> list[TaskInfo]:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select task_id,soil_property,task_name,create_dt,stage,description,prediction_type,summary from task order by create_dt asc")
        rows = cursor.fetchall()
        all_tasks = []
        for row in rows:
            one_task = TaskInfo()
            one_task.task_id = row[0]
            one_task.soil_property = row[1]
            one_task.task_name = row[2]
            one_task.create_dt = row[3]
            one_task.stage = TaskStage(row[4])
            one_task.description = row[5]
            one_task.prediction_type = PredictionType(row[6])
            one_task.summary = row[7]
            all_tasks.append(one_task)
        cursor.close()
        conn.close()
        return all_tasks

    '''
    删除一条任务
    '''

    @staticmethod
    def delete_task(task_id: str):
        """删除当前任务项"""
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("delete from task where task_id=%s", (task_id,))
        conn.commit()
        cursor.close()
        conn.close()

    @staticmethod
    def get_conversation_histories(task_id: str):
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select conversation_id,conversation_index,conversation_type, content,create_dt \
        from task_conversation where task_id=%s order by conversation_index asc", (task_id,))
        rows = cursor.fetchall()
        all_one_conversations = []
        for row in rows:
            one_conversation = TaskConversation()
            one_conversation.conversation_id = row[0]
            one_conversation.conversation_index = row[1]
            one_conversation.conversation_type = ConversationMessageType(row[2])
            one_conversation.content = row[3]
            one_conversation.create_dt = row[4]
            all_one_conversations.append(one_conversation)
        cursor.close()
        conn.close()
        return all_one_conversations

    @staticmethod
    def add_conversation(task_id: str, conversation: TaskConversation):
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select max(conversation_index) from task_conversation where task_id=%s", (task_id,))
        rows = cursor.fetchall()
        if rows[0][0] is None:
            conversation_index = 1
        else:
            conversation_index = rows[0][0] + 1
        cursor.execute(
            "insert into task_conversation(conversation_id, task_id, conversation_index, conversation_type, \
            content, create_dt) values(%s,%s,%s,%s,%s,%s)", (str(uuid.uuid1()), task_id,
                                                             conversation_index,
                                                             conversation.conversation_type.value,
                                                             conversation.content,
                                                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),))
        conn.commit()
        cursor.close()
        conn.close()

    '''
    添加用户消息
    '''

    @staticmethod
    def add_user_conversation(task_id: str, content: str):
        conversation = TaskConversation()
        conversation.content = content  # 消息内容
        conversation.conversation_type = ConversationMessageType.UserMessage  # 用户消息
        TaskDataAccess.add_conversation(task_id, conversation)

    '''
    添加系统消息
    '''

    @staticmethod
    def add_system_conversation(task_id: str, content: str):
        conversation = TaskConversation()
        conversation.content = content  # 消息内容
        conversation.conversation_type = ConversationMessageType.SystemMessage  # 系统消息
        TaskDataAccess.add_conversation(task_id, conversation)

    '''
    添加Agent的消息
    '''

    @staticmethod
    def add_agent_conversation(task_id: str, content: str):
        conversation = TaskConversation()
        conversation.content = content  # 消息内容
        conversation.conversation_type = ConversationMessageType.AgentMessage  # Agent消息
        TaskDataAccess.add_conversation(task_id, conversation)

    '''
    在确定土壤属性制图任务的要求后，写入到数据库
    '''

    @staticmethod
    def confirm_task_basic_info(task_id: str, soil_property: str, task_summary: str):
        TaskDataAccess.add_system_conversation(task_id, task_summary)  # 在会话表中也增加一条系统消息
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("update task set soil_property=%s, stage=%s,summary=%s where task_id=%s",
                       (soil_property, TaskStage.DataExplore.value, task_summary, task_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return True

    '''
    更新任务所处的阶段
    '''

    @staticmethod
    def update_task_stage(task_id: str, stage: TaskStage):
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("update task set stage=%s where task_id=%s", (stage.value, task_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return True
