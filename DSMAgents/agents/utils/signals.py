import threading
from PyQt5.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from agents.utils.views import DEUIType
from data_access.task import TaskInfo

'''
获取用户输入的事件类
'''
class InputFinishedEvent(threading.Event):
    def __init__(self):
        super().__init__()
        self.data = None

    def set_with_data(self, data):
        self.data = data
        self.set()

'''
Agent相关的通知信号
'''
class AgentSignals(QObject):
    """Signals from a running worker thread.

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc())

    result
        object data returned from processing, anything

    progress
        float indicating % progress
    """
    # 更加UI层的进度指示
    task_progress = pyqtSignal(str, str)  # 发出在用户交互区更新界面进度的请求

    # 接收任务阶段的事件
    wait_input_task_desc = pyqtSignal(str, InputFinishedEvent, str, str)    # 请求用户输入任务描述的通知
    user_confirm_task_analysis_result = pyqtSignal(str, InputFinishedEvent, dict) # 请求用户做出选择（重新进入分析任务环节还是已确认分析结果）
    task_analysis_finished = pyqtSignal(str, str, str)              # 土壤属性制图任务分析结束,第一个子图执行完成的最后

    # 数据探索阶段的事件
    user_set_data_source = pyqtSignal(str, InputFinishedEvent, str, str)             # 请求用户设置数据源的通知
    user_confirm_predict_var = pyqtSignal(str, InputFinishedEvent, list, str,str)   # 请求用户指定预测变量
    user_confirm_interpretation_vars = pyqtSignal(str, InputFinishedEvent, list, list)  # 请求用户指定解释变量
    user_confirm_categorical_variables = pyqtSignal(str, InputFinishedEvent, list, list, str)   # 请求用户修正类别型变量
    user_choice = pyqtSignal(str, DEUIType, InputFinishedEvent, str)                      # 请求用户做出选择（指定数据源还是直接指定预测变量）
    user_confirm_data_explorer = pyqtSignal(str, InputFinishedEvent, dict, dict, dict) # 请求用户确认数据探索的最终结果
    data_source_ready = pyqtSignal(str, str)                                   # 数据准备事件确认,发生于第二个子图执行完成的最后

    # 数据建模阶段的事件
    user_confirm_build_model_result = pyqtSignal(str, InputFinishedEvent, list) # 请求用户做出选择（进行模型堆叠还是进入下一环节）

    # 评估阶段的事件
    user_choice_mapping_model = pyqtSignal(str, InputFinishedEvent, TaskInfo, list)   # 请求用户对预测信息进行设置和确认
    user_confirm_mapping_result = pyqtSignal(str, InputFinishedEvent, TaskInfo, list) # 请求用户确认制图的结果
    user_confirm_evaluating_params = pyqtSignal(str, InputFinishedEvent) # 请求用户确认评估参数
    user_confirm_evaluating_result = pyqtSignal(str, InputFinishedEvent, list, list) # 请求用户确认评估的结果

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(float)

    def __init__(self, task_id:str):
        super().__init__()
        self.task_id = task_id
        self.input_finished_event = None

    '''
    创建输入事件类
    '''
    def create_input_finished_event(self):
        if self.input_finished_event is None:
            self.input_finished_event = InputFinishedEvent()
        return self.input_finished_event