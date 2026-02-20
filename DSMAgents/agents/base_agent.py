import asyncio
import threading
from agents.utils.signals import AgentSignals
from abc import ABC, abstractmethod
from agents.data_structure.task_state import TaskState

'''
用作所有Agent的基类
'''


class BaseAgent(ABC):

    '''
    构造方法
    agent_signals：Agent的通知信号
    '''

    def __init__(self, agent_signals: AgentSignals):
        self.agent_signals = agent_signals
        self.interrput_event = threading.Event()
        self.quit_chat = False  # 初始默认尚不能退出chat

    '''
    显示UI进度
    message:要显示的消息字符串
    '''

    def show_ui_progress(self, message: str):
        self.agent_signals.task_progress.emit(self.agent_signals.task_id, message)

    '''
    初始化工作流方法
    checkpointer:保存点对象
    '''

    @abstractmethod
    def init_workflow(self, checkpointer):
        pass

    '''
    异步主控函数
    global_state:全局状态对象
    '''
    @abstractmethod
    async def async_run(self, global_state: TaskState):
        pass

    '''
    供外部调用的同步主控函数
    global_state:全局状态对象
    '''

    def run(self, global_state: TaskState) -> TaskState:
        return asyncio.run(self.async_run(global_state))
