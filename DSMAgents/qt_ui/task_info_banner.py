from PyQt5.QtWidgets import QLabel
from data_access.task import TaskInfo

class TaskInfoWidget(QLabel):
    """显示任务信息"""

    def __init__(self, parent=None):
        super().__init__(parent)

    '''
    更新任务信息
    '''
    def update_task_info(self, task_info: TaskInfo):
        self.setText(task_info.task_name)
