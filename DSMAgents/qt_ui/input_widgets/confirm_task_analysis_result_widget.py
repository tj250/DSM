from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt
from data_access.task import TaskDataAccess
from agents.utils.views import ConversationMessageType,TaskAnalysisConfirmUIChoice
from ..utility import create_table_widget
from ..basic_widget.countdown_button import CountdownButton
from qt_ui.interactive_widget import InteractiveWidget

'''
文本区域输入widget
'''
class ConfirmTaskAnalysisResultWidget(QWidget):

    def __init__(self, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent


    def set_confirm_info(self, confirm_info:dict):
        # ---- 左侧待确认信息区域 ----
        table_content = []
        for key,value in confirm_info.items():
            table_content.append([key, value])
        # 创建显示表格
        table_widget = create_table_widget(["待确认项", "待确认内容"], table_content)
        table_widget.resizeRowToContents(1) # 允许任务概述行高度随内容多少自适应
        # 不显示表头
        table_widget.horizontalHeader().setVisible(False)

        # ---- 右侧按钮区域 ----
        button_container = QWidget()
        # button_container.setStyleSheet("background-color: #FFE4B5;")  # 设置容器背景色
        button_layout = QVBoxLayout(button_container)
        # 创建按钮
        again_btn = QPushButton("重新进行分析")
        again_btn.clicked.connect(self.emit_continue_analysis_message)
        button_layout.addWidget(again_btn)

        ok_btn = CountdownButton("进入下一环节", 10)
        ok_btn.clicked.connect(self.emit_enter_next_step_message)
        # ok_btn = QPushButton("进入下一环节.....")
        # ok_btn.clicked.connect(self.emit_enter_next_step_message)
        button_layout.addWidget(ok_btn)

        # 总体布局
        layout = QHBoxLayout()
        # 设置布局间距和边距
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(table_widget, stretch=1)
        layout.addWidget(button_container, stretch=0)
        self.setLayout(layout)

    '''
    重新指定数据源
    '''
    def emit_continue_analysis_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(TaskAnalysisConfirmUIChoice.AnalyzingAgain)
        # 2-用户消息写入数据库
        user_message = "用户确认需重新开启新一轮任务分析"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        # self.parent_widget.parent().parent().show_loading()

    '''
    用户确认进入下一个环节
    '''
    def emit_enter_next_step_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(TaskAnalysisConfirmUIChoice.EnterNextStep)
        # 2-用户消息写入数据库
        user_message = "用户已确认任务分析的结果，进入下一个环节"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "进入下一环节...")

