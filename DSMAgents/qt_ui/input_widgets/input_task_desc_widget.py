from PyQt5.QtWidgets import (QFrame, QWidget, QStyleOptionButton, QHBoxLayout, QTextEdit, QPushButton,QStyle)
from PyQt5.QtCore import Qt, QSize
from qt_ui.toast_widget import ToastWidget
from data_access.task import TaskDataAccess
from agents.utils.views import ConversationMessageType
from ..basic_widget.countdown_button import CountdownButton
from qt_ui.interactive_widget import InteractiveWidget

'''
文本区域输入widget
'''


class TextInputWidget(QWidget):

    def __init__(self, task_desc:str, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent

        # ---- 左侧文本输入区域 ----
        self.text_edit = QTextEdit()
        if task_desc == '':
            self.text_edit.setPlaceholderText("请在此输入内容，描述需要执行的土壤属性制图任务")
        else:
            self.text_edit.setText(task_desc)
        self.text_edit.setFrameStyle(QFrame.Shape.NoFrame)
        self.text_edit.setStyleSheet("QTextEdit { background-color: white; }")


        # ---- 右侧按钮区域 ----
        # 创建按钮
        ok_btn = CountdownButton("确定", 10)
        ok_btn.clicked.connect(self.emit_user_chat_message)

        # 总体布局
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下
        layout.addWidget(self.text_edit, stretch=1)  # 自动拉伸填充空间
        layout.addWidget(ok_btn, stretch=0)
        self.setLayout(layout)

    '''
    发送用户输入文本
    '''

    def emit_user_chat_message(self):
        if self.text_edit.toPlainText().strip() == '':
            self.toast = ToastWidget("请输入内容，再进行提交！！！", self.parent())
            self.toast.show_toast()
        else:
            task_desc = self.text_edit.toPlainText().strip()
            # 1-通知agent从中断处恢复执行
            # todo:按钮自身暂时禁用
            self.parent_widget.finished_event.set_with_data(task_desc)
            # 2-用户消息写入数据库
            TaskDataAccess.add_user_conversation(self.parent_widget.task_id, task_desc)
            # 将更新后的任务描述写入数据库
            TaskDataAccess.update_task_desc(self.parent_widget.task_id, task_desc)
            # 3-在会话列表中添加一条用户消息
            self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage,
                                                                    self.text_edit.toPlainText())
            # 显示进度
            self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "已提交任务描述")


