from PyQt5.QtWidgets import (QRadioButton, QWidget, QVBoxLayout, QLayout, QHBoxLayout, QButtonGroup, QPushButton)
from PyQt5.QtCore import Qt
from qt_ui.toast_widget import ToastWidget
from data_access.task import TaskDataAccess
from agents.utils.views import ConversationMessageType
from ..utility import create_table_widget, add_table_item
from qt_ui.interactive_widget import InteractiveWidget

'''
选择预测变量的widget
'''
class ChosePredictionVariableWidget(QWidget):
    def __init__(self, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent

    def set_list(self, candidate_vars:list, predict_var:str):
        table_widget = create_table_widget(["操作项", "操作内容"], [])
        options_widget = QWidget()
        options_layout = QHBoxLayout()
        options_layout.setAlignment(Qt.AlignmentFlag.AlignJustify)
        options_layout.setSizeConstraint(QLayout.SetMinimumSize)

        self.radio_buttons = []
        self.button_group = QButtonGroup()  # 管理互斥性

        for idx, text in enumerate(candidate_vars):
            radio = QRadioButton(text)
            self.radio_buttons.append(radio)
            options_layout.addWidget(radio)
            self.button_group.addButton(radio, idx)  # 可选：为按钮分配ID
            if text == predict_var:
                radio.setChecked(True)

        options_widget.setLayout(options_layout)
        # 向表格中添加第一行
        add_table_item(table_widget, ["请选择一个变量作为预测变量", options_widget])
        table_widget.resizeRowToContents(0) # 允许任务概述行高度随内容多少自适应
        table_widget.horizontalHeader().setVisible(False)        # 不显示表头

        # ---- 右侧按钮区域 ----
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.emit_user_selected_message)

        # 布局
        layout = QHBoxLayout()
        # 设置布局间距和边距
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(table_widget, stretch=1)
        layout.addWidget(self.ok_btn, stretch=0)

        self.setLayout(layout)

    '''
    发送用户选择
    '''
    def emit_user_selected_message(self):
        # 确定选择的因变量
        depent_var_name = ''
        for radio in self.radio_buttons:
            if radio.isChecked():
                depent_var_name = radio.text()

        if depent_var_name == '':
            self.toast = ToastWidget("必须选择一个变量作为预测变量！！！", self.parent())
            self.toast.show_toast()
        else:
            # 1-通知agent从中断处恢复执行
            # todo:按钮自身暂时禁用
            self.parent_widget.finished_event.set_with_data(depent_var_name)
            # 2-用户消息写入数据库
            user_message = f"选择的预测变量为：{depent_var_name}"
            TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
            # 3-在会话列表中添加一条用户消息
            self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)

            # 显示进度
            # self.parent_widget.parent().parent().show_loading()