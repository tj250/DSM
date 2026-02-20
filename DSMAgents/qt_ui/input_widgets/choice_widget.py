from PyQt5.QtWidgets import (QRadioButton, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QButtonGroup, QPushButton)
from PyQt5.QtCore import Qt
from data_access.task import TaskDataAccess
from agents.utils.views import DEUIType, DEUIChoice, ConversationMessageType
from qt_ui.toast_widget import ToastWidget
from ..interactive_widget import InteractiveWidget

'''
选择更换数据源，还是直接指定变量的类型
'''


class ChoiceWidget(QWidget):

    def __init__(self, parent: InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent

    '''
    choice_ui_type:1-指定预测变量，2-修正类别型变量
    '''

    def set_options(self, choice_ui_type: DEUIType):
        self.choice_ui_type = choice_ui_type
        # 布局
        layout = QVBoxLayout()
        # 设置布局间距和边距
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下

        # ---- 1-提示 ----
        task_name_label = QLabel("请做出选择", self)
        layout.addWidget(task_name_label)

        # ---- 2-变量 radio button list ----
        content_layout = QHBoxLayout()
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # 顶部对齐
        self.radio_buttons = []
        self.button_group = QButtonGroup()  # 管理互斥性
        choices = ['重新指定数据源', '直接指定预测变量' if choice_ui_type == DEUIType.Prediction else '修正类别型变量']
        for idx, text in enumerate(choices):
            radio = QRadioButton(text)
            self.radio_buttons.append(radio)
            content_layout.addWidget(radio)
            self.button_group.addButton(radio, idx)  # 可选：为按钮分配ID
        layout.addLayout(content_layout)

        # ---- 3-底部按钮区域 ----
        bottom_btn_layout = QHBoxLayout()
        bottom_btn_layout.setAlignment(Qt.AlignmentFlag.AlignRight)  # 右对齐

        # 创建按钮
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.emit_user_selected_message)

        # 设置按钮固定大小
        for btn in [self.ok_btn]:
            btn.setFixedSize(80, 25)  # 宽80px，高25px
            bottom_btn_layout.addWidget(btn)

        layout.addLayout(bottom_btn_layout)

        self.setLayout(layout)

    '''
    发送用户选择
    '''

    def emit_user_selected_message(self):
        # 确定选择的因变量
        choice = DEUIChoice.NoneChoice
        if self.radio_buttons[0].isChecked():
            choice = DEUIChoice.ChangeDataSource  # 重新指定数据源
        elif self.radio_buttons[1].isChecked():
            choice = DEUIChoice.CorrectPredictionVariable  # 直接指定预测变量

        if choice == DEUIChoice.NoneChoice:
            self.toast = ToastWidget("必须选择一个项！！！", self.parent())
            self.toast.show_toast()
        else:
            # 1-通知agent从中断处恢复执行
            # todo:按钮自身暂时禁用
            self.parent_widget.finished_event.set_with_data(choice)
            # 2-用户消息写入数据库
            user_message = "用户的选择为：{}".format('重新指定数据源' if choice == DEUIChoice.ChangeDataSource else ( \
                '直接指定响应变量' if self.choice_ui_type == DEUIType.Prediction else '修正类别型变量'))
            TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
            # 3-在会话列表中添加一条用户消息
            self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage,
                                                                       user_message)

            # 显示进度
            # self.parent_widget.parent().parent().show_loading()
