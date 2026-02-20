from PyQt5.QtWidgets import (QCheckBox, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QButtonGroup, QPushButton)
from PyQt5.QtCore import Qt
from qt_ui.toast_widget import ToastWidget
from data_access.task import TaskDataAccess
from agents.utils.views import ConversationMessageType
from ..utility import create_table_widget, add_table_item
from qt_ui.interactive_widget import InteractiveWidget

'''
选择多个解释变量的widget
'''
class ChoseInterpretationVariablesWidget(QWidget):
    def __init__(self, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent

    '''
    初始化一组group checkbox，其中interpretation_vars包含所有的项，categorical_vars指示哪些项初始应该被选中
    '''
    def set_list(self, interpretation_vars:list, categorical_vars:list):
        table_widget = create_table_widget(["操作项", "操作内容"], [])
        options_widget = QWidget()
        options_layout = QHBoxLayout()
        options_layout.setAlignment(Qt.AlignmentFlag.AlignJustify)

        self.checkbox_buttons = []
        self.button_group = QButtonGroup()  # 管理互斥性
        self.button_group.setExclusive(False)
        for idx, text in enumerate(interpretation_vars):
            checkbox = QCheckBox(text)
            options_layout.addWidget(checkbox)
            self.checkbox_buttons.append(checkbox)
            self.button_group.addButton(checkbox, idx)  # 可选：为按钮分配ID
            if text in categorical_vars:
                checkbox.setChecked(True) # 设置为初始选中
        options_widget.setLayout(options_layout)
        # 向表格中添加第一行
        add_table_item(table_widget, ["请更正解释变量的选择", options_widget])
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
        # 确定选择的自变量（解释变量）
        interpretation_vars = []
        for checkbox in self.checkbox_buttons:
            if checkbox.isChecked():
                interpretation_vars.append(checkbox.text())

        if len(interpretation_vars) < 2:
            self.toast = ToastWidget("必须选择至少两项！！！", self.parent())
            self.toast.show_toast()
        else:
            # 1-通知agent从中断处恢复执行
            # todo:按钮自身暂时禁用
            self.parent_widget.finished_event.set_with_data(interpretation_vars)
            # 2-用户消息写入数据库
            user_message = "更正后的解释变量为：" + ",".join(interpretation_vars)
            TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
            # 3-在会话列表中添加一条用户消息
            self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)

            # 显示进度
            # self.parent_widget.parent().parent().show_loading()