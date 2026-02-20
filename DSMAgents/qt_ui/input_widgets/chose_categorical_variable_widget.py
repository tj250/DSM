from PyQt5.QtWidgets import (QCheckBox, QWidget, QLabel, QScrollArea, QSizePolicy, QHBoxLayout, QButtonGroup,
                             QPushButton, QVBoxLayout, QGridLayout)
from PyQt5.QtCore import Qt
from qt_ui.toast_widget import ToastWidget
from data_access.task import TaskDataAccess
from agents.utils.views import ConversationMessageType
from ..utility import create_table_widget, add_table_item
from qt_ui.interactive_widget import InteractiveWidget

'''
选择多个类别型变量的widget
'''
class ChoseCategoricalVariablesWidget(QWidget):

    def __init__(self, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent

    '''
    初始化一组group checkbox，其中interpretation_vars包含所有的项，categorical_vars指示哪些项初始应该被选中
    '''
    def set_list(self, interpretation_vars:list, categorical_vars:list):
        self.setFixedWidth(1500)
        # 创建提示控件
        tip_label = QLabel("请更正类别型变量的选择", self)
        font = tip_label.font()
        font.setBold(True)
        tip_label.setFont(font)

        # 创建滚动区域以容纳大量单选按钮
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 创建内容部件
        content_widget = QWidget()
        # 设置布局属性以实现高度自适应
        content_layout = QGridLayout(content_widget)
        content_layout.setSpacing(8)
        content_layout.setContentsMargins(15, 10, 15, 15)

        self.button_group = QButtonGroup()  # 管理互斥性
        self.button_group.setExclusive(False)
        self.checkbox_buttons = []
        # 计算每行可容纳的按钮数量（根据按钮宽度和间距）
        button_width = 80  # 预估每个按钮的宽度
        available_width = 470  # 可用宽度（500-边距）
        buttons_per_row = max(1, available_width // (button_width + 8))
        for idx, text in enumerate(interpretation_vars):
            checkbox = QCheckBox(text)
            checkbox.setFixedHeight(30)  # 设置固定高度确保一致性
            row = idx // buttons_per_row
            col = idx % buttons_per_row
            content_layout.addWidget(checkbox, row, col)
            self.checkbox_buttons.append(checkbox)
            self.button_group.addButton(checkbox, idx)  # 可选：为按钮分配ID
            if text in categorical_vars:
                checkbox.setChecked(True) # 设置为初始选中

        # 设置内容部件的大小策略以实现高度自适应
        content_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        # 将内容部件设置为滚动区域的子部件
        scroll_area.setWidget(content_widget)

        content_layout = QVBoxLayout()  # 左侧内容区总体为垂直向布局
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # 靠左，靠顶部对齐
        content_layout.addWidget(tip_label)
        content_layout.addWidget(scroll_area)

        # ---- 右侧按钮区域 ----
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.emit_user_selected_message)

        # 布局
        layout = QHBoxLayout()
        # 设置布局间距和边距
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(content_layout, stretch=1)
        layout.addWidget(self.ok_btn, stretch=0)
        self.setLayout(layout)

    '''
    发送用户选择
    '''
    def emit_user_selected_message(self):
        # 确定选择的因变量
        categorical_vars = []
        for checkbox in self.checkbox_buttons:
            if checkbox.isChecked():
                categorical_vars.append(checkbox.text())

        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(categorical_vars)
        # 2-用户消息写入数据库
        user_message = "更正后的类别型变量为：" + ",".join(categorical_vars)
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)

        # 显示进度
        # self.parent_widget.parent().parent().show_loading()