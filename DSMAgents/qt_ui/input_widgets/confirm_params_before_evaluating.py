from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QVBoxLayout, QLabel, QWidget,
                             QCheckBox, QFileDialog, QButtonGroup)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize
from data_access.task import TaskDataAccess
from agents.utils.views import UncertaintyType, ConversationMessageType
from qt_ui.toast_widget import ToastWidget
from ..interactive_widget import InteractiveWidget
import config

'''
选择更换数据源，还是直接指定变量的类型
'''


class ConfirmEvaluatingParamsWidget(QWidget):

    def __init__(self, parent: InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent

    def set_options(self):
        # 布局
        layout = QVBoxLayout()
        # 设置布局间距和边距
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下

        # ---- 1-提示 ----
        task_name_label = QLabel("请选择不确定性评估的指标", self)
        layout.addWidget(task_name_label)

        # ---- 2-不确定性指标列表 ----
        metric_layout = QHBoxLayout()
        metric_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # 顶部对齐

        self.checkbox_buttons = []
        self.button_group = QButtonGroup()  # 管理互斥性
        self.button_group.setExclusive(False)
        choices = [UncertaintyType.STD, UncertaintyType.Percentile,UncertaintyType.Confidence95,UncertaintyType.VariationCoefficient]
        for choice in choices:
            checkbox = QCheckBox(choice.value)
            checkbox.setChecked(True)  # 设置为初始选中
            metric_layout.addWidget(checkbox)
            self.checkbox_buttons.append(checkbox)
            self.button_group.addButton(checkbox)

        layout.addLayout(metric_layout)

        dependent_validation_file_layout = QHBoxLayout()
        # 3.1 创建图标按钮（替换为实际图标路径）
        btn_browse_validation_file = QPushButton("指定独立验证集文件...")  # 浏览模板文件按钮
        btn_browse_validation_file.setIcon(QIcon("./qt_ui/icons/browse_file.png"))
        btn_browse_validation_file.setIconSize(QSize(20, 20))
        btn_browse_validation_file.setToolTip(
            "该文件通常为行/列式的结构化数据文件，表中每一行代表一个样本，需要包括用于验证的属性真值列和坐标值列。");
        btn_browse_validation_file.setFixedWidth(220)
        btn_browse_validation_file.clicked.connect(self.browser_validation_file)
        # 3.2 创建数据源路径显示控件
        self.validation_file_label = QLabel("", self)
        font = self.validation_file_label.font()
        font.setBold(True)
        self.validation_file = config.DEFAULT_VALIDATION_FILE
        self.validation_file_label.setFont(font)
        self.validation_file_label.setText(self.validation_file)  # 初始化
        dependent_validation_file_layout.addWidget(btn_browse_validation_file, stretch=0)
        dependent_validation_file_layout.addWidget(self.validation_file_label, stretch=1)
        layout.addLayout(dependent_validation_file_layout)

        # ---- 5-底部按钮区域 ----
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
    指定用于建模的数据文件
    '''

    def browser_validation_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.parent(),
            "指定独立数据集验证文件...",
            config.DEFAULT_VALIDATION_FILE,
            "表格数据 (*.csv);;shp文件 (*.shp);;gdb文件 (*.gdb)"
        )
        if len(file_paths) > 0:
            # 将项添加到列表
            self.validation_file_label.setText(file_paths[0])
            self.validation_file = file_paths[0]

    '''
    发送用户选择
    '''

    def emit_user_selected_message(self):
        # 确定选择的不确定分布指标
        uncertainty_metrics_type = []
        for checkbox in self.checkbox_buttons:
            if checkbox.isChecked():
                uncertainty_metrics_type.append(UncertaintyType(checkbox.text()))

        if len(uncertainty_metrics_type) == 0:
            self.toast = ToastWidget("必须选择至少一种不确定评估指标！！！", self.parent())
            self.toast.show_toast()
        else:
            # 1-通知agent从中断处恢复执行
            # todo:按钮自身暂时禁用
            user_choice = {"uncertainty_metrics_type": uncertainty_metrics_type,
                           'indepent_file': self.validation_file}
            self.parent_widget.finished_event.set_with_data(user_choice)
            # 2-用户消息写入数据库
            user_message = "用户选择的不确定性评估指标为："+','.join([item.value for item in uncertainty_metrics_type])
            TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
            # 3-在会话列表中添加一条用户消息
            self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage,
                                                                       user_message)
            # 显示进度
            self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "开始评估...")

