from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel
)
from DSMAlgorithms.base.base_data_structure import algorithms_dict, AlgorithmsType
from PyQt5.QtCore import Qt
from agents.utils.views import BMUIChoice,ConversationMessageType
from data_access.task import TaskDataAccess
from ..basic_widget.countdown_button import CountdownButton
from ..utility import create_table_widget
from ..interactive_widget import InteractiveWidget

# 封装的表格组件
class ModelInfoWidget(QWidget):
    def __init__(self, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent

    def set_confirm_info(self, confirm_info:list):
        # ---- 创建左侧的表格内容显示区域 ----
        data = []
        for data_item in confirm_info:
            if isinstance(data_item.algorithms_type, list):
                algorithms_name = "|".join([algorithms_dict[item] for item in data_item.algorithms_type])
            else:
                if data_item.algorithms_type == AlgorithmsType.CUSTOM:
                    algorithms_name = data_item.algorithms_name  # 直接使用原来的名称
                else:
                    algorithms_name = algorithms_dict[data_item.algorithms_type]  # 转义未中文命名
            data.append([algorithms_name, str(data_item.CV_best_score), str(data_item.R2),str(data_item.RMSE)])
        table_widget = create_table_widget(["算法名称", "交叉验证最佳分数", "R2指标", "RMSE指标"], data)

        # ---- 右侧按钮区域 ----
        button_container = QWidget()
        # button_container.setStyleSheet("background-color: #FFE4B5;")  # 设置容器背景色
        button_layout = QVBoxLayout(button_container)

        algorithms_count_label = QLabel("")
        font = algorithms_count_label.font()
        font.setBold(True)
        algorithms_count_label.setFont(font)
        algorithms_count_label.setText(f"总计算法数量：{len(data)}个")  # 设置默认（原有已经有的）的数据源
        button_layout.addWidget(algorithms_count_label)

        # 创建按钮
        again_btn = QPushButton("优化建模结果（堆叠）")
        again_btn.clicked.connect(self.emit_stacking_message)
        button_layout.addWidget(again_btn)

        ok_btn = CountdownButton("进入预测阶段", 100000)
        ok_btn.clicked.connect(self.emit_enter_next_step_message)
        # ok_btn = QPushButton("进入下一环节")
        # ok_btn.clicked.connect(self.emit_enter_next_step_message)
        button_layout.addWidget(ok_btn)

        # 主布局
        layout = QHBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(table_widget, stretch=1)
        layout.addWidget(button_container, stretch=0)

        self.setLayout(layout)

    '''
    重新指定数据源
    '''
    def emit_stacking_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(BMUIChoice.Stacking)
        # 2-用户消息写入数据库
        user_message = "用户确认需对模型进行堆叠优化"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "正在进行堆叠泛化...")

    '''
    用户确认进入下一个环节
    '''
    def emit_enter_next_step_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(BMUIChoice.EnterNextStep)
        # 2-用户消息写入数据库
        user_message = "用户确认建模结果满足要求，进入下一个环节"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "进入下一环节...")