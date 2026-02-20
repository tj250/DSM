from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, QRadioButton, QStackedWidget,
                             QCheckBox, QButtonGroup, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt, QSize
from agents.utils.views import ConversationMessageType, UserConfirmUIChoice
from data_access.task import TaskDataAccess
from ..utility import create_table_widget, add_table_item
from ..basic_widget.countdown_button import CountdownButton
from data_access.task import TaskInfo
from agents.data_structure.base_data_structure import MappingMetrics
from qt_ui.interactive_widget import InteractiveWidget
from DSMAlgorithms import algorithms_dict,AlgorithmsType

'''
比较制图结果的widget(制图结果验证环节)
'''


class ConfirmMappingResultsWidget(QWidget):

    def __init__(self, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent
        self.algorithms_metrics = None

    '''
    最后一列Button被选中的响应事件
    '''
    def on_model_selected(self, index: int):
        print(f"选中的模型为{self.algorithms_metrics[index].algorithms_type}")
        self.parent_widget.ref_feedback_widget.show_mapping_result(self.algorithms_metrics[index].algorithms_id,
                                                                   self.algorithms_metrics[index].algorithms_type)

    def set_confirm_info(self, task:TaskInfo, algorithms_metrics: list[dict]):
        self.algorithms_metrics = algorithms_metrics
        # 左侧表格区域
        data = []
        for mapping_metric in algorithms_metrics:
            if mapping_metric.algorithms_type == AlgorithmsType.STACKING:
                # algorithm_names = [item for item in mapping_metric.algorithms_name]
                data.append([mapping_metric.algorithms_name,
                             '{:.4f}'.format(mapping_metric.min_value),
                             '{:.4f}'.format(mapping_metric.max_value),
                             '{:.4f}'.format(mapping_metric.mean_value),
                             '{:.4f}'.format(mapping_metric.CV_best_score),
                             '{:.4f}'.format(mapping_metric.R2),
                             '{:.4f}'.format(mapping_metric.RMSE)])
            else:
                data.append([mapping_metric.algorithms_name,
                             '{:.4f}'.format(mapping_metric.min_value),
                             '{:.4f}'.format(mapping_metric.max_value),
                             '{:.4f}'.format(mapping_metric.mean_value),
                             '{:.4f}'.format(mapping_metric.CV_best_score),
                             '{:.4f}'.format(mapping_metric.R2),
                             '{:.4f}'.format(mapping_metric.RMSE)])
        self.table_widget = create_table_widget(["算法名称", "像元最小值", "像元最大值", "像元均值", '建模时交叉验证分数',"R2", "RMSE"], data,
                                                select_col_name="选择",
                                                selected_handler=self.on_model_selected,
                                                select_type="radiobutton")
        widget_row_0_4 = self.table_widget.cellWidget(0, self.table_widget.columnCount() - 1)
        radio = widget_row_0_4.findChild(QRadioButton)
        radio.setChecked(True)

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
        again_btn = QPushButton("重新进行制图")
        again_btn.clicked.connect(self.emit_mapping_again_message)
        button_layout.addWidget(again_btn)

        ok_btn = CountdownButton("确定", 1000000)
        ok_btn.clicked.connect(self.emit_confirm_message)
        button_layout.addWidget(ok_btn)

        # 左右两侧的总体布局
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.table_widget, stretch=1)
        layout.addWidget(button_container, stretch=0)

        self.setLayout(layout)

    '''
    重新制图
    '''
    def emit_mapping_again_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(UserConfirmUIChoice.MappingAgain)
        # 2-用户消息写入数据库
        user_message = "用户确认需重新进行制图"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        # self.parent_widget.parent().parent().show_loading()

    '''
    用户确认进入下一个环节
    '''
    def emit_confirm_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(UserConfirmUIChoice.Confirm)
        # 2-用户消息写入数据库
        user_message = "用户已确认制图结果，进入下一个环节"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "进入下一环节...")