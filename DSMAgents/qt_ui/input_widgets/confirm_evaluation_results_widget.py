from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton, QLabel)
from PyQt5.QtCore import Qt
from agents.utils.views import ConversationMessageType, UserConfirmUIChoice
from data_access.task import TaskDataAccess
from ..utility import create_table_widget
from ..basic_widget.countdown_button import CountdownButton
from data_access.task import TaskInfo
from agents.data_structure.base_data_structure import EvaluatingMetrics
from qt_ui.interactive_widget import InteractiveWidget
from DSMAlgorithms import AlgorithmsType
from agents.utils.views import UncertaintyType
from qt_ui.toast_widget import ToastWidget
'''
确认制图结果的widget
'''


class ConfirmEvaluatingResultsWidget(QWidget):

    def __init__(self, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent
        self.evaluating_results = None
        self.uncertainty_metrics_type = None

    '''
    最后一列Button被选中的响应事件
    '''
    def on_model_selected(self, index: int):
        print(f"选中的模型为{self.evaluating_results[index].algorithms_type}")
        if self.evaluating_results[index].PICP is None:
            self.toast = ToastWidget("未能生成该算法的不确定性评估结果，因此无法查看", self.parent())
            self.toast.show_toast()
        else:
            self.parent_widget.ref_feedback_widget.show_uncertainty_mapping_result(self.evaluating_results[index].algorithms_id,
                                                                   self.evaluating_results[index].algorithms_type,
                                                                               self.uncertainty_metrics_type)

    def set_confirm_info(self, uncertainty_metrics_type:list[UncertaintyType], evaluating_results: list[EvaluatingMetrics]):
        self.uncertainty_metrics_type = uncertainty_metrics_type
        self.evaluating_results = evaluating_results
        # 左侧表格区域
        data = []
        for evaluating_metrics in evaluating_results:
            if evaluating_metrics.algorithms_type == AlgorithmsType.STACKING:
                data.append([evaluating_metrics.algorithms_name,
                             '{:.4f}'.format(evaluating_metrics.PICP) if evaluating_metrics.PICP is not None else "计算失败",
                             '{:.4f}'.format(evaluating_metrics.MPIW) if evaluating_metrics.MPIW is not None else "计算失败",
                             '{:.4f}'.format(evaluating_metrics.R2),
                             '{:.4f}'.format(evaluating_metrics.RMSE)])
            else:
                data.append([evaluating_metrics.algorithms_name,
                             '{:.4f}'.format(evaluating_metrics.PICP) if evaluating_metrics.PICP is not None else "计算失败",
                             '{:.4f}'.format(evaluating_metrics.MPIW) if evaluating_metrics.MPIW is not None else "计算失败",
                             '{:.4f}'.format(evaluating_metrics.R2),
                             '{:.4f}'.format(evaluating_metrics.RMSE)])
        self.table_widget = create_table_widget(["算法名称", "PICP(90%)", "MPIW", "独立验证集R2", "独立验证集RMSE"], data,
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
        again_mapping_btn = QPushButton("重新进行制图")
        again_mapping_btn.clicked.connect(self.emit_mapping_again_message)
        button_layout.addWidget(again_mapping_btn)

        again_evaluating_btn = QPushButton("重新进行评估")
        again_evaluating_btn.clicked.connect(self.emit_evaluating_again_message)
        button_layout.addWidget(again_evaluating_btn)

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
    重新评估
    '''
    def emit_evaluating_again_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(UserConfirmUIChoice.EvaluatingAgain)
        # 2-用户消息写入数据库
        user_message = "用户确认需重新进行评估"
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
        user_message = "用户已确认制图结果，结束DSM任务"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "结束DSM任务")