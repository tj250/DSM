from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QPushButton, QFileDialog)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize
from DSMAlgorithms.base.base_data_structure import algorithms_dict, AlgorithmsType
import config
from qt_ui.toast_widget import ToastWidget
from agents.utils.views import ConversationMessageType, RegressionPredictionType
from data_access.task import TaskDataAccess
from ..utility import create_table_widget, add_table_item
from ..basic_widget.countdown_button import CountdownButton
from data_access.task import TaskInfo
from agents.data_structure.base_data_structure import RegressionModelInfo
from qt_ui.interactive_widget import InteractiveWidget

'''
在真正开展预测前，请用户对预测相关信息进行确认的widget
'''


class ConfirmInfoBeforeMappingWidget(QWidget):

    def __init__(self, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent
        self.table_widget = None
        self.selected_cartography_area_file = None
        self.cartography_area_file_label = None
        self.selected_covariates_path = None
        self.covariates_path_label = None
        self.model_metrics = None
        self.toast = None

    '''
    最后一列Button被选中的响应事件
    '''

    def on_model_selected(self, index: int):
        print(f"选中的模型为{self.model_metrics[index].algorithms_type}")

    def set_confirm_info(self, task: TaskInfo, model_metrics: list[RegressionModelInfo]):
        self.model_metrics = model_metrics
        # 1、协变量栅格文件的路径指定
        # 1.1 创建图标按钮（替换为实际图标路径）
        btn_browser_covariates_path = QPushButton("环境协变量目录...")  # 浏览模板文件按钮
        btn_browser_covariates_path.setIcon(QIcon("./qt_ui/icons/browse_folder.png"))
        btn_browser_covariates_path.setIconSize(QSize(20, 20))
        btn_browser_covariates_path.setFixedWidth(200)
        btn_browser_covariates_path.setToolTip(
            "目录下放置一系列栅格格式的环境协变量文件，每个文件代表一个协变量的栅格图，栅格图区域范围和制图区域文件需要保持一致。");
        btn_browser_covariates_path.clicked.connect(self.browser_covariates_path)
        # 1.2 创建数据源路径显示控件
        self.covariates_path_label = QLabel("", self)
        font = self.covariates_path_label.font()
        font.setBold(True)
        self.covariates_path_label.setFont(font)
        covariates_path = task.covariates_path  # 从最初的任务设置中获取协变量路径
        if covariates_path == '':  # 用户初始并未指定协变量
            covariates_path = config.DEFAULT_COVARIATES_PATH
        self.covariates_path_label.setText(covariates_path)  # 设置默认（原有已经有的）的数据源
        self.selected_covariates_path = covariates_path
        # 1.3 按钮和label组合为一个水平布局,形成内容区的第一行
        covariates_path_layout = QHBoxLayout()
        covariates_path_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)  # 垂直居中对齐
        covariates_path_layout.addSpacing(5)  # 图标间距
        covariates_path_layout.addWidget(btn_browser_covariates_path, stretch=0)
        covariates_path_layout.addWidget(self.covariates_path_label, stretch=1)

        # 2、制图范围栅格文件的指定
        # 2.1 创建图标按钮（替换为实际图标路径）
        btn_browser_cartography_area_file = QPushButton("制图区域文件...")  # 浏览模板文件按钮
        btn_browser_cartography_area_file.setIcon(QIcon("./qt_ui/icons/browse_file.png"))
        btn_browser_cartography_area_file.setIconSize(QSize(20, 20))
        btn_browser_cartography_area_file.setFixedWidth(200)
        btn_browser_cartography_area_file.setToolTip(
            "该文件通常为GeoTIFF格式的栅格图，将对图中的所有有值像素区域进行预测并赋值。");
        btn_browser_cartography_area_file.clicked.connect(self.browser_cartography_area_file)
        # 2.2 创建数据源路径显示控件
        self.cartography_area_file_label = QLabel("", self)
        font = self.cartography_area_file_label.font()
        font.setBold(True)
        self.cartography_area_file_label.setFont(font)
        mapping_area_file = task.mapping_area_file  # 从最初的任务设置中获取协变量路径
        if mapping_area_file == '':  # 用户初始并未指定协变量
            mapping_area_file = config.DEFAULT_MAPPING_AREA_FILE
        self.cartography_area_file_label.setText(mapping_area_file)  # 设置默认（原有已经有的）的数据源
        self.selected_cartography_area_file = mapping_area_file
        # 2.3 按钮和label组合为一个水平布局,形成内容区的第二行
        cartography_area_layout = QHBoxLayout()
        cartography_area_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)  # 垂直居中对齐
        cartography_area_layout.addSpacing(5)  # 图标间距
        cartography_area_layout.addWidget(btn_browser_cartography_area_file, stretch=0)
        cartography_area_layout.addWidget(self.cartography_area_file_label, stretch=1)

        # 3、选择建模的算法（列出Top K个）
        data = []
        for data_item in model_metrics:
            if isinstance(data_item.algorithms_type, list):
                algorithm_names = []
                for i in range(len(data_item.algorithms_type)):
                    if data_item.algorithms_type[i] == AlgorithmsType.CUSTOM:
                        algorithm_names.append(data_item.algorithms_name)
                    else:
                        algorithm_names.append(algorithms_dict[data_item.algorithms_type[i]])

                data.append(['|'.join(algorithm_names), '{:.4f}'.format(data_item.CV_best_score), '{:.4f}'.format(data_item.R2),
                             '{:.4f}'.format(data_item.RMSE)])
            else:
                if data_item.algorithms_type == AlgorithmsType.CUSTOM:
                    data.append([data_item.algorithms_name, '{:.4f}'.format(data_item.CV_best_score), '{:.4f}'.format(data_item.R2),
                                 '{:.4f}'.format(data_item.RMSE)])
                else:
                    data.append([algorithms_dict[data_item.algorithms_type], '{:.4f}'.format(data_item.CV_best_score), '{:.4f}'.format(data_item.R2),
                                 '{:.4f}'.format(data_item.RMSE)])
        self.table_widget = create_table_widget(["算法名称", "交叉验证最佳分数", "R2指标", "RMSE指标"], data,
                                                select_col_name="选择",
                                                selected_handler = self.on_model_selected)
        widget_row_0_4 = self.table_widget.cellWidget(0, self.table_widget.columnCount() - 1)
        radio = widget_row_0_4.findChild(QCheckBox)
        radio.setChecked(True)

        content_layout = QVBoxLayout()  # 左侧内容区总体为垂直向布局
        covariates_path_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # 靠左，靠顶部对齐
        content_layout.addLayout(covariates_path_layout)
        content_layout.addLayout(cartography_area_layout)
        content_layout.addWidget(self.table_widget)

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

        ok_btn = CountdownButton("确定", 100000)
        ok_btn.clicked.connect(self.emit_confirm_message)
        button_layout.addWidget(ok_btn)

        # 左右两侧的总体布局
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(content_layout, stretch=1)
        layout.addWidget(button_container, stretch=0)

        self.setLayout(layout)

    '''
    指定表示制图范围的栅格数据文件
    '''

    def browser_cartography_area_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.parent(),
            "指定制图区域数据文件...",
            r"E:\data\DSM_Test\template.tif",
            "栅格模板文件 (*.tif);;img文件 (*.img)"
        )
        if len(file_paths) > 0:
            # 将项添加到列表
            self.cartography_area_file_label.setText(f"已选定：{file_paths[0]}")
            self.selected_cartography_area_file = file_paths[0]

    '''
    指定协变量栅格数据文件所在的目录
    '''

    def browser_covariates_path(self):
        dir_path = QFileDialog.getExistingDirectory(
            self.parent(),
            "指定环境协变量栅格文件所在的目录...",
            r"E:\data\DSM_Test\Covariates",  # 初始目录
            QFileDialog.Option.ShowDirsOnly  # 只显示目录
        )
        if dir_path:
            self.covariates_path_label.setText(f"已选定：{dir_path}")
            self.selected_covariates_path = dir_path

    '''
    用户确认开始建模
    '''

    def emit_confirm_message(self):
        # 1-确定是否选择了至少一个预测模型
        selected_algorithms = {}
        for row in range(self.table_widget.rowCount()):
            widget = self.table_widget.cellWidget(row, self.table_widget.columnCount() - 1)
            radio = widget.findChild(QCheckBox)
            if radio.isChecked():
                selected_algorithms[self.model_metrics[row].algorithms_id] = self.model_metrics[row].algorithms_name
        if len(selected_algorithms) == 0:
            self.toast = ToastWidget("必须选择至少一个预测模型！！！", self.parent())
            self.toast.show_toast()
            return

        user_message = f"预测模型为：{",".join([algorithms_name for _, algorithms_name in selected_algorithms.items()])}。"
        # user_message = f"预测模型为：{",".join(['|'.join([algorithms_dict[item] for item in algorithms_type]) if isinstance(algorithms_type, list) else algorithms_dict[algorithms_type] for _, algorithms_type in selected_algorithms.items()])}。"

        # 2-确定选择的数据源的有效性
        if self.selected_cartography_area_file == '':
            self.toast = ToastWidget("请指定制图区域数据文件！！！", self.parent())
            self.toast.show_toast()
            return
        if self.selected_covariates_path == '':
            self.toast = ToastWidget("请指定环境协变量栅格文件所在目录！！！", self.parent())
            self.toast.show_toast()
            return

        # 用户消息写入数据库
        user_message += f"制图区域数据文件为：{self.selected_cartography_area_file}，环境协变量栅格文件所在的目录为：{self.selected_covariates_path}。"

        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(
            (selected_algorithms, self.selected_covariates_path, self.selected_cartography_area_file))
        # 2-用户消息写入数据库
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 将更新后的预测数据源写入数据库
        TaskDataAccess.update_task_prediction_data_setting(self.parent_widget.task_id,
                                                           self.selected_covariates_path,
                                                           self.selected_cartography_area_file)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "开始生成栅格图...")
