from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QLabel, QLineEdit, QMessageBox,QWidget,
                             QDialogButtonBox, QTextEdit, QFileDialog,QStackedWidget,QButtonGroup,
                             QRadioButton,QGroupBox)
from PyQt5.QtGui import QPixmap,QIcon
from PyQt5.QtCore import Qt, QSize
from data_access.task import TaskInfo
from agents.utils.views import PredictionType
import config

class TaskDialog(QDialog):
    def __init__(self, task_info: TaskInfo, parent=None):
        super().__init__(parent)
        self.resize(600, 400)
        self.setWindowTitle("任务信息")
        self.task_info = task_info
        self._init_ui(task_info)

    def _init_ui(self, task_info: TaskInfo):
        """初始化对话框界面"""
        layout = QVBoxLayout()

        # 1-任务名称
        task_name_layout = QHBoxLayout()
        task_name_label = QLabel("请输入任务名称:", self)
        task_name_label.setFixedWidth(150)
        self.task_name_edit = QLineEdit(self)
        if task_info.task_name == '':
            self.task_name_edit.setText('黑河流域_土壤容重') # only for debug
        else:
            self.task_name_edit.setText(task_info.task_name)
        task_name_layout.addWidget(task_name_label, stretch=0)
        task_name_layout.addWidget(self.task_name_edit, stretch=1)

        # 2-任务描述
        task_desc_layout = QHBoxLayout()
        task_desc_label = QLabel("请输入任务描述:", self)
        task_desc_label.setFixedWidth(150)
        self.task_desc_edit = QTextEdit(self)
        self.task_desc_edit.setFixedHeight(100)
        if task_info.description == '':
            self.task_desc_edit.setPlainText('针对土壤容重进行建模，然后进行预测制图。研究区域为高原山地地区，年日照充足，降雨量较少，属于极端干旱或半干旱区，整体自然植被覆盖度低，人类土地利用活动强度大，生态系统脆弱。') # only for debug
        else:
            self.task_desc_edit.setPlainText(task_info.description)
        task_desc_layout.addWidget(task_desc_label, stretch=0)
        task_desc_layout.addWidget(self.task_desc_edit, stretch=1)

        # 3-样点数据
        sample_layout = QHBoxLayout()
        # 3.1 创建图标按钮（替换为实际图标路径）
        btn_browse_sample_file = QPushButton("指定样点数据文件...")  # 浏览模板文件按钮
        btn_browse_sample_file.setIcon(QIcon("./qt_ui/icons/browse_file.png"))
        btn_browse_sample_file.setIconSize(QSize(20, 20))
        btn_browse_sample_file.setToolTip(
            "该文件通常为行/列式的结构化数据文件，表中每一行代表一个样本，每一列代表了样本的特征，即预测变量和所有相关的协变量。");
        btn_browse_sample_file.setFixedWidth(220)
        btn_browse_sample_file.clicked.connect(self.browser_sample_file)
        # 3.2 创建数据源路径显示控件
        self.sample_label = QLabel("", self)
        font = self.sample_label.font()
        font.setBold(True)
        self.sample_label.setFont(font)
        if self.task_info.sample_file == '':
            self.sample_file = config.DEFAULT_SAMPLE_FILE
        else:
            self.sample_file = self.task_info.sample_file
        self.sample_label.setText(self.sample_file)  # 初始化
        sample_layout.addWidget(btn_browse_sample_file, stretch=0)
        sample_layout.addWidget(self.sample_label, stretch=1)

        # 4-制图区域
        cartography_area_layout = QHBoxLayout()
        # 4.1 创建图标按钮（替换为实际图标路径）
        btn_browse_cartography_area_file = QPushButton("指定制图区域文件...")  # 浏览模板文件按钮
        btn_browse_cartography_area_file.setIcon(QIcon("./qt_ui/icons/browse_file.png"))
        btn_browse_cartography_area_file.setIconSize(QSize(20, 20))
        btn_browse_cartography_area_file.setToolTip(
            "该文件通常为GeoTIFF格式的栅格图，将对图中的所有有值像素区域进行预测并赋值。");
        btn_browse_cartography_area_file.setFixedWidth(220)
        btn_browse_cartography_area_file.clicked.connect(self.browser_cartography_area_file)
        # 4.2 创建数据源路径显示控件
        self.cartography_area_label = QLabel("", self)
        font = self.cartography_area_label.font()
        font.setBold(True)
        self.cartography_area_label.setFont(font)
        if self.task_info.mapping_area_file == '':
            self.mapping_area_file = config.DEFAULT_MAPPING_AREA_FILE
        else:
            self.mapping_area_file = self.task_info.mapping_area_file
        self.cartography_area_label.setText(self.mapping_area_file)  # 初始化
        cartography_area_layout.addWidget(btn_browse_cartography_area_file, stretch=0)
        cartography_area_layout.addWidget(self.cartography_area_label, stretch=1)

        # 5-环境协变量
        covariates_path_layout = QHBoxLayout()
        # 5.1 创建图标按钮（替换为实际图标路径）
        btn_browse_covariates_path = QPushButton("指定环境协变量目录...")  # 浏览模板文件按钮
        btn_browse_covariates_path.setIcon(QIcon("./qt_ui/icons/browse_folder.png"))
        btn_browse_covariates_path.setIconSize(QSize(20, 20))
        btn_browse_covariates_path.setToolTip(
            "该目录下放置一系列栅格格式的环境协变量文件，每个文件代表一个协变量的栅格图，栅格图区域范围和制图区域文件需要保持一致。");
        btn_browse_covariates_path.setFixedWidth(220)
        btn_browse_covariates_path.clicked.connect(self.browser_covariates_path)
        # 5.2 创建数据源路径显示控件
        self.covariates_path_label = QLabel("", self)
        font = self.covariates_path_label.font()
        font.setBold(True)
        self.covariates_path_label.setFont(font)
        if self.task_info.covariates_path == '':
            self.covariates_path = config.DEFAULT_COVARIATES_PATH
        else:
            self.covariates_path = self.task_info.covariates_path
        self.covariates_path_label.setText(self.covariates_path)  # 初始化
        covariates_path_layout.addWidget(btn_browse_covariates_path, stretch=0)
        covariates_path_layout.addWidget(self.covariates_path_label, stretch=1)


        # 6-确认/取消按钮组
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal, self
        )

        # 修改信号连接方式
        self.button_box.accepted.connect(self._validate_input)
        self.button_box.rejected.connect(self.reject)

        # 添加组件到布局
        layout.addLayout(task_name_layout)
        layout.addLayout(task_desc_layout)
        layout.addLayout(sample_layout)
        layout.addLayout(cartography_area_layout)
        layout.addLayout(covariates_path_layout)
        layout.addWidget(self.button_box)

        self.setLayout(layout)


    def _validate_input(self):
        """验证输入内容"""
        if self.task_name_edit.text().strip() == "":
            QMessageBox.warning(
                self,
                "输入错误",
                "输入内容不能为空！",
                QMessageBox.StandardButton.Ok
            )
            # 保持对话框打开
            return
        # 输入有效时关闭对话框
        self.accept()

    def get_all_input(self) -> TaskInfo:
        """获取输入文本（需在对话框关闭后调用）"""
        self.task_info.task_name = self.task_name_edit.text().strip()
        self.task_info.description = self.task_desc_edit.toPlainText().strip()
        self.task_info.sample_file = self.sample_file
        self.task_info.mapping_area_file = self.mapping_area_file
        self.task_info.covariates_path = self.covariates_path
        return self.task_info

    '''
    指定用于建模的数据文件
    '''
    def browser_sample_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.parent(),
            "指定样点数据文件...",
            config.DEFAULT_SAMPLE_FILE,
            "表格数据 (*.csv);;shp文件 (*.shp);;gdb文件 (*.gdb)"
        )
        if len(file_paths)>0:
            # 将项添加到列表
            self.sample_label.setText(file_paths[0])
            self.sample_file = file_paths[0]

    '''
    指定制图区域模板文件
    '''

    def browser_cartography_area_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.parent(),
            "指定制图区域数据文件...",
            config.DEFAULT_MAPPING_AREA_FILE,
            "栅格文件 (*.tif);;img文件 (*.img)"
        )
        if len(file_paths) > 0:
            # 将项添加到列表
            self.cartography_area_file_label.setText(file_paths[0])
            self.mapping_area_file = file_paths[0]

    '''
    指定环境协变量文件所在的目录
    '''

    def browser_covariates_path(self):
        dir_path = QFileDialog.getExistingDirectory(
            self.parent(),
            "指定环境协变量栅格文件所在的目录...",
            config.DEFAULT_COVARIATES_PATH,  # 初始目录
            QFileDialog.Option.ShowDirsOnly  # 只显示目录
        )
        if dir_path:
            self.covariates_directory_label.setText(dir_path)
            self.covariates_directory = dir_path

