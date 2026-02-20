import os.path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QSizePolicy)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QSize
import shutil
import config
from data_access.task import TaskDataAccess
from qt_ui.toast_widget import ToastWidget
from agents.utils.views import ConversationMessageType
from ..basic_widget.countdown_button import CountdownButton
from qt_ui.interactive_widget import InteractiveWidget

'''
浏览建模所需数据源的widget
'''


class BrowseDataWidget(QWidget):

    def __init__(self, data_source: str, parent: InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent

        # 创建图标按钮（替换为实际图标路径）
        self.btn_browser_file = QPushButton("指定用于建模的土壤样点数据文件...")  # 浏览模板文件按钮
        self.btn_browser_file.setIcon(QIcon("./qt_ui/icons/browse_file.png"))
        self.btn_browser_file.setIconSize(QSize(20, 20))
        self.btn_browser_file.setFixedWidth(300)
        self.btn_browser_file.setToolTip(
            "该文件通常为行/列式的结构化数据文件，表中每一行代表一个样本，每一列代表了样本的特征，即预测变量和所有相关的环境协变量。");
        self.btn_browser_file.clicked.connect(self.browse_data_file)

        # 创建数据源路径显示控件
        self.data_source_label = QLabel("", self)
        font = self.data_source_label.font()
        font.setBold(True)
        self.data_source_label.setFont(font)
        if data_source == '':
            data_source = config.DEFAULT_SAMPLE_FILE
        self.data_source_label.setText(data_source)  # 设置默认（原有已经有的）的数据源
        self.selected_file = data_source

        # ---- 左侧指定数据源区域 ----
        data_source_layout = QHBoxLayout()
        data_source_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)  # 垂直居中对齐
        data_source_layout.addSpacing(5)  # 图标间距
        data_source_layout.addWidget(self.btn_browser_file, stretch=0)
        data_source_layout.addWidget(self.data_source_label, stretch=1)

        # 创建数据源路径显示控件
        tip_label = QLabel("""请注意：
                            <ul>
                                <li>用于建模的数据应为<b>行列方式</b>存储的表格式数据，如ESRI shapefile（*.shp）、平面文本格式数据（*.csv）等。</li>
                                <li>数据文件中应包含预测变量以及备选的环境协变量的<b>名称</b>。</li>
                                <li>表格中的所有行列内容不应该有Null值。<br></li>
                            </ul>
                           应在外部数据处理软件仔细对数据文件进行处理以满足上述约定。""", self)
        tip_label.setTextFormat(Qt.TextFormat.RichText)  # 关键：启用富文本模式
        tip_label.setWordWrap(False)
        tip_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,  # 水平策略：自动扩展
            QSizePolicy.Policy.MinimumExpanding  # 垂直策略：根据内容扩展高度
        )

        # 设置对齐方式（左对齐，顶部对齐）
        tip_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # font = tip_label.font()
        # font.setBold(True)
        # tip_label.setFont(font)

        content_layout = QVBoxLayout()  # 左侧内容区总体为垂直向布局
        data_source_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # 靠左，靠顶部对齐
        content_layout.addLayout(data_source_layout)
        content_layout.addWidget(tip_label)

        # 创建右侧的确定按钮
        ok_btn = CountdownButton("确定", 10)
        ok_btn.clicked.connect(self.emit_data_source_confirm_message)
        # self.ok_btn = QPushButton("确定")
        # self.ok_btn.clicked.connect(self.emit_data_source_confirm_message)

        # 总体布局,左侧为浏览数据源按钮，中间为数据列表区域，最右侧为确认完成区域
        layout = QHBoxLayout()
        # 设置布局间距和边距
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下
        layout.addLayout(content_layout, stretch=1)
        layout.addWidget(ok_btn, stretch=0)
        self.setLayout(layout)

    def _load_scaled_icon(self, path: str) -> QPixmap:
        """ 加载并缩放图标至合适大小 """
        pixmap = QPixmap(path)
        return pixmap.scaled(
            20, 20,  # 根据高度20px计算宽度（保持比例）
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

    '''
    清空原有的数据选择
    '''

    def clear(self):
        self.data_source_list_widget.clear()

    '''
    指定待挖掘的数据文件
    '''

    def browse_data_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.parent(),
            "选择一个文件",
            "",
            "表格数据 (*.csv);;shp文件 (*.shp);;gdb文件 (*.gdb)"
        )
        # self.icon_label = QLabel()
        # self.icon_label.setPixmap(QPixmap(icon_path).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio))
        # self.icon_label.setFixedSize(32, 32)  # 固定图标区域大小
        if len(file_paths) > 0:
            # 将项添加到列表
            self.data_source_label.setText("用于土壤属性制图的样点数据文件为：" + file_paths[0])
            self.selected_file = file_paths[0]

    '''
    发送用户输入文本
    '''

    def emit_data_source_confirm_message(self):
        if self.selected_file == '':
            self.toast = ToastWidget("请指定土壤样点数据所在位置！！！", self.parent())
            self.toast.show_toast()
        else:
            if not os.path.exists(self.selected_file):
                self.toast = ToastWidget("指定的土壤样点数据文件不存在！！！", self.parent())
                self.toast.show_toast()
                return
            # 1-通知agent从中断处恢复执行
            self.parent_widget.finished_event.set_with_data(self.selected_file)
            # 2-用户消息写入数据库
            user_message = f"用户选择的土壤样点数据文件为：{self.selected_file}"
            TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
            # 将更新后的数据源写入数据库
            TaskDataAccess.update_task_build_model_data_source(self.parent_widget.task_id, self.selected_file)
            # 3-在会话列表中添加一条用户消息
            self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)

            # 将该文件也复制一份到docker映射的目录下，以便容器可以发现
            shutil.copy(self.selected_file, os.path.join(config.LOCAL_DATA_PATH, os.path.basename(self.selected_file)))

            # 显示进度
            self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "开始数据探索...")
