from PyQt5.QtWidgets import QWidget,  QHBoxLayout, QPushButton,QDialog
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from .custom_model_dialog import CustomModelDialog
from DSMAlgorithms import CustomModelData


class CustomModelWidget(QWidget):
    """添加任务按钮（包含图标和文字）"""
    # 定义自定义信号(创建新任务)
    messageSent = pyqtSignal(CustomModelData)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        # 布局
        layout = QHBoxLayout()
        # 关键设置 1：消除布局间距
        layout.setSpacing(0)  # 项之间的间隔设为0
        # 关键设置 2：移除布局边距
        layout.setContentsMargins(0, 0, 0, 0)  # 左、上、右、下边距
        self.setLayout(layout)

        # 按钮
        self.push_button = QPushButton("管理自定义模型")
        self.push_button.setIcon(QIcon("./qt_ui/icons/new_task.png"))
        self.push_button.setIconSize(QSize(20, 20))
        self.push_button.setFixedWidth(150)
        self.push_button.setToolTip("对自定义的模型进行管理");
        self.push_button.clicked.connect(self.manage_custom_models)  # 绑定按钮点击事件
        layout.addWidget(self.push_button, alignment=Qt.AlignmentFlag.AlignCenter)

    def manage_custom_models(self):
        dialog = CustomModelDialog(self.parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 触发信号，创建新任务
            self.messageSent.emit(dialog.get_all_input())




