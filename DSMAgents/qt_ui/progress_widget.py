from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QApplication, QPushButton, QVBoxLayout
from PyQt5.QtGui import QMovie, QFont
from PyQt5.QtCore import Qt, QSize


class ProgressIndicator(QWidget):
    def __init__(self, animation_path="loading.gif", parent=None):
        super().__init__(parent)

        # 初始化UI
        self.setup_ui(animation_path)

    def setup_ui(self, animation_path):
        # 主布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 左侧动画区域
        self.animation_label = QLabel(self)
        self.animation_label.setFixedSize(QSize(100, 100))
        # self.animation_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        # 加载动画
        self.movie = QMovie(animation_path)
        self.movie.setScaledSize(QSize(100, 100))
        self.animation_label.setMovie(self.movie)
        self.movie.start()

        # 右侧文本区域
        self.text_label = QLabel("等待操作开始...", self)
        # self.text_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        # 设置加粗字体
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.text_label.setFont(font)

        # 将控件加入布局
        layout.addWidget(self.animation_label)
        layout.addWidget(self.text_label)

    def update_progress_text(self, text):
        """ 更新进度文本 """
        self.text_label.setText(text)
        QApplication.processEvents()  # 强制立即更新UI

