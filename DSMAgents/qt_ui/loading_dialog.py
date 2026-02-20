import sys
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QDialog, QVBoxLayout, QLabel)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QMovie


# 模态加载对话框
class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("处理中...")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)  # 应用级模态
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.FramelessWindowHint)
        # 设置无边框透明背景
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # 设置GIF动画
        target_size = QSize(200, 200)
        self.movie = QMovie('./qt_ui/icons/loading.gif')
        self.movie.setScaledSize(target_size)
        self.label = QLabel(self)
        self.label.setMovie(self.movie)

        # 窗口布局
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # 固定窗口大小为GIF尺寸
        self.setFixedSize(target_size)

    def showEvent(self, event):
        self.movie.start()
        super().showEvent(event)

    def closeEvent(self, event):
        self.movie.stop()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            event.ignore()  # 屏蔽ESC键‌:ml-citation{ref="1,5" data="citationList"}
        elif event.modifiers() == Qt.KeyboardModifier.AltModifier and event.key() == 16777251: # 不起作用
            event.ignore()  # 屏蔽Alt+F4‌:ml-citation{ref="2,4" data="citationList"}
        else:
            super().keyPressEvent(event)