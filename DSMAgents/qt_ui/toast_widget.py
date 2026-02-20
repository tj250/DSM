from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QPropertyAnimation, QTimer, QPoint
from PyQt5.QtGui import QGuiApplication


class ToastWidget(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        # 窗口属性设置
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # 样式配置
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(50, 50, 50, 200);
                color: red;
                border-radius: 8px;
                padding: 12px 20px;
                font: bold 14px "微软雅黑";
                min-width: 120px;
            }
        """)

        # 内容设置
        self.setText(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.adjustSize()  # 自动调整大小

        # 初始化透明度
        self.setWindowOpacity(0.0)

        # 动画系统
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(500)  # 单程动画时间

    def show_toast(self, display_time=1000):
        """显示渐入动画并定时关闭"""
        self.update_position()
        self.show()

        # 渐入动画
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.start()

        # 定时触发渐出
        QTimer.singleShot(display_time, self.fade_out)

    def fade_out(self):
        """执行渐出动画"""
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.finished.connect(self.close)
        self.animation.start()

    def update_position(self):
        """根据父窗口位置居中显示"""
        if not self.parent():
            return

        # 获取父窗口中心点（屏幕坐标）
        parent_center = self.parent().geometry().center()
        # 计算提示框左上角坐标
        toast_pos = parent_center - QPoint(self.width() // 2, self.height() // 2)

        # 防止超出屏幕
        screen_geo = QGuiApplication.primaryScreen().availableGeometry()
        toast_pos.setX(max(screen_geo.left(), min(toast_pos.x(), screen_geo.right() - self.width())))
        toast_pos.setY(max(screen_geo.top(), min(toast_pos.y(), screen_geo.bottom() - self.height())))

        self.move(toast_pos)