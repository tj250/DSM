from PyQt5.QtWidgets import QPushButton, QApplication
from PyQt5.QtCore import QTimer, QEvent, QPoint
from PyQt5.QtGui import QMouseEvent
from typing import Callable, Optional

'''
具有倒计时功能的按钮
'''
class CountdownButton(QPushButton):
    def __init__(self,
                 text: str = "确认",
                 countdown_sec: int = 5,
                 on_click: Optional[Callable[[], None]] = None,
                 parent=None):
        """
        参数说明：
        :param text: 按钮显示文本
        :param countdown_sec: 倒计时总时长（秒）
        :param on_click: 点击处理函数（可选）
        :param parent: 父组件
        """
        super().__init__(parent)
        self.original_text = text
        self.countdown_sec = countdown_sec
        self.remaining_sec = countdown_sec
        self.user_clicked = False  # 标记是否用户主动点击

        # 初始化定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_display)

        # 连接点击处理
        if callable(on_click):
            self.clicked.connect(on_click)
        self.clicked.connect(self._handle_click)

        # 设置初始显示
        self._update_display()

    def start_countdown(self):
        """启动倒计时"""
        self.setEnabled(True)
        self.remaining_sec = self.countdown_sec
        self.user_clicked = False
        self.timer.start(1000)
        QApplication.instance().installEventFilter(self)

    def stop_countdown(self):
        """停止倒计时"""
        self.timer.stop()
        self.setText(self.original_text)
        QApplication.instance().removeEventFilter(self)

    def _update_display(self):
        """更新按钮显示"""
        if self.remaining_sec > 0:
            self.setText(f"{self.original_text}（{self.remaining_sec}秒）")
            self.remaining_sec -= 1
        else:
            self.timer.stop()
            if not self.user_clicked:  # 仅自动触发未点击的情况
                self.clicked.emit()
            self.stop_countdown()

    def _handle_click(self):
        """点击按钮后的处理"""
        self.user_clicked = True
        self.stop_countdown()

    def eventFilter(self, obj, event):
        """全局事件过滤器"""
        if event.type() == QEvent.Type.MouseButtonPress:
            # 转换为鼠标事件获取全局坐标
            # 提取原事件属性
            event_type = event.type()  # 事件类型（如 QEvent.Type.MouseButtonPress）
            local_pos = event.pos()  # 鼠标位置（QPointF 类型）
            button = event.button()  # 触发的具体按钮（如 Qt.MouseButton.LeftButton）
            buttons = event.buttons()  # 当前所有按下的按钮组合（如 Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton）
            modifiers = event.modifiers()  # 键盘修饰符（如 Qt.KeyboardModifier.ShiftModifier）
            mouse_event = QMouseEvent(event_type,local_pos,button,buttons,modifiers)

            global_pos = mouse_event.globalPos()

            # 获取按钮的全局几何区域
            btn_geo = self.geometry()
            btn_global_pos = self.mapToGlobal(QPoint(0, 0))
            btn_global_rect = btn_geo.translated(btn_global_pos - self.pos())

            # 判断点击是否在按钮区域外
            if not btn_global_rect.contains(global_pos) and self.isVisible():
                self.stop_countdown()
        return super().eventFilter(obj, event)

    def set_countdown_seconds(self, seconds):
        """动态设置倒计时秒数"""
        self.countdown_sec = seconds
        self.remaining_sec = seconds

    def showEvent(self, event):
        """显示时自动开始倒计时"""
        self.start_countdown()
        super().showEvent(event)

    def hideEvent(self, event):
        """隐藏时停止倒计时"""
        self.stop_countdown()
        super().hideEvent(event)
