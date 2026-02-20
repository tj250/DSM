from PyQt5.QtWidgets import (
    QHBoxLayout, QMenu, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QPoint
from data_access.task import TaskInfo,TaskDataAccess

from PyQt5.QtWidgets import QApplication, QListWidget, QListWidgetItem, QLabel, QWidget
from agents.utils.views import ConversationMessageType


class MessageListItemWidget(QWidget):
    """ 自定义列表项控件：左侧图标 + 右侧富文本 """

    def __init__(self, icon_path: str, rich_text: str, parent=None):
        super().__init__(parent)

        # 创建水平布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)  # 调整边距
        layout.setSpacing(10)  # 图标与文本间距

        # 左侧图标控件
        self.icon_label = QLabel()
        self.icon_label.setPixmap(QPixmap(icon_path).scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio))
        self.icon_label.setFixedSize(32, 32)  # 固定图标区域大小
        layout.addWidget(self.icon_label)

        # 右侧富文本控件
        self.text_label = QLabel()
        self.text_label.setTextFormat(Qt.TextFormat.RichText)  # 启用富文本
        self.text_label.setText(rich_text)
        self.text_label.setWordWrap(True)  # 允许换行
        layout.addWidget(self.text_label, stretch=1)  # 拉伸填充剩余空间

        # 设置控件大小策略
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
    def text(self):
        return self.text_label.text()

class ConversationWidget(QListWidget):
    """消息历史列表Widget"""
    # 定义自定义信号(任务选择发生变化)
    # item_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 启用自定义上下文菜单
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        self.parent = parent
        # 添加3个自定义项
        # self.add_custom_item(
        #     icon_path="./qt_ui/icons/usermessage.png",  # 替换为实际图标路径
        #     rich_text='<b style="color: #FF5733;">苹果</b><br>'
        #               '<span style="font-size: 10pt;">红富士苹果，产自山东</span>'
        # )

        #
        # self.add_custom_item(
        #     icon_path="./qt_ui/icons/AIMessage.png",
        #     rich_text='<b style="color: #FFC300;">香蕉</b><br>'
        #               '<span style="font-size: 10pt;">海南香蕉，香甜软糯</span>'
        # )

        # self.add_custom_item(
        #     icon_path="./qt_ui/icons/systemmessage.png",
        #     rich_text='<b style="color: #FF851B;">橙子</b><br>'
        #               '<span style="font-size: 10pt;">赣南脐橙，果粒饱满</span>'
        # )

    '''
    在会话列表中添加一条消息
    '''
    def append_feedback(self, message_type: ConversationMessageType, rich_text: str):
        if message_type == ConversationMessageType.SystemMessage:
            icon_path = "./qt_ui/icons/SystemMessage.png"
        elif message_type == ConversationMessageType.UserMessage:
            icon_path = "./qt_ui/icons/UserMessage.png"
        elif message_type == ConversationMessageType.AgentMessage:
            icon_path = "./qt_ui/icons/AIMessage.png"

        # 创建自定义控件
        widget = MessageListItemWidget(icon_path, rich_text)

        # 创建列表项并设置大小
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())  # 关键：设置项大小匹配控件

        # 将项添加到列表
        self.addItem(item)
        self.setItemWidget(item, widget)  # 关联控件与项

        # 确保会话较多时，显示最底部的会话信息
        self.scrollToBottom()
    '''
    在会话列表中添加任务概要信息
    '''
    def append_task_summary(self, soil_property: str, task_summary: str):
        self.append_feedback(ConversationMessageType.SystemMessage, \
                                                 '最终确定的待制图土壤属性为：<b style=color: "#FFC300;">{}</b> <br> <b style=color: "#FFC300;">{}</b>'.format(
                                                     soil_property, task_summary))
    '''
    更新会话列表的显示
    '''
    def update_conversation_list(self, task_info: TaskInfo):
        self.clear()
        conversations = TaskDataAccess.get_conversation_histories(task_info.task_id)
        if len(conversations) == 0: # 新任务
            TaskDataAccess.add_system_conversation(task_info.task_id, "请描述需要执行的土壤属性制图任务")
            conversations = TaskDataAccess.get_conversation_histories(task_info.task_id)

        for conversation in conversations:
            self.append_feedback(conversation.conversation_type, conversation.content)

        # 检测当前任务是否已经完成了accept task阶段
        # if task_info.stage > 1:
        #     self.append_task_summary(task_info.soil_property, task_info.summary)

        # 确保会话较多时，显示最底部的会话信息
        self.scrollToBottom()

    def _show_context_menu(self, pos: QPoint):
        # 获取点击的项
        item = self.itemAt(pos)
        if not item:
            return  # 空白处不显示菜单

        # 创建菜单
        menu = QMenu(self)
        # assign_action = QAction("复制", self)
        # assign_action.triggered.connect(lambda: self.copy_item_text(item))
        # menu.addAction(assign_action)

        # 显示菜单（全局坐标）
        menu.exec(self.viewport().mapToGlobal(pos))

    def copy_item_text(self, item: QListWidgetItem):
        # 将文本复制到剪贴板
        clipboard = QApplication.clipboard()
        clipboard.setText(self.itemWidget(item).text())