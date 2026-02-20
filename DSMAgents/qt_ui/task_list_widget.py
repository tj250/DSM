from PyQt5.QtWidgets import QWidget, QListWidget,  QHBoxLayout, QLabel, QListWidgetItem,QMenu,QMessageBox,QDialog
from PyQt5.QtCore import Qt, QPoint
from data_access.task import TaskDataAccess,TaskInfo
from .task_info_dialog import TaskDialog


class TaskListItem(QWidget):
    """历史任务列表项控件（包含图像和标签）"""

    def __init__(self, soil_property:str, task_name:str, parent=None):
        super().__init__(parent)
        # 布局
        layout = QHBoxLayout()
        self.setLayout(layout)

        # 图标-指示土壤属性制图任务的类型
        soil_property_label = QLabel(soil_property)
        layout.addWidget(soil_property_label)

        # 标签
        task_name_label = QLabel(task_name)
        layout.addWidget(task_name_label)

class TaskListWidget(QListWidget):
    """任务历史列表Widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        # 启用右键菜单策略
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        all_tasks = TaskDataAccess.get_all_tasks()
        for task in all_tasks:
            self.add_custom_item(task.task_id, task.soil_property, task.task_name)

    '''
    向列表中添加一项，并选中新添加的项
    '''
    def add_custom_item(self, task_id:str, soil_property:str, task_name:str):
        """向列表添加自定义项"""
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, task_id)  # 将 task_id存储到 UserRole
        self.insertItem(0, item)
        # 创建自定义控件并关联到项
        self.setItemWidget(item, QLabel(task_name))  # 关键：将控件绑定到项
        # item.clicked.connect(self.create_task)  # 绑定列表项的点击事件
        self.setCurrentItem(item)

    '''
    已创建新任务的槽函数
    '''
    def on_new_task_created(self, task_info:TaskInfo):
        # 槽函数：处理接收到的消息
        new_task_info = TaskDataAccess.create_task(task_info)
        self.add_custom_item(new_task_info.task_id, new_task_info.soil_property, new_task_info.task_name)

    def show_context_menu(self, pos: QPoint):
        # 获取当前点击的项
        item = self.itemAt(pos)
        if not item:
            return  # 如果点击空白处不显示菜单

        # 创建菜单
        menu = QMenu(self)
        rename_action = menu.addAction("修改")
        delete_action = menu.addAction("删除")

        # 连接菜单动作
        rename_action.triggered.connect(lambda: self.rename_item(item))
        delete_action.triggered.connect(lambda: self.delete_item(item))

        # 显示菜单
        menu.exec(self.mapToGlobal(pos))

    def rename_item(self, item):
        task_info = TaskDataAccess.get_task_info(item.data(Qt.ItemDataRole.UserRole))
        dialog = TaskDialog(task_info, parent=self.parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            TaskDataAccess.update_task(dialog.get_all_input())
            current_item = self.currentItem()
            if current_item:
                current_item.setText(task_info.task_name) # 无法刷新列表，待研究

    '''
    删除某一任务
    '''
    def delete_item(self, item):
        # 确认删除
        reply = QMessageBox.question(
            self, "删除确认", "确定要删除此项任务吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            TaskDataAccess.delete_task(item.data(Qt.ItemDataRole.UserRole))
            self.takeItem(self.row(item))


