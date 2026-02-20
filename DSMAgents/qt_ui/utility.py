from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QCheckBox,
    QLabel,
    QRadioButton,
    QButtonGroup
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QSize

'''
创建一个统一样式风格的表格
'''


def create_table_widget(headers: list[str], contents: list[list[str]], select_col_name=None, selected_handler=None, select_type='checkbox'):
    # ---- 创建左侧的表格内容显示区域 ----
    table_widget = QTableWidget()
    table_widget.setColumnCount(len(headers) + (0 if select_col_name is None else 1))
    table_widget.setRowCount(len(contents))
    # 设置表头
    if select_col_name is None:
        table_widget.setHorizontalHeaderLabels(headers)
    else:
        table_widget.setHorizontalHeaderLabels(headers + [select_col_name])  # 包含最后一列选择项
        # 创建按钮组确保单选
        button_group = QButtonGroup(table_widget)
        button_group.setExclusive(select_type != 'checkbox')
        button_group.idClicked.connect(selected_handler)

    # 表头样式
    header = table_widget.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    header.setFont(QFont("宋体", 9, QFont.Weight.Bold))
    # 固定宽度（禁止用户调整）
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
    header.resizeSection(0, 300)  # 必须配合使用

    table_widget.verticalHeader().setVisible(False)
    # 填充数据
    for row, items in enumerate(contents):
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            # 设置适当的高度提示，可根据需要调整
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col == 1:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table_widget.setItem(row, col, item)
            table_widget.resizeRowsToContents()
        if select_col_name is not None:  # 需要增加一个选择列（radiobutton类型）
            if select_type == 'checkbox':
                button = QCheckBox()
            else:
                button = QRadioButton()
                button.setAutoExclusive(False)
            button_group.addButton(button, row)  # 将按钮加入组并设置ID
            # 将RadioButton放入容器中居中显示
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.addWidget(button)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)

            table_widget.setCellWidget(row, len(headers), widget)
    # 表格样式
    table_widget.setAlternatingRowColors(True)
    table_widget.setStyleSheet("alternate-background-color: #f0f0f0;")
    return table_widget


'''
向表格中增加一行
'''


def add_table_item(table_widget: QTableWidget, row_content: list):
    row_count = table_widget.rowCount()
    table_widget.insertRow(row_count)
    for col in range(len(row_content)):
        if type(row_content[col]) is str:
            item = QTableWidgetItem(row_content[col])
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col == 1:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table_widget.setItem(row_count, col, item)
        else:
            cell_widget = QWidget()
            cell_layout = QHBoxLayout()
            cell_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 关键：布局居中
            cell_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
            cell_layout.addWidget(row_content[col])
            cell_widget.setLayout(cell_layout)
            table_widget.setCellWidget(row_count, col, cell_widget)
    # item.setSizeHint(QSize(200, 200))
    # table_widget.setRowHeight(0, 200)
