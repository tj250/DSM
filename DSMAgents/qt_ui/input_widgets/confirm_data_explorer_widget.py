from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt
from data_access.task import TaskDataAccess
from agents.utils.views import DEUIChoice, ConversationMessageType
from ..utility import create_table_widget
from ..basic_widget.countdown_button import CountdownButton
from qt_ui.interactive_widget import InteractiveWidget

'''
文本区域输入widget
'''


class ConfirmDataExplorerWidget(QWidget):

    def __init__(self, parent:InteractiveWidget):
        super().__init__(parent)
        self.parent_widget = parent

    def insert_newline_every_80_chars(self, text):
        """
        在字符串中每隔50个字符后插入换行符
        """
        result = []
        for i, char in enumerate(text):
            if i > 0 and i % 80 == 0:
                result.append('\n')
            result.append(char)
        return ''.join(result)

    def set_confirm_info(self, confirm_info: dict, cate_vars_info: dict, feature_importances: dict):
        # ---- 左侧内容布局区域 ----
        table_content = []
        # 增加类别信息列表
        for key,value in confirm_info.items():
            table_content.append([key,self.insert_newline_every_80_chars(value)])
        table_content.append(["类别型变量的类别详细信息", str(cate_vars_info)])
        table_content.append(["解释变量的特征重要性", self.insert_newline_every_80_chars(str(feature_importances))])
        # 创建显示表格
        table_widget = create_table_widget(["待确认项", "待确认内容"], table_content)
        for i in range(table_widget.rowCount()):
            table_widget.resizeRowToContents(i) # 行高可以根据内容长度增加行高

        # ---- 右侧按钮区域 ----
        button_container = QWidget()
        # button_container.setStyleSheet("background-color: #FFE4B5;")  # 设置容器背景色
        button_layout = QVBoxLayout(button_container)

        # 创建按钮
        again_btn = QPushButton("重新指定数据源")
        again_btn.clicked.connect(self.emit_reset_datasource_message)
        button_layout.addWidget(again_btn)

        correct_prediction_variable_btn = QPushButton("更正预测变量")
        correct_prediction_variable_btn.clicked.connect(self.emit_correct_prediction_variable_message)
        button_layout.addWidget(correct_prediction_variable_btn)

        correct_interpretation_variable_btn = QPushButton("更正解释变量")
        correct_interpretation_variable_btn.clicked.connect(self.emit_correct_interpretation_variable_message)
        button_layout.addWidget(correct_interpretation_variable_btn)

        correct_categorical_variables_btn = QPushButton("更正类别型变量")
        correct_categorical_variables_btn.clicked.connect(self.emit_correct_categorical_variables_message)
        button_layout.addWidget(correct_categorical_variables_btn)

        ok_btn = CountdownButton("探索性空间数据分析", 1000000)
        ok_btn.clicked.connect(self.emit_enter_esda_message)
        button_layout.addWidget(ok_btn)

        # 总体布局
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # 左,上,右,下
        layout.addWidget(table_widget, stretch=1)
        layout.addWidget(button_container, stretch=0)

        self.setLayout(layout)

    '''
    重新指定数据源
    '''

    def emit_reset_datasource_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(DEUIChoice.ChangeDataSource)
        # 2-用户消息写入数据库
        user_message = "用户确认需重新指定数据源"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        # self.parent_widget.parent().parent().show_loading()

    '''
    更正预测变量
    '''

    def emit_correct_prediction_variable_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(DEUIChoice.CorrectPredictionVariable)
        # 2-用户消息写入数据库
        user_message = "用户确认需更正预测变量"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        # self.parent_widget.parent().parent().show_loading()

    '''
    更正解释变量
    '''

    def emit_correct_interpretation_variable_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(DEUIChoice.CorrectInterpretationVariable)
        # 2-用户消息写入数据库
        user_message = "用户确认需更正解释变量"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        # self.parent_widget.parent().parent().show_loading()




    '''
    更正类别型变量
    '''

    def emit_correct_categorical_variables_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(DEUIChoice.CorrectCategoticalVariables)
        # 2-用户消息写入数据库
        user_message = "用户确认需更正类别型变量"
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        # self.parent_widget.parent().parent().show_loading()

    '''
    用户确认进入下一个环节--探索型空间数据分析
    '''

    def emit_enter_esda_message(self):
        # 1-通知agent从中断处恢复执行
        # todo:按钮自身暂时禁用
        self.parent_widget.finished_event.set_with_data(DEUIChoice.EnterEsda)
        # 2-用户消息写入数据库
        user_message = "用户确认数据源满足建模要求，准备执行探索性空间数据分析..."
        TaskDataAccess.add_user_conversation(self.parent_widget.task_id, user_message)
        # 3-在会话列表中添加一条用户消息
        self.parent_widget.ref_conversation_widget.append_feedback(ConversationMessageType.UserMessage, user_message)
        # 显示进度
        self.parent_widget.parent().parent().on_show_progress(self.parent_widget.task_id, "执行探索性空间数据分析...")
