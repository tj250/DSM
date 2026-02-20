from PyQt5.QtWidgets import QWidget, QStackedLayout
from PyQt5.QtCore import Qt
from data_access.task import TaskInfo
from agents.data_structure.base_data_structure import MappingMetrics,EvaluatingMetrics
from agents.utils.signals import InputFinishedEvent
from agents.utils.views import DEUIType
from .conversation_widget import ConversationWidget
from qt_ui.feedback_widget import FeedbackWidget
from .progress_widget import ProgressIndicator
from agents.utils.views import UncertaintyType

'''
与用户进行输入交互的widget
'''


class InteractiveWidget(QWidget):
    """ 承载布局的控件，用于固定总高度 """

    def __init__(self, fixed_height: int, conversation_widget: ConversationWidget, feedback_widget: FeedbackWidget):
        super().__init__()
        self.ref_conversation_widget = conversation_widget # 应用到的左上角的会话区域
        self.ref_feedback_widget = feedback_widget # 引用到的右上侧的交互区域
        self.stacked_layout = QStackedLayout()
        self.stacked_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setLayout(self.stacked_layout)
        self.setFixedHeight(fixed_height)  # 设置控件固定高度
        self.progress_indicator_widget = None

    def clear_sub_layout(self, layout):
        """递归清理子布局"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_sub_layout(item.layout())
        self.progress_indicator_widget = None  # 进度控件设置为None

    def clear_layout(self):
        # 1. 清除现有布局及子控件
        old_layout = self.stacked_layout
        if old_layout:
            # 递归删除布局内所有控件和子布局:ml-citation{ref="6,7" data="citationList"}
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()  # 安全删除控件:ml-citation{ref="2,5" data="citationList"}
                elif item.layout():
                    # 递归处理子布局
                    self.clear_sub_layout(item.layout())
                    # 解除原布局与QWidget的关联:ml-citation{ref="4" data="citationList"}
            old_layout.setParent(None)

    '''
    更新进度文本
    '''

    def update_progress_indicator(self, progress_text: str):
        if (self.progress_indicator_widget is None or
                type(self.stacked_layout.currentWidget()) is not ProgressIndicator):
            self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
            self.stacked_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.progress_indicator_widget = ProgressIndicator("./qt_ui/icons/loading.gif")
            self.stacked_layout.addWidget(self.progress_indicator_widget)
        self.progress_indicator_widget.update_progress_text(progress_text)

    '''
    准备输入-清空之前内容，并给出用户输入提示
    '''

    def prepare_user_input_task_desc(self, task_desc: str) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.input_task_desc_widget import TextInputWidget
        text_input_widget = TextInputWidget(task_desc, self)
        self.stacked_layout.addWidget(text_input_widget)  # 0:添加第一个布局对应的控件容器：输入文本

    '''
    准备浏览数据源-清空之前的列表
    '''

    def prepare_browse_data_source(self, data_source: str) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.set_data_source_for_build_model_widget import BrowseDataWidget
        browse_data_widget = BrowseDataWidget(data_source, self)
        self.stacked_layout.addWidget(browse_data_widget)  # 1:添加第二个布局对应的控件容器：浏览数据

    '''
    设置预测变量-清空之前的列表
    '''

    def prepare_select_predict_var(self, candidate_vars: list, predict_var: str) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.chose_prediction_variable_widget import ChosePredictionVariableWidget
        select_item_widget = ChosePredictionVariableWidget(self)
        select_item_widget.set_list(candidate_vars, predict_var)
        self.stacked_layout.addWidget(select_item_widget)

    '''
    设置预测变量-清空之前的列表
    '''

    def prepare_select_interpretation_vars(self, candidate_vars: list, interpretation_vars: list) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.chose_interpretation_variables_widget import ChoseInterpretationVariablesWidget
        select_item_widget = ChoseInterpretationVariablesWidget(self)
        select_item_widget.set_list(candidate_vars, interpretation_vars)
        self.stacked_layout.addWidget(select_item_widget)

    '''
    选择重新指定数据源还是直接指定预测变量
    '''

    def prepare_choice(self, choice_ui_type: DEUIType) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.choice_widget import ChoiceWidget
        choice_widget = ChoiceWidget(self)
        choice_widget.set_options(choice_ui_type)  # 设置就行是指定预测变量，还是修正类别型变量
        self.stacked_layout.addWidget(choice_widget)

    '''
    准备确认任务分析结果还是重新进入新一轮对话，以再次进行任务分析
    '''

    def prepare_confirm_task_analysis_result(self, confirm_info: dict) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.confirm_task_analysis_result_widget import ConfirmTaskAnalysisResultWidget
        confirm_task_analysis_result_widget = ConfirmTaskAnalysisResultWidget(self)
        confirm_task_analysis_result_widget.set_confirm_info(confirm_info)
        self.stacked_layout.addWidget(confirm_task_analysis_result_widget)  # 4:添加第四个布局对应的控件容器:选择某一项

    '''
    请求用户确认数据探索的结果
    '''

    def prepare_confirm_data_explorer(self, confirm_info: dict, cate_vars_info: dict,
                                      feature_importances: dict) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.confirm_data_explorer_widget import ConfirmDataExplorerWidget
        confirm_data_explorer_widget = ConfirmDataExplorerWidget(self)
        confirm_data_explorer_widget.set_confirm_info(confirm_info, cate_vars_info, feature_importances)
        self.stacked_layout.addWidget(confirm_data_explorer_widget)  # 4:添加第四个布局对应的控件容器:选择某一项

    '''
    选择模型堆叠还是进入下一环节（选定预测模型）
    '''

    def prepare_confirm_build_model(self, confirm_info: list) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.confirm_build_model_result_widget import ModelInfoWidget
        confirm_build_model_result_widget = ModelInfoWidget(self)
        confirm_build_model_result_widget.set_confirm_info(confirm_info)
        self.stacked_layout.addWidget(confirm_build_model_result_widget)  # 4:添加第四个布局对应的控件容器:选择某一项

    '''
    设置类别型变量-清空之前的列表
    '''

    def prepare_select_categorical_vars(self, interpretation_vars: list, categorical_vars: list) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.chose_categorical_variable_widget import ChoseCategoricalVariablesWidget
        set_categorical_variables_widget = ChoseCategoricalVariablesWidget(self)
        set_categorical_variables_widget.set_list(interpretation_vars, categorical_vars)
        self.stacked_layout.addWidget(set_categorical_variables_widget)  # 5:添加第四个布局对应的控件容器:选择某一项

    '''
    预测前，请求用户确认预测所需输入信息
    '''

    def prepare_confirm_prediction_input(self, task: TaskInfo, model_metrics: list) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.confirm_info_before_mapping import ConfirmInfoBeforeMappingWidget
        confirm_info_before_prediction_widget = ConfirmInfoBeforeMappingWidget(self)
        confirm_info_before_prediction_widget.set_confirm_info(task, model_metrics)
        self.stacked_layout.addWidget(confirm_info_before_prediction_widget)  # 4:添加第四个布局对应的控件容器:选择某一项

    '''
    请求用户确认制图结果
    '''

    def prepare_confirm_mapping_result(self, task: TaskInfo, mapping_results: list[MappingMetrics]) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.confirm_mapping_results_widget import ConfirmMappingResultsWidget
        confirm_mapping_results_widget = ConfirmMappingResultsWidget(self)
        confirm_mapping_results_widget.set_confirm_info(task, mapping_results)
        self.stacked_layout.addWidget(confirm_mapping_results_widget)

    '''
    评估前，请求用户确认评估所需参数信息
    '''

    def prepare_confirm_evaluating_params(self) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.confirm_params_before_evaluating import ConfirmEvaluatingParamsWidget
        confirm_params_before_evaluation_widget = ConfirmEvaluatingParamsWidget(self)
        confirm_params_before_evaluation_widget.set_options()
        self.stacked_layout.addWidget(confirm_params_before_evaluation_widget)  # 4:添加第四个布局对应的控件容器:选择某一项


    '''
    请求用户确认评估结果
    '''

    def prepare_confirm_evaluating_result(self, uncertainty_metrics_type: list[UncertaintyType], evaluating_results: list[EvaluatingMetrics]) -> None:
        self.clear_sub_layout(self.stacked_layout)  # 清理现有布局
        from .input_widgets.confirm_evaluation_results_widget import ConfirmEvaluatingResultsWidget
        confirm_evaluating_results_widget = ConfirmEvaluatingResultsWidget(self)
        confirm_evaluating_results_widget.set_confirm_info(uncertainty_metrics_type, evaluating_results)
        self.stacked_layout.addWidget(confirm_evaluating_results_widget)


    '''
    记录相关联的对象，这些对象需要被输入控件的处理事件中使用，具体包括：
    1、当前正在处理的任务ID
    2、输入完毕向agent发出通知的事件对象，该事件对象中包含了用户输入的内容，用于传给agent
    '''

    def set_related_objects(self, task_id, event: InputFinishedEvent):
        self.task_id = task_id
        self.finished_event = event
