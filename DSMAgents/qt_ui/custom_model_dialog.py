from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QMessageBox,QComboBox,QPushButton,
                             QDialogButtonBox, QTextEdit,QButtonGroup,QRadioButton, QCheckBox)
from PyQt5.QtCore import Qt
from data_access.build_model import BuildModelDataAccess
from DSMAlgorithms import CustomModelData
import uuid
import json

DEFAULT_MODEL_NAME = "新模型"
class CustomModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(1200, 800)
        self.setWindowTitle("自定义基础模型")
        self._init_ui()

    def _init_ui(self):
        """初始化对话框界面"""
        self.layout = QVBoxLayout()
        # 创建一个按钮
        self.add_button = QPushButton("添加新模型", self)
        # 连接点击事件
        self.add_button.clicked.connect(self.on_add_button_clicked)
        # 添加到布局中（例如添加到选项布局中）
        self.layout.addWidget(self.add_button)

        self.models = BuildModelDataAccess.get_extend_models()
        # 创建一个pyqt5的下拉列表，列表中添加项
        custom_model_selection_layout = QHBoxLayout()
        custom_model_selection_label = QLabel("选择自定义模型:", self)
        custom_model_selection_label.setFixedWidth(300)
        self.model_type_combo = QComboBox(self)
        for model in self.models:
            self.model_type_combo.addItem(model.custom_model_name)
        custom_model_selection_layout.addWidget(custom_model_selection_label, stretch=0)
        custom_model_selection_layout.addWidget(self.model_type_combo, stretch=1)
        self.layout.addLayout(custom_model_selection_layout)

        # 连接选择事件
        self.model_type_combo.currentIndexChanged.connect(self.on_combo_selection_changed)

        if len(self.models) > 0:
            self._init_custom_model(self.models[0])

    def on_add_button_clicked(self):
        """处理按钮点击事件"""
        # 在这里添加你的业务逻辑
        new_custom_model = CustomModelData()
        new_custom_model.custom_model_id = str(uuid.uuid1())
        new_custom_model.custom_model_name = DEFAULT_MODEL_NAME
        new_custom_model.class_code = "在此输入代码"
        new_custom_model.data_transform = 0
        BuildModelDataAccess.create_custom_model(new_custom_model)
        self.models.insert(0, new_custom_model)
        # 列表中添加一个新项
        self.model_type_combo.insertItem(0, new_custom_model.custom_model_name)
        self.model_type_combo.setCurrentIndex(0)


    def on_combo_selection_changed(self, index):
        """处理下拉列表索引变化事件"""
        # selected_text = self.model_type_combo.itemText(index)
        # print(f"选择了索引 {index}: {selected_text}")
        custom_model_info = self.models[index]
        self.model_name_edit.setText(custom_model_info.custom_model_name)
        self.model_desc_edit.setPlainText(custom_model_info.description)
        self.class_code_edit.setPlainText(custom_model_info.class_code)
        self.fix_params_edit.setPlainText(json.dumps(custom_model_info.special_args))
        if custom_model_info.special_enum_conversion_args is None:
            self.special_enum_conversion_args_edit.setText("")
        else:
            self.special_enum_conversion_args_edit.setText(json.dumps(custom_model_info.special_enum_conversion_args))
        self.dyn_eval_args_edit.setText(custom_model_info.dyn_eval_args)
        if custom_model_info.enum_conversion_args is None:
            self.enum_conversion_args_edit.setText("")
        else:
            self.enum_conversion_args_edit.setText(json.dumps(custom_model_info.enum_conversion_args))
        if custom_model_info.complex_lamda_args is None:
            self.complex_lamda_args_edit.setText("")
        else:
            self.complex_lamda_args_edit.setText(json.dumps(custom_model_info.complex_lamda_args))
        self.radio_buttons[custom_model_info.data_transform].setChecked(True)
        if custom_model_info.can_stacking is not None:
            self.can_stacking_option.setChecked(custom_model_info.can_stacking)
        if custom_model_info.X_with_geometry is not None:
            self.x_with_geometry_option.setChecked(custom_model_info.X_with_geometry)
        if custom_model_info.can_deal_unknown_distribution is not None:
            self.can_deal_unknown_distribution_option.setChecked(custom_model_info.can_deal_unknown_distribution)
        if custom_model_info.can_deal_high_dims is not None:
            self.can_deal_high_dims_option.setChecked(custom_model_info.can_deal_high_dims)
        if custom_model_info.can_deal_small_samples is not None:
            self.can_deal_small_samples_option.setChecked(custom_model_info.can_deal_small_samples)
        if custom_model_info.can_deal_multicollinearity is not None:
            self.can_deal_multicollinearity_option.setChecked(custom_model_info.can_deal_multicollinearity)
        if custom_model_info.can_deal_heterogeneity is not None:
            self.can_deal_heterogeneity_option.setChecked(custom_model_info.can_deal_heterogeneity)

    '''
    以某个模型信息初始化界面
    '''
    def _init_custom_model(self, custom_model_info: CustomModelData):
        # 1-模型名称
        model_name_layout = QHBoxLayout()
        model_name_label = QLabel("模型名称:", self)
        model_name_label.setFixedWidth(300)
        self.model_name_edit = QLineEdit(self)
        if custom_model_info.custom_model_name == '':
            self.model_name_edit.setText('支持向量回归(RBF核)') # only for debug
        else:
            self.model_name_edit.setText(custom_model_info.custom_model_name)
        model_name_layout.addWidget(model_name_label, stretch=0)
        model_name_layout.addWidget(self.model_name_edit, stretch=1)

        # 2-模型描述
        model_desc_layout = QHBoxLayout()
        model_desc_label = QLabel("模型描述:", self)
        model_desc_label.setFixedWidth(300)
        self.model_desc_edit = QTextEdit(self)
        self.model_desc_edit.setFixedHeight(100)
        if custom_model_info.description == '':
            self.model_desc_edit.setPlainText('基于RBF核的支持向量回归') # only for debug
        else:
            self.model_desc_edit.setPlainText(custom_model_info.description)
        model_desc_layout.addWidget(model_desc_label, stretch=0)
        model_desc_layout.addWidget(self.model_desc_edit, stretch=1)

        # 3-类代码
        class_code_layout = QHBoxLayout()
        class_code_label = QLabel("符合sklearn风格的算法的类代码:", self)
        class_code_label.setFixedWidth(300)
        self.class_code_edit = QTextEdit(self)
        self.class_code_edit.setFixedHeight(100)
        if custom_model_info.class_code is None:
            self.class_code_edit.setPlainText('') # only for debug
        else:
            self.class_code_edit.setPlainText(custom_model_info.class_code)
        class_code_layout.addWidget(class_code_label, stretch=0)
        class_code_layout.addWidget(self.class_code_edit, stretch=1)

        # 4-特殊参数列表
        fix_params_layout = QHBoxLayout()
        fix_params_label = QLabel("特殊参数:", self)
        fix_params_label.setFixedWidth(300)
        self.fix_params_edit = QTextEdit(self)
        self.fix_params_edit.setFixedHeight(100)
        if custom_model_info.class_code is None:
            self.fix_params_edit.setPlainText('') # only for debug
        else:
            self.fix_params_edit.setPlainText(json.dumps(custom_model_info.special_args))
        fix_params_layout.addWidget(fix_params_label, stretch=0)
        fix_params_layout.addWidget(self.fix_params_edit, stretch=1)

        # 5-特殊参数中需枚举转换的参数列表
        special_enum_conversion_args_layout = QHBoxLayout()
        special_enum_conversion_args_label = QLabel("特殊参数中需枚举转换的参数信息:", self)
        special_enum_conversion_args_label.setFixedWidth(300)
        self.special_enum_conversion_args_edit = QLineEdit(self)
        if custom_model_info.special_enum_conversion_args is None:
            self.special_enum_conversion_args_edit.setText('') # only for debug
        else:
            self.special_enum_conversion_args_edit.setText(json.dumps(custom_model_info.special_enum_conversion_args))
        special_enum_conversion_args_layout.addWidget(special_enum_conversion_args_label, stretch=0)
        special_enum_conversion_args_layout.addWidget(self.special_enum_conversion_args_edit, stretch=1)

        # 6-特殊参数中需动态执行的参数列表
        dyn_eval_args_layout = QHBoxLayout()
        dyn_eval_args_label = QLabel("特殊参数中需动态eval的参数名:", self)
        dyn_eval_args_label.setFixedWidth(300)
        self.dyn_eval_args_edit = QLineEdit(self)
        if custom_model_info.dyn_eval_args is None:
            self.dyn_eval_args_edit.setText('') # only for debug
        else:
            self.dyn_eval_args_edit.setText(custom_model_info.dyn_eval_args)
        dyn_eval_args_layout.addWidget(dyn_eval_args_label, stretch=0)
        dyn_eval_args_layout.addWidget(self.dyn_eval_args_edit, stretch=1)

        # 7-可调参数中需枚举转换的参数列表
        enum_conversion_args_layout = QHBoxLayout()
        enum_conversion_args_label = QLabel("可调参数中需枚举转换的参数信息:", self)
        enum_conversion_args_label.setFixedWidth(300)
        self.enum_conversion_args_edit = QLineEdit(self)
        if custom_model_info.enum_conversion_args is None:
            self.enum_conversion_args_edit.setText('') # only for debug
        else:
            self.enum_conversion_args_edit.setText(json.dumps(custom_model_info.enum_conversion_args))
        enum_conversion_args_layout.addWidget(enum_conversion_args_label, stretch=0)
        enum_conversion_args_layout.addWidget(self.enum_conversion_args_edit, stretch=1)

        # 8-需通过lamda表达式处理的参数列表
        complex_lamda_args_layout = QHBoxLayout()
        complex_lamda_args_label = QLabel("可调参数中需通过lambda表达式处理的参数信息:", self)
        complex_lamda_args_label.setFixedWidth(300)
        self.complex_lamda_args_edit = QLineEdit(self)
        if custom_model_info.enum_conversion_args is None:
            self.complex_lamda_args_edit.setText('') # only for debug
        else:
            self.complex_lamda_args_edit.setText(json.dumps(custom_model_info.complex_lamda_args))
        complex_lamda_args_layout.addWidget(complex_lamda_args_label, stretch=0)
        complex_lamda_args_layout.addWidget(self.complex_lamda_args_edit, stretch=1)


        # 9-数据变换类型
        data_transform_layout = QHBoxLayout()
        data_transform_label = QLabel("算法的数据变换要求:", self)
        data_transform_label.setFixedWidth(300)
        data_transform_layout.addWidget(data_transform_label, stretch=0)
        self.radio_buttons = []
        self.button_group = QButtonGroup()  # 管理互斥性
        choices = ['不变换', 'Z-Score标准化', 'Min-Max归一化']
        for idx, text in enumerate(choices):
            radio = QRadioButton(text)
            self.radio_buttons.append(radio)
            data_transform_layout.addWidget(radio)
            self.button_group.addButton(radio, idx)  # 可选：为按钮分配ID
            if custom_model_info.data_transform == idx:
                radio.setChecked(True)

        # 10-其余各个选项
        options_layout = QHBoxLayout()

        self.can_stacking_option = QCheckBox('可用于堆叠')
        self.can_stacking_option.setChecked(custom_model_info.can_stacking == True)
        options_layout.addWidget(self.can_stacking_option)

        self.x_with_geometry_option = QCheckBox('输入数据需要带有空间坐标')
        self.x_with_geometry_option.setChecked(custom_model_info.X_with_geometry == True)
        options_layout.addWidget(self.x_with_geometry_option)

        self.can_deal_high_dims_option = QCheckBox('可处理高维数据')
        self.can_deal_high_dims_option.setChecked(custom_model_info.can_deal_high_dims == True)
        options_layout.addWidget(self.can_deal_high_dims_option)

        self.can_deal_small_samples_option = QCheckBox('可处理小样本数据')
        self.can_deal_small_samples_option.setChecked(custom_model_info.can_deal_small_samples == True)
        options_layout.addWidget(self.can_deal_small_samples_option)

        self.can_deal_unknown_distribution_option = QCheckBox('可处理未知分布的数据')
        self.can_deal_unknown_distribution_option.setChecked(custom_model_info.can_deal_unknown_distribution == True)
        options_layout.addWidget(self.can_deal_unknown_distribution_option)

        self.can_deal_multicollinearity_option = QCheckBox('可处理具有多重共线性的数据')
        self.can_deal_multicollinearity_option.setChecked(custom_model_info.can_deal_multicollinearity == True)
        options_layout.addWidget(self.can_deal_multicollinearity_option)

        self.can_deal_heterogeneity_option = QCheckBox('能够针对空间异质性建模')
        self.can_deal_heterogeneity_option.setChecked(custom_model_info.can_deal_heterogeneity == True)
        options_layout.addWidget(self.can_deal_heterogeneity_option)

        # 6-确认/取消按钮组
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal, self
        )

        # 修改信号连接方式
        self.button_box.accepted.connect(self._validate_input)
        self.button_box.rejected.connect(self.reject)

        # 添加组件到布局
        self.layout.addLayout(model_name_layout)
        self.layout.addLayout(model_desc_layout)
        self.layout.addLayout(class_code_layout)
        self.layout.addLayout(fix_params_layout)
        self.layout.addLayout(special_enum_conversion_args_layout)
        self.layout.addLayout(dyn_eval_args_layout)
        self.layout.addLayout(enum_conversion_args_layout)
        self.layout.addLayout(complex_lamda_args_layout)
        self.layout.addLayout(data_transform_layout)
        self.layout.addLayout(options_layout)
        self.layout.addWidget(self.button_box)

        self.setLayout(self.layout)



    def _validate_input(self):
        """验证输入内容"""
        if self.model_name_edit.text().strip() == "":
            QMessageBox.warning(
                self,
                "模型名称输入错误",
                "输入内容不能为空！",
                QMessageBox.StandardButton.Ok
            )
            # 保持对话框打开
            return
        if self.model_name_edit.text().strip() == DEFAULT_MODEL_NAME:
            QMessageBox.warning(
                self,
                "模型名称输入错误",
                "模型名称不能使用默认的名称！",
                QMessageBox.StandardButton.Ok
            )
            # 保持对话框打开
            return
        model_name_occur_times = 0
        for model_info in self.models:
            if model_info.custom_model_name == self.model_name_edit.text().strip():
                model_name_occur_times += 1
        if model_name_occur_times > 1:
            QMessageBox.warning(
                self,
                "模型名称输入错误",
                "模型名称不能与其它模型重复！",
                QMessageBox.StandardButton.Ok
            )
            # 保持对话框打开
            return
        if self.class_code_edit.toPlainText().strip() == "":
            QMessageBox.warning(
                self,
                "类代码片段输入错误",
                "输入内容不能为空！",
                QMessageBox.StandardButton.Ok
            )
            # 保持对话框打开
            return


        # 输入有效时关闭对话框
        self.accept()

    def get_all_input(self) -> CustomModelData:
        """获取输入文本（需在对话框关闭后调用）"""
        custom_model = CustomModelData()
        custom_model.custom_model_id = self.models[self.model_type_combo.currentIndex()].custom_model_id
        custom_model.custom_model_name = self.model_name_edit.text().strip()
        custom_model.description = self.model_desc_edit.toPlainText().strip()
        custom_model.class_code = self.class_code_edit.toPlainText().strip()
        custom_model.special_args = json.loads(self.fix_params_edit.toPlainText().strip())
        if self.special_enum_conversion_args_edit.text().strip() == '':
            custom_model.special_enum_conversion_args = None
        else:
            custom_model.special_enum_conversion_args = json.loads(self.special_enum_conversion_args_edit.text().strip())
        custom_model.dyn_eval_args = self.dyn_eval_args_edit.text().strip()
        if self.enum_conversion_args_edit.text().strip() == '':
            custom_model.enum_conversion_args = None
        else:
            custom_model.enum_conversion_args = json.loads(self.enum_conversion_args_edit.text().strip())
        if self.complex_lamda_args_edit.text().strip() == '':
            custom_model.complex_lamda_args = None
        else:
            custom_model.complex_lamda_args = json.loads(self.complex_lamda_args_edit.text().strip())
        if self.radio_buttons[0].isChecked():
            custom_model.data_transform = 0
        elif self.radio_buttons[1].isChecked():
            custom_model.data_transform = 1
        else:
            custom_model.data_transform = 2
        custom_model.can_deal_heterogeneity = self.can_deal_heterogeneity_option.isChecked()
        custom_model.can_deal_high_dims = self.can_deal_high_dims_option.isChecked()
        custom_model.can_deal_multicollinearity = self.can_deal_multicollinearity_option.isChecked()
        custom_model.can_deal_unknown_distribution = self.can_deal_unknown_distribution_option.isChecked()
        custom_model.X_with_geometry = self.x_with_geometry_option.isChecked()
        custom_model.can_stacking = self.can_stacking_option.isChecked()
        custom_model.can_deal_small_samples = self.can_deal_small_samples_option.isChecked()
        return custom_model


