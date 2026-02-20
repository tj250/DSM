from abc import ABC
import pickle, os
import algorithms_config
from data.mean_encoder import MeanEncoder
import pandas as pd
from DSMAlgorithms.base.base_data_structure import AlgorithmsType, DataTransformType


'''
数字土壤制图模型基类
'''


class DSMBaseModel(ABC):
    """
    建立此基础模型包装器的目的主要是为了在predict时，能够对传入的协变量数据进行相应的正确处理。这是基于如下两个方面的考虑：
    1、单模型建模时，采用的模型是原始模型，例如（SVR等），由于需要进行参数搜索时的交叉验证，传入fit方法的数据只能有一套，所有数据中的类别变量经过平均编码处理或者采用xgboost自身支持的类别化
    2、堆叠建模时，采用的模型是自定义的模型（...StakingWrapper），而非原始模型（例如SVRStackingWrapper等），由于采用的sklearn堆叠回归器的特殊性，

    algorithms_type:算法的类型
    model:模型
    zscore_normalize:数据z-score或min-max归一化的变换参数
    mean_encoder:使用到的平均编码对象
    """

    def __init__(self, algorithms_type: AlgorithmsType, model, transform_info: dict, mean_encoder: dict = None):
        self.model = model
        self.algorithms_type = algorithms_type
        self.transform_info = transform_info
        if mean_encoder is not None:
            self.mean_encoder = MeanEncoder.Deserialize(mean_encoder)  # 平均编码对象

        self.anti_multicolinear = False         # 默认对多重共线性敏感,True：不敏感/鲁棒，False：敏感
        self.anti_outliers = False              # 默认对异常值敏感,True:不敏感，False:敏感
        self.need_feature_normalize = False     # 默认无需特征归一化
        self.need_feature_scaling = True        # 默认需要特征缩放
        self.with_native_categorical = False    # 默认不支持原生处理类别化变量

        # 特征归一化设置
        if algorithms_type == AlgorithmsType.MLP or algorithms_type == AlgorithmsType.PLSR:  # MLP,PLSR采用min-max归一化
            self.need_feature_normalize = True

        # 设置无需特征缩放（多个解释变量的量纲差异不会对建模结果造成影响）
        if algorithms_type == AlgorithmsType.XGBR or algorithms_type == AlgorithmsType.XGBRFR or algorithms_type == AlgorithmsType.RFR:
            self.need_feature_scaling = False     # XGBR、XGBRFR、RFR无需特征缩放

        # 设置原生支持类别化变量处理
        if algorithms_type == AlgorithmsType.XGBR or algorithms_type == AlgorithmsType.XGBRFR:
            self.with_native_categorical = True     # XGBoost原生支持类别话处理


    '''
    预测
    注意：传入的X中的每一列的值应为原始值，未经过任何数据变换处理。本方法内部将根据模型的特性对数据进行相应的变化处理。包括：
    1、删除不需要的坐标值列
    2、连续型变量进行归一化(mlp和ck)或Z-Score标准化
    3、类别化处理，非xgboost模型进行平均编码（MeanEncoder），注：xgboost模型在入参前需进行过类别化处理，X中的类别列类型应该为pd.Categorical
    '''

    def predict(self, X: pd.DataFrame):
        need_geometry = (self.algorithms_type == AlgorithmsType.MGWR or
                         self.algorithms_type == AlgorithmsType.CK or
                         self.algorithms_type == AlgorithmsType.RK)
        # 1、不需要坐标值列但是X列中包含了坐标值列（堆叠模型时，一个模型需要，但是两外一个又不需要情况夏会发生）,需要去除
        if not need_geometry and algorithms_config.CSV_GEOM_COL_X in X.columns:
            X.pop(algorithms_config.CSV_GEOM_COL_X)
            X.pop(algorithms_config.CSV_GEOM_COL_Y)
        # 2、对连续型变量，采用和建模时相同的z-score值或min-max值进行标准化或归一化处理
        for col in X.columns:
            # 该列为连续型的数值列，并且不是坐标值列（self.algorithms_type为CK，RK，MGWR才有可能出现此种情况）
            if col != algorithms_config.CSV_GEOM_COL_X and col != algorithms_config.CSV_GEOM_COL_Y and not isinstance(X[col].dtype, pd.CategoricalDtype):
                if DataTransformType(self.transform_info['transform_type']) == DataTransformType.Normalize:  # min-max归一化
                    X[col] = (X[col] - self.transform_info[col]['min']) / (self.transform_info[col]['max'] - self.transform_info[col]['min'] + 1e-8)
                elif DataTransformType(self.transform_info['transform_type']) == DataTransformType.ZScore:
                    X[col] = (X[col] - self.transform_info[col]['mean']) / self.transform_info[col]['std']
        # 3、对类别型变量进行处理
        if self.with_native_categorical:  # XGBoost类算法，无需特别处理
            return self.model.predict(X)
        else:
            X_encoder = self.mean_encoder.transform(X)  # 非XGBoost中的回归算法，采用与建模时相同的平均编码器进行平均编码
            return self.model.predict(X_encoder)



