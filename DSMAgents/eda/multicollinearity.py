import numpy as np
import pandas as pd
import statsmodels.api as sm
import config
from DSMAlgorithms import MeanEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.formula.api import ols
import math


def calculate_vif(X):
    """
   计算给定自变量矩阵X的方差膨胀因子(VIF)
   """
    # 添加常数项
    X = add_constant(X)
    # 计算每个特征的VIF，注意：通过以下语句末尾的[1:]去除了常数项
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)[1:]
    return vif

'''
基于VIF的多重共线性检测与消除
'''
def vif_feature_selection(X:pd.DataFrame, threshold=10)->pd.DataFrame:
    """
   基于VIF的特征选择
   """
    vif = calculate_vif(X)
    if len(X.columns) > 200 and math.isinf(vif.max()):
        return False, X  # 超高维度数据，并且存在完全多重共线性（VIF为无穷大），则不再检测，而是直接退出
    while vif.max() > threshold:
        # 移除具有最大VIF值的特征
        feature_to_remove = vif.idxmax()
        X = X.drop(columns=[feature_to_remove])

        # 重新计算VIF
        vif = calculate_vif(X)
    return True, X

'''
递归计算方差膨胀系数
'''


def calculate_vif2(data, col):
    data = data.loc[:, col]  # 读取对应列标数据
    vif = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])][1:]
    if max(vif) >= 10:
        index = np.argmax(vif) + 1  # 得到最大值的标号
        del col[index]  # 删除vif值最大的一项
        return calculate_vif2(data, col)  # 递归过程
    else:
        vif = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])][1:]
        return col, vif


'''
环境变量预测属性时的多重共线性检测，计算VIF因子
注意：由于记录数的不同，不同属性相关的环境变量检测结果（剩余的列）可能会存在差异，而属性列的值不会导致差异。
categorical_variables:类别型变量名称的列表
gdf:GeoDataFrame：要处理的样点数据
prop_name：待预测的属性名称
返回值：最终确定需要保留的目标列
'''


def multicollinearity_elimination(with_geometry:bool, categorical_variables: list[str], gdf: pd.DataFrame,
                                  prop_name: str) -> list[str]:
    # 1、剔除几何体列，分离自变量和因变量（平均编码需要）
    if with_geometry:
        df = gdf.drop(config.DF_GEOM_COL, axis=1)
    else:
        df = gdf
    y = df[prop_name]
    X = df.drop(prop_name, axis=1)

    # 2、必须对类别型变量进行平均编码后处理,否则类别值会被错误地认为是连续型的数值
    if categorical_variables is not None and len(categorical_variables) > 0:  # 如果存在类别型变量
        col_names = []
        for var_name in categorical_variables:  # 遍历每一个定类变量,字典中的变量名称必须和dataframe中的列名称保持一致
            if var_name in X.columns:
                col_names.append(var_name)
        mean_encoder = MeanEncoder(col_names, target_type='regression')
        if len(col_names) > 0:  # 如果有定类变量需要处理
            X = mean_encoder.fit_transform(X, y)

    # 3、递归寻找多种共线性，并删除，直至得到最终的无共线性的列
    successful, X = vif_feature_selection(X)
    return successful, X

    # 另一种实现方式，仅返回保留的列
    # cols = VIF(X, y)
    # return cols


'''
使用VIF检测并消除多重共线性
'''
def VIF(X:pd.DataFrame, y:pd.Series) -> list[str]:
    X = sm.add_constant(X)
    cols = X.columns.values.tolist()
    cols, vif = calculate_vif2(X, cols)
    print(vif)
    model = sm.OLS(y, X[cols]).fit()   # 此处重新拟合，仅用于输出统计信息
    print('R2:' + str(model.rsquared))
    print('Prob(F-Stat):' + str(model.f_pvalue))
    print('const:' + str(model.params['const']))
    print(model.summary())
    cols.remove('const')
    return cols

