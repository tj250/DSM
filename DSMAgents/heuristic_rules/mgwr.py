import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import OLS
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.linear_model import Ridge
from eda.linear_relation import linear_relation_test
from eda.geodetector import factor_detector

'''
检测多尺度地理加权回归模型是否适用于特定的任务
X:自变量数据
y:因变量数据
continuous_variables:自变量中的连续变量列表
p:显著性检验的p值

OLS 是一种全局模型。 假设数据生成过程在空间上是平稳的，因此单个系数可以解释每个解释变量和因变量之间的关系。 
GWR 是一种局部模型，它通过允许系数随空间变化来放宽空间平稳性的假设。 然而，在 GWR 中，通过要求所有解释变量使用相同的邻域，
假设所有局部关系在相同的空间比例上运行。 例如，如果一个解释变量使用 20 个相邻要素，则所有解释变量也必须使用 20 个相邻要素。
然而，MGWR 不仅允许系数随空间变化，而且允许比例随不同解释变量变化。 MGWR 通过为每个解释变量使用单独的邻域，
说明每个解释变量和因变量之间关系的不同空间比例以做到这一点。 这样，
可以将在相对较大的空间比例上运行的解释变量（例如温度或大气压力）与在较小空间比例上运行的变量（例如人口密度或收入中位数）结合起来。

如果您仍然不确定将哪个本地模型（GWR 或 MGWR）应用于您的数据，请从 MGWR 开始。 当 MGWR 运行时，它也在特定设置下执行 GWR。 
在地理处理消息中，您可以找到 GWR 诊断并将其与 MGWR 的诊断进行比较。 或者，
您可以运行多个工具（OLS、GWR 和 MGWR）并使用地理处理消息中列出的 AICc 来比较模型并选择最佳模型。 
如果您选择运行多个工具，请缩放所有模型或不缩放所有模型以确保输出具有可比性。
'''
def hypothesis_testing(X:pd.DataFrame, y:pd.Series, continuous_variables:list[str], p=0.05)->bool:
    # 检验1：变量尺度效应假设,需验证不同解释变量具有独立的最优空间带宽（即各变量影响范围存在差异）。
    # 可通过探索性分析（如散点图）或局部GWR结果初步判断变量尺度的异质性
    # 暂不做检测，由事后参数比较确定
    return True


if __name__ == '__main__':
    # 模拟数据
    X = pd.DataFrame(np.random.rand(100, 5), columns=['x1', 'x2', 'x3', 'x4', 'x5'])
    y = 2*X['x1'] + 3*X['x2'] + np.random.normal(0, 0.5, 100)
    hypothesis_testing(X, y, "x1,x2,x3,x4,x5")