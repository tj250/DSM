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
from eda.multicollinearity import vif_feature_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from eda.linear_relation import linear_relation_test

'''
检测线性回归模型是否适用于特定的任务
X:自变量数据
y:因变量数据
continuous_variables:自变量中的连续变量列表
p:显著性检验的p值
'''
def hypothesis_testing(X:pd.DataFrame, y:pd.Series, continuous_variables:list[str], p=0.05)->bool:
    # 检验1：自动化的线性关系检验(摆脱需要人参与的可视化后判断)
    have_linear_relation = linear_relation_test(X, y, continuous_variables, p)
    if not have_linear_relation:
        return False

    # 检验2：多重共线性检测与消除
    X_new = vif_feature_selection(X)

    # 特征选择（基于正则化）
    ridge = Ridge(alpha=0.5).fit(X_new, y)
    selector = SelectFromModel(ridge, threshold="median").fit(X_new, y)
    X_selected = selector.transform(X_new)

    # 基于特征选择后的数据重新建模
    X_with_const = sm.add_constant(X_selected)
    model = OLS(y, X_with_const).fit()

    # 检验3：残差正态性
    stat, p = shapiro(model.resid)
    print(f'Shapiro-Wilk test: p={p:.4f}')
    if p < p:  # 非正态分布
        return False

    # 检验4：同方差性检验，线性回归模型的误差项的方差应保持恒定（即同方差性）
    _, p2, _, _ = het_breuschpagan(model.resid,  model.model.exog)
    print(f'Breusch-Pagan test: p={p2:.4f}')
    if p2 < p:  # 存在异方差，拒绝原假设
        return False

    return True


if __name__ == '__main__':
    # 模拟数据
    X = pd.DataFrame(np.random.rand(100, 5), columns=['x1', 'x2', 'x3', 'x4', 'x5'])
    y = 2*X['x1'] + 3*X['x2'] + np.random.normal(0, 0.5, 100)
    hypothesis_testing(X, y, "x1,x2,x3,x4,x5")