import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def safe_vif(data):
    """
    计算VIF并处理可能的除零错误
    """
    X = add_constant(data)  # 添加常数项
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    vif_values = []
    for i in range(X.shape[1]):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_values.append(vif)
        except:
            # 捕获除零错误，标记为无限大
            vif_values.append(np.inf)

    vif_data["VIF"] = vif_values
    return vif_data


# 示例用法
data = pd.DataFrame({
    'x1': [1, 2, 3, 4],
    'x2': [2, 4, 6, 8],  # x2与x1完全线性相关
    'x3': [3, 1, 4, 2]
})

result = safe_vif(data)
print(result)
