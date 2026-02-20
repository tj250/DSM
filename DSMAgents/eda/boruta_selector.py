
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# 1. 生成模拟回归数据
X, y = make_regression(
    n_samples=200,
    n_features=15,
    n_informative=8,
    noise=0.1,
    random_state=42
)
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

# 2. 初始化Boruta选择器
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=7,
    random_state=42
)
feat_selector = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    verbose=2,
    alpha=0.05,
    random_state=42
)

# 3. 执行特征选择
feat_selector.fit(X, y)

# 4. 结果输出
print("\n特征选择结果:")
results = pd.DataFrame({
    'Feature': feature_names,
    'Selected': feat_selector.support_,   # 特征是否被选择
    'Ranking': feat_selector.ranking_,     # 特征重要性排名
    'WeakSelected':feat_selector.support_weak_
})
print(results)

# 5. 可视化重要性排序
plt.figure(figsize=(10,6))
plt.barh(
    range(len(feature_names)),
    feat_selector.ranking_,
    tick_label=feature_names
)
plt.title('Boruta特征重要性排序')
plt.xlabel('Ranking (越小越重要)')
plt.tight_layout()
plt.show()
