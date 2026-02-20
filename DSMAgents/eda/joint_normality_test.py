
import numpy as np
from scipy.stats import anderson

# 生成示例数据（两个正态分布变量）
np.random.seed(0)
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
data = np.random.multivariate_normal(mean, cov, size=100)

# 提取两个变量
x = data[:, 0]
y = data[:, 1]

# 计算协方差矩阵
cov_matrix = np.cov(data, rowvar=False)

# 计算Mahalanobis距离
inv_cov = np.linalg.inv(cov_matrix)
mahalanobis = np.sqrt(np.sum((data - mean) @ inv_cov * (data - mean), axis=1))

# 多变量Anderson-Darling检验
ad_result = anderson(mahalanobis, dist='chi2', params=[2])

# 输出检验结果
print(f"统计量: {ad_result.statistic:.4f}")
print(f"临界值: {ad_result.critical_values}")
print(f"P值: {ad_result.significance_level:.4f}")

# 判断是否符合正态分布
if ad_result.statistic < ad_result.critical_values[2]:
    print("结论: 数据符合联合正态分布")
else:
    print("结论: 数据不符合联合正态分布")
