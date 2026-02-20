import numpy as np
import pandas as pd
from scipy import stats
from .data_distribution import normality_distribution_test, normality_test_after_transormed
import pingouin as pg

# Henze-Zirkler检验
'''
Henze-Zirkler's Test:一种非参数检验方法，对样本大小和分布形态不敏感，但计算复杂度轻。
Henze-Zirkler检验的返回值由检验统计量（HZ）和对应的p值组成，其解读需结合统计量与分布特性：
‌检验统计量（HZ）‌:
该统计量衡量样本数据分布与多元正态分布的特征函数距离，数值越大表示偏离正态性越显著‌
具体计算基于样本协方差矩阵和特征函数差异，具有仿射不变性‌
p值解读‌:
p > 显著性水平（如0.05）‌：不能拒绝原假设，认为数据服从多元正态分布‌
p ≤ 显著性水平‌：拒绝原假设，认为数据不满足多元正态性‌

'''
def henze_zirkler_test(data):
    n, p = data.shape
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_cov = np.linalg.inv(cov)

    # 计算检验统计量
    D = np.zeros(n)
    for i in range(n):
        diff = data[i] - mean
        D[i] = diff @ inv_cov @ diff.T

    beta = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4) ** (1 / (p + 4)) * (n ** (1 / (p + 4)))
    HZ = n * (1 / (n ** 2) * np.sum([np.sum([np.exp(- (beta ** 2) / 2 * np.linalg.norm(data[i] - data[j]) ** 2)
                                             for j in range(n)]) for i in range(n)])
              - 2 * ((1 + beta ** 2) ** (-p / 2)) * (1 / n) * np.sum(
                [np.exp(- (beta ** 2) / (2 * (1 + beta ** 2)) * D[i])
                 for i in range(n)])
              + ((1 + 2 * beta ** 2) ** (-p / 2)))

    # 计算p值
    w = (1 + beta ** 2) * (1 + 3 * beta ** 2)
    a = 1 + 2 * beta ** 2
    L = (1 - a ** (-2)) * (p * (p + 2)) / (8 * a ** 2) + p / (2 * a ** 2)
    mu = 1 - a ** (-1) * (1 + (p * beta ** 2) / a + (p * (p + 2) * beta ** 4) / (2 * a ** 2))
    sigma2 = 2 * (1 - a ** (-2)) + 4 * L + 2 * (1 - w ** (-1)) * p * (p + 2)
    p_value = 1 - stats.norm.cdf((HZ - mu) / np.sqrt(sigma2))

    return HZ, p_value


# Mardia检验（偏度和峰度检验）
'''
Mardia’s Test:检验多元数据的偏度和峰度，适用于大规模样本。
'''
def mardia_test(data):
    n, p = data.shape
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_cov = np.linalg.inv(cov)

    # 计算偏度
    A = np.zeros((n, p))
    for i in range(n):
        A[i] = data[i] - mean
    g1 = np.sum([(A[i] @ inv_cov @ A[j]) ** 3 for i in range(n) for j in range(n)]) / (6 * n ** 2)

    # 计算峰度
    g2 = np.sum([(A[i] @ inv_cov @ A[i]) ** 2 for i in range(n)]) / n
    kurtosis = (g2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)

    return g1, kurtosis



'''
使用皮尔逊方法对两个变量进行线性相关性分析
x:自变量
y:因变量
threshold：阈值，超过此阈值，则认为具有较强的线性相关性
p:相关性是否显著的阈值

首先需要先进行双变量正态分布(二元正态分布)检验，如果通过，方可进行皮尔逊相关性检验
'''
def pearson_analyze(x: pd.Series, y: pd.Series, threshold = 0.2, p=0.05):
    x_arr = x.to_numpy()
    y_arr = y.to_numpy()
    del_list = []
    for i in range(x_arr.shape[0]): # 寻找有异常值的行，排除掉
        if x_arr[i] == -9999 or y_arr[i] == -9999:
            del_list.append(i)
    x_arr = np.delete(x_arr, del_list)
    y_arr = np.delete(y_arr, del_list)

    xy = np.column_stack((x_arr, y_arr))
    _, _, normal = pg.multivariate_normality(xy, alpha=p) # 双变量正态分布检验
    if normal: # 符合双变量正态分布,则可以进一步进行pearson检验
        correlation, p = stats.pearsonr(x_arr, y_arr)
        return p < p and correlation > threshold
    else:
        if normality_distribution_test(x_arr) and normality_distribution_test(y_arr):
            return True
        return False



'''
对类别型变量进行单因素方差检验
x:自变量
y:因变量
threshold：阈值，超过此阈值，则认为具有较强的线性相关性
p:相关性是否显著的阈值

单因素方差分析（One-Way ANOVA）‌ 的核心用途就是分析 ‌一个类别型自变量（分类变量）‌ 与 ‌一个连续型因变量（数值变量）‌ 之间的关系。
适用条件‌：
因变量需近似正态分布（每组数据）。
各组方差需齐性（方差相近）。
数据需相互独立。
注意：
1、此方法内部，如果不同类别有超过一半不满足正态分布（包括变换后不满足正态分布），则退出，无法通过anova检验
2、方差齐性检验通不过，也会退出，无法通过anova检验
3、anova结果还要求f统计量大于阈值
'''
def anova_analyze(x: pd.Series, y: pd.Series, threshold = 1, p=0.05):
    df = pd.concat([x, y], axis=1)
    # 1. 正态性检验（每组）
    groups = df.groupby(x.name)[y.name]
    normality_results = {}
    unnormal_count = 0
    new_groups = {}
    for name, group in groups: # 遍历每一个类别(每一组)
        if not normality_distribution_test(group, p):  # 该组不符合正态分布，则尝试进一步进行变换后检测
            transformed_data, _, _ = normality_test_after_transormed(group, p)
            normality_results[name] = transformed_data is not None
            if normality_results[name]: # 如果变换后符合正态分布,则需要将原始数据进行变换
                new_groups[name] = transformed_data  # 变换，用于后续计算
            else:
                unnormal_count+=1  # 不满足正态分布的类别计数
                new_groups[name] = group.to_numpy()
        else:
            normality_results[name] = True  # 该类别数据满足正态性
            new_groups[name] = group.to_numpy()
    if unnormal_count > 0.5 * len(groups):  # 不满足正态分布的类别过多，超过了类别数量的一半,则认为难以完成anova检验，直接退出
        return False

    # 2. 方差齐性检验
    stat, p_levene = stats.levene(*[group for name, group in new_groups.items()])
    if p_levene <= p: # 方差齐性检测不通过(各组方差存在显著差异-方差不齐)
        # 需进行非参数检验（Kruskal-Wallis检验）,或Welch anova，或Brown-Forsythe anova
        # 这里使用Kruskal-Wallis检验
        statistic, p_kruskal = stats.kruskal(*[group for name, group in groups])
        return p_kruskal < p and statistic > threshold
    else:  # 方差齐
        # 3. 单因素ANOVA
        groups_array = [group.values for name, group in groups]
        f_stat, p_oneway = stats.f_oneway(*groups_array)
        return p_oneway < p and f_stat > threshold


'''
自变量和因变量之间的线性关系检验
X:自变量
y:因变量
continuous_variables:自变量中的连续变量名称列表
p:显著性检测阈值
'''
def linear_relation_test(X:pd.DataFrame, y:pd.Series, continuous_variables:list[str], p=0.05)->bool:
    have_linear_relation = False
    for x_column in X.columns:
        if x_column in continuous_variables:  # 连续型解释变量与响应变量线性相关性分析
            if pearson_analyze(X[x_column], y, threshold=0.2, p=p):
                have_linear_relation = True # 存在至少一个解释变量和响应变量存在线性相关关系
                break
        else:  # 类别型解释变量与响应变量单因素方差分析
            if anova_analyze(X[x_column], y, threshold=1, p=p):
                have_linear_relation = True  # 存在至少一个解释变量和响应变量存在线性相关关系
                break
    return have_linear_relation








def anova_test():  # only for test
    data = {
        'Drug': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
        'BloodPressure': np.concatenate([np.random.normal(120, 5, 10),
                                         np.random.normal(115, 5, 10),
                                         np.random.normal(125, 5, 10)])
    }
    df = pd.DataFrame(data)
    # 1. 正态性检验（每组）
    groups = df.groupby('Drug')['BloodPressure']
    normality_results = {}
    for name, group in groups:
        stat, p = stats.shapiro(group)
        normality_results[name] = {'Shapiro_p': p}

    # 2. 方差齐性检验
    abc = [group for name, group in groups]
    stat, p_levene = stats.levene(*[group for name, group in groups])

    # 3. 单因素ANOVA
    groups_array = [group.values for name, group in groups]
    f_stat, p_anova = stats.f_oneway(*groups_array)




if __name__ == '__main__':
    # 生成示例数据（二元正态分布）
    np.random.seed(42)
    mu = [0, 0]  # 均值向量
    cov = [[1, 0.5], [0.5, 1]]  # 协方差矩阵
    data = np.random.multivariate_normal(mu, cov, 100)

    hz_stat, hz_p = henze_zirkler_test(data)
    g1, kurtosis = mardia_test(data)

    mvn = pg.multivariate_normality(data)

    print(f"Henze-Zirkler检验统计量: {hz_stat:.4f}, p值: {hz_p:.4f}")
    print(f"Mardia偏度检验: {g1:.4f}, 峰度检验: {kurtosis:.4f}")