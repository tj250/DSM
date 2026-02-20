from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
import numpy as np
from scipy.stats import multivariate_normal

'''
双变量正态分布检验：
通过计算数据点的马氏距离（考虑协方差矩阵的距离），若距离平方服从卡方分布，则可能符合双变量正态分布‌
'''
def twovars_normality_test(x1, x2):
    data = np.vstack((x1, x2))
    # 假设data是二维数组，每行一个样本
    cov = np.cov(data.T)  # 计算协方差矩阵
    inv_cov = np.linalg.inv(cov)
    mean = np.mean(data, axis=0)

    # 计算马氏距离平方
    mahalanobis_dist = [mahalanobis(x, mean, inv_cov)**2 for x in data]

    # 卡方检验
    _, p_value = chi2.fit(mahalanobis_dist, df=2)  # df=变量数
    print(f"卡方检验p值: {p_value}")  # p>0.05则接受正态假设

'''
多变量正态分布检验
1、使用multivariate_normal拟合均值和协方差矩阵
2、计算每个数据点的概率密度值，通过统计量初步判断正态性
'''
def multivars_normality_test(x1, x2):
    data = np.vstack((x1, x2))
    # 拟合多元正态分布模型
    mvn = multivariate_normal(data.mean(axis=0), np.cov(data.T, rowvar=False))

    # 检验概率密度
    pdf_values = mvn.pdf(data)
    print(f"样本概率密度均值: {pdf_values.mean()}, 方差: {pdf_values.var()}")

    # 可视化检验（需matplotlib）
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = data[:, 0]
    y = data[:, 1]
    z = mvn.pdf(data)
    ax.scatter(x, y, z, c='r', marker='o')
    plt.show()