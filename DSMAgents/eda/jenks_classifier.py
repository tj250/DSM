import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
使用kmeans对数据进行聚类分类,返回变换后的数据
'''

def jenks_breaks(data, n_bins=4):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
    return discretizer.fit_transform(data)

'''
对连续值进行类别化处理
data:必须为二维numpy数组
'''
def optimal_jenks_bins(data, max_classes=10):
    """自动确定Jenks最优分类数"""
    results = []

    for n in range(2, max_classes + 1):
        try:
            breaks = jenks_breaks(data, n)
            # 计算组内方差
            group_variances = []
            for i in range(1, len(breaks)):
                group_data = data[(data >= breaks[i - 1]) & (data < breaks[i])]
                if len(group_data) > 0:
                    group_variances.append(np.var(group_data))
            total_variance = sum(group_variances)
            results.append((n, total_variance))
        except Exception as e:
            print(f"Error calculating {n} classes: {str(e)}")
            continue

    # 手肘法可视化
    # plt.figure(figsize=(8, 4))
    # plt.plot([r[0] for r in results], [r[1] for r in results], 'o-')
    # plt.xlabel('Number of Classes')
    # plt.ylabel('Within-group Variance')
    # plt.title('Elbow Method for Optimal Jenks Bins')
    # plt.grid()
    # plt.show()

    # 选择拐点作为最优类别数
    optimal_n = 3  # 默认值
    if len(results) > 1:
        variances = [r[1] for r in results]
        diffs = np.diff(variances)
        optimal_n = results[np.argmin(diffs)][0]

    return optimal_n


def classify_with_jenks(data, n_classes=None):
    """执行Jenks分类"""
    if n_classes is None:
        n_classes = optimal_jenks_bins(data)
    # data = np.array(data).reshape(-1, 1)
    return jenks_breaks(data, n_classes)


# 测试示例
if __name__ == "__main__":
    np.random.seed(4300)
    test_data = np.concatenate([
        np.random.normal(10, 2, 500),
        np.random.normal(30, 3, 500),
        np.random.normal(60, 5, 500)
    ])

    optimal_classes = optimal_jenks_bins(test_data)
    classified = classify_with_jenks(test_data, optimal_classes)
    print(f"Optimal classes: {optimal_classes}")
    print("Classification counts:\n", classified.value_counts())
