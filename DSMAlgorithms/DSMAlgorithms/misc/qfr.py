import numpy as np
from sklearn.ensemble import RandomForestRegressor

'''
分位数随机森林(QFR Algorithm)的实现

# 代码示例（预测中位数和90%区间）：
# pred_quantiles = qrf.predict(X[:5], quantiles=[0.05, 0.5, 0.95])
# print("分位数预测结果:\n", pred_quantiles)

# ‌预测结果举例解读‌
# 若模型输出某样本的预测分位数为：[0.05=10, 0.5=25, 0.95=40]，表示：
#
# ‌有5%概率‌真实值≤10（悲观情况）
# ‌有50%概率‌真实值≤25（最可能值）
# ‌有95%概率‌真实值≤40（乐观情况）
# 真实值落在10到40之间的概率为90%

# 示例‌：预测某地土壤pH值区间为[5.3, 6.8, 7.5]（0.05/0.5/0.95分位数）
# 解读：95%把握认为真实pH在5.3-7.5之间，最可能值约6.8，且酸性风险（<5.3）低于5%

# 业务决策支持‌
# ‌风控场景‌（金融）：0.05分位数预示“最坏情况”，助机构预留缓冲资金221。
# ‌资源调度‌（能源）：0.95分位数指导储备上限，避免供应短缺

# 通俗总结‌：分位数预测如同天气预报中的“温度范围”（如15℃~28℃），告诉你目标值大概率落在哪个区间，而非只报一个“可能不准”的单一温度值
'''


class QuantileRandomForest(RandomForestRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keep_samples = True  # 保留叶节点样本

    def predict(self, X, quantiles=None):
        if quantiles is None:
            return super().predict(X)  # 默认返回均值

        # 获取每棵树叶节点的样本值
        leaf_ids = self.apply(X)  # 样本到达的叶节点ID
        predictions = []
        for tree in self.estimators_:
            # 收集每个叶节点的训练样本值
            leaf_values = [tree.y_train[leaf_id == tree.apply(tree.X_train)]
                           for leaf_id in leaf_ids]
            # 计算分位数
            preds = [np.quantile(vals, quantiles) for vals in leaf_values]
            predictions.append(preds)

        # 聚合多棵树的结果
        return np.mean(predictions, axis=0)

