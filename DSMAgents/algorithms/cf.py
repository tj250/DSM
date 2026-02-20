from surprise import Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split
import pandas as pd

# 协同过滤算法

def cf(data: pd.DataFrame):
    data = Dataset.load_builtin('ml-100k')
    # 将数据集划分为训练集和测试集
    trainset, testset = train_test_split(data, test_size=0.25)

    # 构建基于用户的协同过滤算法模型
    sim_options = {'name': 'cosine', 'user_based': True}
    algo = KNNBasic(sim_options=sim_options)

    # 在训练集上训练模型
    algo.fit(trainset)

    # 在测试集上评估模型性能
    predictions = algo.test(testset)

    # 输出模型的RMSE指标
    accuracy.rmse(predictions)