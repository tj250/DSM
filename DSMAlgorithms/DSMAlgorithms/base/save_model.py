import os, json
import os.path
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from db_access.task_dealer import TaskDataAccess

'''
将回归建模的结果进行保存。

algorithms_id:算法ID或堆叠算法ID
algorithms：算法的类型名称
best_model:最优模型
best_param：最优参数
cv_best_score:交叉验证的最佳分数
test_ground_truth_y:测试集上的Ground Truth Y
test_pred_y：测试集上的预测Y
importance_param:特征重要性参数
zscore_normalize:数据的z-score信息或min-max归一化信息
mean_encoder:平均编码(除了xgboost的模型之外，其余模型均应提供此参数的值，此对象用于对类别型的协变量进行编码处理)
'''


def save_regressor_result(algorithms_id: str, algorithms: str, best_model, best_param, cv_r2,
                          test_ground_truth_y,
                          test_pred_y, importance_param, zscore_normalize: dict, mean_encoder=None):
    print(f"参数搜索(交叉验证)时的R2:{cv_r2:.4f}")  # 贝叶斯搜索时，如果是优化MAE，MSE，RMSE指标，返回的是负值，需要取反
    # 计算均方误差 (MSE) 、均方根误差（RMSE）和决定系数 (R2)
    MSE = round(mean_squared_error(test_ground_truth_y, test_pred_y), 4)
    RMSE = round(MSE ** 0.5, 4)
    R2 = round(r2_score(test_ground_truth_y, test_pred_y), 4)
    # print("均方误差 (MSE):", MSE)
    print('测试集均方根误差(RMSE): ', RMSE)
    print("测试集决定系数 (R2):", R2)

    importances_json = None
    if algorithms == 'xgbr' or algorithms == 'xgbrfr':
        # 通过的feature_importances_属性，我们来查看模型的特征重要性：
        importances = best_model.feature_importances_  # 获取特征重要性
        importance_data = {}
        for i in range(len(importances)):
            importance_data[importance_param[i]] = float(importances[i])
        items = importance_data.items()
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)  # 倒序排列
        sorted_dict = dict(sorted_items)
        importances_json = json.dumps(sorted_dict, ensure_ascii=False)

    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    param_str = ''
    if best_param is not None:
        for key, value in best_param.items():
            param_str += key
            param_str += ':'
            param_str += str(value)
            param_str += ','
        param_str = param_str[:-1]
    # 保存R2和模型参数
    data = {'best_score': round(float(cv_r2), 4), 'R2': R2, 'RMSE': RMSE, 'ModelParams': param_str,
            'importance': '' if importances_json is None else importances_json}

    # 将建模结果写入postgres数据库
    if algorithms == 'stacking':
        TaskDataAccess.save_stacking_model(algorithms_id, best_model, zscore_normalize,
                                           None if mean_encoder is None else mean_encoder.serialize(),
                                           data)
    else:
        TaskDataAccess.save_model(algorithms_id, best_model, zscore_normalize, None if mean_encoder is None else mean_encoder.serialize(),
                                  data)
