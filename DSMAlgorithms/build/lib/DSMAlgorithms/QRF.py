import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def calculate_picp(y_true, lower, upper):
    """计算预测区间覆盖概率(PICP)"""
    return np.mean((y_true >= lower) & (y_true <= upper)) * 100


def calculate_mpiw(lower, upper):
    """计算平均预测区间宽度(MPIW)"""
    return np.mean(upper - lower)


def qrf_uncertainty_analysis(X, y, quantiles=[0.05, 0.5, 0.95], test_size=0.2, random_state=42):
    """
    分位数随机森林不确定性分析主函数
    参数:
        X: 环境协变量 (n_samples, n_features)
        y: 土壤属性值 (n_samples,)
        quantiles: 需要预测的分位数
    """
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 训练QRF模型
    qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=random_state)
    qrf.fit(X_train, y_train)

    # 预测分位数
    pred_quantiles = qrf.predict(X_test, quantiles=quantiles)
    lower = pred_quantiles[:, 0]  # 0.05分位数
    median = pred_quantiles[:, 1]  # 中位数
    upper = pred_quantiles[:, 2]  # 0.95分位数

    # 计算不确定性指标
    rmse = np.sqrt(mean_squared_error(y_test, median))
    picp = calculate_picp(y_test, lower, upper)
    mpiw = calculate_mpiw(lower, upper)

    # 结果汇总
    results = {
        'RMSE': rmse,
        'PICP': picp,
        'MPIW': mpiw,
        'Lower_bound': lower,
        'Median': median,
        'Upper_bound': upper,
        'True_values': y_test
    }

    return results, qrf


# 示例用法
if __name__ == "__main__":
    # 模拟数据 (实际应用中替换为土壤数据)
    np.random.seed(42)
    X = np.random.rand(1000, 5)  # 5个环境变量
    y = 10 * X[:, 0] + 5 * X[:, 1] + np.random.normal(0, 1, 1000)

    # 运行QRF不确定性分析
    results, model = qrf_uncertainty_analysis(X, y)

    # 打印关键指标
    print(f"RMSE: {results['RMSE']:.3f}")
    print(f"PICP: {results['PICP']:.1f}% (目标: 90%)")
    print(f"MPIW: {results['MPIW']:.3f}")
