import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


def multivariate_linear_assumption_test(data, response_col, predictor_cols):
    """
    多元线性回归假设检验
    检验所有解释变量与响应变量的线性关系
    """

    # 提取数据
    y = data[response_col]
    X = data[predictor_cols]

    # 1. 线性关系检验
    print("=" * 50)
    print("1. 线性关系检验")
    print("=" * 50)

    # 计算相关系数矩阵
    corr_matrix = data.corr()
    print("相关系数矩阵:")
    print(corr_matrix.round(4))

    # 绘制相关系数热力图
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title("变量相关系数矩阵热力图")
    # plt.show()

    # 逐个检验解释变量与响应变量的线性关系
    linear_test_results = {}
    for col in predictor_cols:
        # 计算皮尔逊相关系数
        corr_coef, p_value = stats.pearsonr(data[col], y)
        linear_test_results[col] = {
            'correlation': corr_coef,
            'p_value': p_value
        }
        print(f"{col} 与 {response_col} 的相关性: 相关系数={corr_coef:.4f}, P值={p_value:.4f}")

        # 判断线性关系显著性
        if p_value < 0.05:
            print(f"  -> {col} 与响应变量存在显著线性关系")
        else:
            print(f"  -> {col} 与响应变量线性关系不显著")

    # 2. 多元线性回归模型拟合
    print("\n" + "=" * 50)
    print("2. 多元线性回归模型")
    print("=" * 50)

    # 拟合模型
    model = LinearRegression()
    model.fit(X, y)

    # 预测值
    y_pred = model.predict(X)

    # 计算R²
    r2 = r2_score(y, y_pred)
    print(f"模型R²: {r2:.4f}")

    # 输出回归系数
    print("回归系数:")
    for i, col in enumerate(predictor_cols):
        print(f"  {col}: {model.coef_[i]:.4f}")
    print(f"  截距: {model.intercept_:.4f}")

    # 3. 残差分析
    print("\n" + "=" * 50)
    print("3. 残差分析")
    print("=" * 50)

    # 计算残差
    residuals = y - y_pred

    # 残差正态性检验 (Shapiro-Wilk检验)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"残差正态性检验 (Shapiro-Wilk): 统计量={shapiro_stat:.4f}, P值={shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print("  -> 残差符合正态分布假设")
    else:
        print("  -> 残差不符合正态分布假设")

    # 绘制残差图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 残差vs拟合值图
    axes[0, 0].scatter(y_pred, residuals)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('拟合值')
    axes[0, 0].set_ylabel('残差')
    axes[0, 0].set_title('残差 vs 拟合值')

    # 残差直方图
    axes[0, 1].hist(residuals, bins=20, edgecolor='black')
    axes[0, 1].set_xlabel('残差')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('残差分布直方图')

    # Q-Q图
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('残差Q-Q图')

    # 残差vs解释变量图 (以第一个变量为例)
    # axes[1, 1].scatter(X.iloc[:, 0], residuals)
    # axes[1, 1].axhline(y=0, color='r', linestyle='--')
    # axes[1, 1].set_xlabel(predictor_cols[0])
    # axes[1, 1].set_ylabel('残差')
    # axes[1, 1].set_title(f'残差 vs {predictor_cols[0]}')
    #
    # plt.tight_layout()
    # plt.show()

    # 4. 方差齐性检验
    print("\n" + "=" * 50)
    print("4. 方差齐性检验")
    print("=" * 50)

    # Breusch-Pagan检验 (简化版)
    # 计算标准化残差
    standardized_residuals = residuals / np.std(residuals)

    # 检验残差与拟合值的关系
    bp_corr, bp_p = stats.pearsonr(y_pred, standardized_residuals ** 2)
    print(f"方差齐性检验 (Breusch-Pagan简化版): 相关系数={bp_corr:.4f}, P值={bp_p:.4f}")
    if bp_p > 0.05:
        print("  -> 满足方差齐性假设")
    else:
        print("  -> 不满足方差齐性假设")

    # 5. 多重共线性检验
    print("\n" + "=" * 50)
    print("5. 多重共线性检验")
    print("=" * 50)

    # 计算方差膨胀因子(VIF)
    from sklearn.linear_model import LinearRegression as LR

    vif_data = pd.DataFrame()
    vif_data["变量"] = predictor_cols
    vif_data["VIF"] = [0 for _ in range(len(predictor_cols))]

    for i in range(len(predictor_cols)):
        X_vif = X.drop(predictor_cols[i], axis=1)
        y_vif = X[predictor_cols[i]]

        vif_model = LR()
        vif_model.fit(X_vif, y_vif)
        r2_vif = vif_model.score(X_vif, y_vif)
        vif = 1 / (1 - r2_vif)
        vif_data.loc[i, "VIF"] = vif

        print(f"{predictor_cols[i]} 的VIF值: {vif:.4f}")
        if vif > 10:
            print(f"  -> {predictor_cols[i]} 存在严重多重共线性")
        elif vif > 5:
            print(f"  -> {predictor_cols[i]} 存在中等程度多重共线性")
        else:
            print(f"  -> {predictor_cols[i]} 多重共线性不严重")

    # 6. 综合结论
    print("\n" + "=" * 50)
    print("6. 综合结论")
    print("=" * 50)

    # 线性关系结论
    linear_significant = [col for col, result in linear_test_results.items() if result['p_value'] < 0.05]
    if len(linear_significant) > 0:
        print(f"具有显著线性关系的变量: {', '.join(linear_significant)}")
    else:
        print("没有发现具有显著线性关系的变量")

    # 模型整体显著性
    if r2 > 0.7:
        print(f"模型整体拟合良好 (R²={r2:.4f})")
    elif r2 > 0.5:
        print(f"模型整体拟合一般 (R²={r2:.4f})")
    else:
        print(f"模型整体拟合较差 (R²={r2:.4f})")

    # 假设检验结论
    assumptions_met = []
    if shapiro_p > 0.05:
        assumptions_met.append("残差正态性")
    if bp_p > 0.05:
        assumptions_met.append("方差齐性")

    if len(assumptions_met) == 2:
        print("模型满足主要假设条件")
    elif len(assumptions_met) == 1:
        print(f"模型部分满足假设条件: {', '.join(assumptions_met)}")
    else:
        print("模型不满足主要假设条件")

    return {
        'linear_test': linear_test_results,
        'model': model,
        'r2': r2,
        'residuals': residuals,
        'vif': vif_data
    }


# 示例数据生成和测试
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n = 100

    # 生成解释变量
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)

    # 生成响应变量 (带有一定的线性关系和噪声)
    y = 2 * x1 + 1.5 * x2 - 0.5 * x3 + np.random.normal(0, 0.5, n)

    # 创建数据框
    data = pd.DataFrame({
        'response': y,
        'predictor1': x1,
        'predictor2': x2,
        'predictor3': x3
    })

    # 执行多元线性回归假设检验
    print("多元线性回归假设检验示例")
    results = multivariate_linear_assumption_test(
        data=data,
        response_col='response',
        predictor_cols=['predictor1', 'predictor2', 'predictor3']
    )
