import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score, CustomKFold

def load_and_preprocess_data(file_path, target_column):
    """
    加载CSV文件并进行数据预处理
    """
    # 读取CSV文件，支持中文路径和编码
    df = pd.read_csv(file_path, engine="python", encoding="utf_8_sig")

    print(f"数据形状: {df.shape}")
    print(f"数据列名: {df.columns.tolist()}")
    print(f"数据前5行:\n{df.head()}")

    # 检查缺失值
    if df.isnull().sum().any():
        print("检测到缺失值，进行删除处理...")
        df = df.dropna()

    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 标准化特征数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns.tolist(), scaler


def perform_10_fold_cross_validation(X, y, feature_names):
    """
    执行10折交叉验证
    """
    import time
    kf = KFold(n_splits=10, shuffle=True, random_state=int(time.time()))
    kf = CustomKFold(
        r'D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\BB.30_1\BB.30_1_folds.csv')
    r2_scores = []
    mse_scores = []
    models = []

    print("\n=== 10折交叉验证结果 ===")
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 添加常数项
        X_train_with_const = sm.add_constant(X_train)
        X_test_with_const = sm.add_constant(X_test)

        # 创建并训练模型
        model = sm.OLS(y_train, X_train_with_const).fit()
        models.append(model)

        # 预测
        y_pred = model.predict(X_test_with_const)

        y_true_all.extend(y_test.values)
        y_pred_all.extend(y_pred)

        # 计算评估指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        r2_scores.append(r2)
        mse_scores.append(mse)

        print(f"折数 {fold}: R² = {r2:.4f}, MSE = {mse:.4f}")

    # 计算平均性能
    avg_r2 = np.mean(r2_scores)
    avg_mse = np.mean(mse_scores)

    print(f"\n平均性能:")
    print(f"平均 R²: {avg_r2:.4f} (+/- {np.std(r2_scores) * 2:.4f})")
    print(f"平均 MSE: {avg_mse:.4f}")

    # Calculate overall performance
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    mean_r2 = r2_score(y_true_all, y_pred_all)
    mean_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    return models, r2_scores, mse_scores


def find_best_model(models, r2_scores):
    """
    根据交叉验证结果选择最佳模型
    """
    best_idx = np.argmax(r2_scores)
    best_model = models[best_idx]
    best_score = r2_scores[best_idx]

    print(f"\n最佳模型: 第 {best_idx + 1} 折")
    print(f"最佳 R² 得分: {best_score:.4f}")

    return best_model, best_idx


def evaluate_best_model(best_model, feature_names):
    """
    评估最佳模型的详细统计信息
    """
    print("\n=== 最佳模型详细统计 ===")
    print(best_model.summary())

    # 输出回归系数
    coefficients = pd.DataFrame({
        '特征': ['截距'] + feature_names,
        '系数': best_model.params,
        '标准误': best_model.bse,
        't值': best_model.tvalues,
        'P值': best_model.pvalues
    })

    print(f"\n回归系数详情:\n{coefficients}")

    return coefficients


def check_model_assumptions(model, X, y):
    """
    检查模型假设条件
    """
    print("\n=== 模型假设检查 ===")

    # 检查多重共线性
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["特征"] = ['const'] + feature_names
    vif_data["VIF"] = [variance_inflation_factor(X_with_const, i) for i in range(X_with_const.shape[1])]

    print(f"方差膨胀因子(VIF):\n{vif_data}")

    # 检查残差正态性
    residuals = model.resid
    print(f"\n残差统计:")
    print(f"残差均值: {residuals.mean():.4f}")
    print(f"残差标准差: {residuals.std():.4f}")

    return vif_data


def main():
    """
    主函数
    """
    # 配置参数 - 请根据实际情况修改
    file_path = r"E:\BB.30_1_pH.csv"  # 替换为你的CSV文件路径
    target_column = "pH_target"  # 替换为你的目标列名

    try:
        # 1. 加载和预处理数据
        X, y, feature_names, scaler = load_and_preprocess_data(file_path, target_column)

        # 2. 执行10折交叉验证
        models, r2_scores, mse_scores = perform_10_fold_cross_validation(X, y, feature_names)

        # 3. 选择最佳模型
        best_model, best_idx = find_best_model(models, r2_scores)

        # 4. 评估最佳模型
        coefficients = evaluate_best_model(best_model, feature_names)

        # 5. 检查模型假设
        # vif_data = check_model_assumptions(best_model, X, y)

        print(f"\n建模完成！最佳模型为第 {best_idx + 1} 折")
        print(f"最终回归方程: y = {best_model.params[0]:.4f}", end="")

        for i, (coef, name) in enumerate(zip(best_model.params[1:], feature_names)):
            print(f" + {coef:.4f}*{name}", end="")
        print()

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except KeyError:
        print(f"错误: 目标列 '{target_column}' 不存在")
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()
