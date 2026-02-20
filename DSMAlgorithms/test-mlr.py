import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
from DSMAlgorithms.base.custom_kfold import QuantileKFold, adjusted_r2, custom_cv_split, r2_score
import algorithms_config

def load_and_preprocess_data(file_path, target_column):
    """
    加载和预处理数据
    """
    # 读取CSV文件
    data = pd.read_csv(file_path)
    print(f"数据形状: {data.shape}")
    print(f"数据前5行:\n{data.head()}")

    # 分离特征和目标变量
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 处理缺失值
    if data.isnull().sum().any():
        print("检测到缺失值，进行删除处理...")
        data = data.dropna()
        X = data.drop(columns=[target_column])
        y = data[target_column]

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns.tolist()


def perform_grid_search(X, y):
    """
    使用网格搜索进行10折交叉验证参数优化
    """
    # 定义参数网格
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    # 创建10折交叉验证对象
    # kfold = KFold(n_splits=10, shuffle=True, random_state=int(time.time()))
    kfold = custom_cv_split(r'D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\BB.30_1\BB.30_1_folds.csv')
    # 定义多个回归模型进行对比
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso()
    }

    best_model = None
    best_score = -np.inf
    best_model_name = ""

    print("\n=== 10折交叉验证结果 ===")
    for name, model in models.items():
        if name == 'LinearRegression':
            # 普通线性回归不需要参数搜索
            scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
            mean_score = np.mean(scores)
            print(f"{name}: 平均R² = {mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_model_name = name
        else:
            # 使用网格搜索进行参数优化
            grid_search = GridSearchCV(
                model, param_grid, cv=kfold,
                scoring='r2', n_jobs=-1
            )
            grid_search.fit(X, y)

            print(f"{name}: 最佳参数 = {grid_search.best_params_}, 最佳R² = {grid_search.best_score_:.4f}")

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_model_name = name

    return best_model, best_model_name, best_score


def evaluate_model(model, X, y, feature_names):
    """
    评估模型性能
    """
    # 10折交叉验证评估
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    mse_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')

    print(f"\n=== 最佳模型评估: {model.__class__.__name__} ===")
    print(f"10折交叉验证 R²: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores) * 2:.4f})")
    print(f"10折交叉验证 MSE: {-np.mean(mse_scores):.4f}")

    # 训练最终模型
    model.fit(X, y)

    # 输出模型系数
    if hasattr(model, 'coef_'):
        coefficients = pd.DataFrame({
            '特征': feature_names,
            '系数': model.coef_
        })
        print(f"\n回归系数:\n{coefficients}")

        if hasattr(model, 'intercept_'):
            print(f"截距项: {model.intercept_:.4f}")

    return model




def main():
    """
    主函数
    """
    # 配置参数 - 请根据实际情况修改
    file_path = r"E:\BB.30_1_pH.csv"  # 替换为你的CSV文件路径
    target_column = "pH_target"  # 替换为你的目标列名

    try:
        # 1. 加载和预处理数据
        X, y, feature_names = load_and_preprocess_data(file_path, target_column)

        # 2. 网格搜索和交叉验证
        best_model, best_model_name, best_score = perform_grid_search(X, y)

        # 3. 评估最佳模型
        final_model = evaluate_model(best_model, X, y, feature_names)

        # 4. 可视化结果
        # visualize_results(final_model, X, y, feature_names)

        print(f"\n建模完成！最佳模型为: {best_model_name}")
        print(f"最佳交叉验证R²得分: {best_score:.4f}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except KeyError:
        print(f"错误: 目标列 '{target_column}' 不存在")
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()
