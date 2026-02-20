import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from LimeSoDa import load_dataset
from LimeSoDa.utils import split_dataset

# Set random seed
np.random.seed(2025)

# Load dataset
BB_250 = load_dataset('BB.30_1')

# Perform 10-fold CV
y_true_all = []
y_pred_all = []

for fold in range(1, 11):
    X_train, X_test, y_train, y_test = split_dataset(BB_250, fold=fold, targets='pH_target')

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_true_all.extend(y_test.values)
    y_pred_all.extend(y_pred)

# Calculate overall performance--和limesoda在线的计算方式保持一致：将10折的预测结果进行总体计算，而非每一折计算出r2后取平均值
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
mean_r2 = r2_score(y_true_all, y_pred_all)
mean_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

print("\nSOC prediction (10-fold CV):")
print(f"Mean R-squared: {mean_r2:.7f}")  # Mean R-squared: 0.7507837
print(f"Mean RMSE: {mean_rmse:.7f}")  # Mean RMSE: 0.2448791