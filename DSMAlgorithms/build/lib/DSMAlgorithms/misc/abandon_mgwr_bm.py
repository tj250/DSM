import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

import algorithms_config
from sklearn.metrics import r2_score
from data.data_dealer import mean_encode_dataset
from DSMAlgorithms.base.save_model import save_regressor_result
from DSMAlgorithms.base.dsm_build_model import DSMBuildModel



'''
对地理加权回归的包装器(仅用于建模)
MGWR 以地理加权回归 (GWR) 为基础构建。 它是一种局部回归模型，允许解释变量的系数随空间变化。 每个解释变量都可以在不同的空间尺度上运行。 
GWR 不考虑这一点，但 MGWR 通过针对每个解释变量允许不同的邻域（带宽）来考虑这一点。 
解释变量的邻域（带宽）将决定用于评估适合目标要素的线性回归模型中该解释变量系数的要素。

此工具对于至少具有数百个要素的数据集最有效。 该工具不适用于较小的数据集。
'''


class MGWRWrap_old(DSMBuildModel):

    """
    建模
    save_file:建模完成后的模型保存文件
    train_X:训练集的X
    train_y:训练集的y
    test_X:验证集的X
    test_y:验证集的y
    """

    def build(self, algorithms_id:str, train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, test_y: pd.Series, zscore_normalize: dict):
        # 获取训练用的各个样本的坐标numpy数组
        geometries = train_X.pop(algorithms_config.DF_GEOM_COL)
        coords_train = []
        for xy_point in geometries:
            coords_train.append([xy_point.x, xy_point.y])

        # 因变量转换为numpy数组
        y = train_y.values.reshape((-1, 1))

        # 对自变量中的定类变量进行平均编码处理
        X,mean_encoder = mean_encode_dataset(self.category_vars, train_X, train_y)

        # 计算空间权重矩阵
        kernel = 'bisquare'  # 第一个可调参数，可选择'bisquare','gaussian','exponential'等
        # 只有kernel可以作为可调参数，其它必须固定
        selector = Sel_BW(coords_train, y, X, multi=True, fixed=False, kernel=kernel)
        # 有两个关键参数可调，其它保持默认
        # search_method:目前仅支持黄金搜索（'golden'）和手动间隔('interval')，第二个可调参数
        # criterion：'AICc','AIC','BIC','CV'，第三个可调参数
        selector.search(max_iter=15, max_iter_multi=50)  # 自动选择最佳带宽，最耗时的一步

        coords_train = np.array(coords_train)
        # 运行MGWR模型,根据数据进行拟合建模
        # 只有kernel可以作为可调参数，其它必须固定，第一个可调参数
        mgwr_model = MGWR(coords_train, y, X, selector, kernel=kernel)
        mgwr_results = mgwr_model.fit()

        # 输出建模结果
        print(mgwr_results.summary())

        # 提取回归系数
        coefficients = mgwr_results.params  # 局部回归系数
        intercepts = coefficients[:, 0]  # 截距项
        slopes = coefficients[:, 1:]  # 自变量系数

        # 在验证集上预测
        geometries_test = test_X.pop(algorithms_config.DF_GEOM_COL)
        coords_validation = []
        for xy_point in geometries_test:
            coords_validation.append([xy_point.x, xy_point.y])
        coords_validation = np.array(coords_validation)

        # 将验证集上的因变量转换为numpy数组
        y_test_gt = test_y.values.reshape((-1, 1))  # 因变量

        # 4. 使用模型预测（测试集）
        X_Validation = mean_encoder.transform(test_X).values  # Mean Encoding所需代码
        predicted_salt_content = []
        for i, coord in enumerate(coords_validation):
            # 找到测试点最近的训练点索引
            distances = np.linalg.norm(coords_train - coord, axis=1)
            nearest_index = np.argmin(distances)
            # 使用最近训练点的回归系数进行预测
            intercept = intercepts[nearest_index]
            slope = slopes[nearest_index]
            prediction = intercept + np.dot(X_Validation[i], slope)
            predicted_salt_content.append(prediction)
        predicted_salt_content = np.array(predicted_salt_content)

        # 计算验证集上的误差（均方根误差）
        # best_mgwr_score = root_mean_squared_error(y_test_gt, predicted_salt_content)
        # 计算验证集上的决定系数
        best_mgwr_score = r2_score(y_test_gt, predicted_salt_content)

        # 5. 可视化局部回归系数
        # coef_df = pd.DataFrame(mgwr_results.params, columns=['Intercept', 'B11', 'B2', 'B3', 'B4'])
        # gdf = gpd.GeoDataFrame(
        #     coef_df, geometry=gpd.points_from_xy(coords_train[:, 0], coords_train[:, 1])
        # )
        # gdf.plot(column='B11', cmap='coolwarm', legend=True)
        # matplotlib.use('TkAgg')
        # plt.title('Spatial Distribution of Coefficients for B11')
        # plt.show()

        best_model = {'train_coordinates': coords_train, 'intercepts': intercepts, 'slopes': slopes}
        # 保存结果
        save_regressor_result(algorithms_id, 'mgwr', best_model, None, best_mgwr_score,
                              test_y.to_numpy(), predicted_salt_content, train_X.columns, zscore_normalize, mean_encoder)
