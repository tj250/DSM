def build_polyr(prop_name, csv_file):
    train_dataset, train_labels, test_dataset, test_labels = split_dataset(prop_name, csv_file)

    degrees = [1, 2, 3, 4]  #
    max_r2 = -1
    for degree in degrees:
        # 使用多项式特征扩展
        poly = PolynomialFeatures(degree=degree)  # 选择多项式的阶数
        X_poly = poly.fit_transform(train_dataset)

        # 创建线性回归模型
        model = LinearRegression()
        model.fit(X_poly, train_labels)

        X_test_poly = poly.transform(test_dataset)
        test_predictions = model.predict(X_test_poly)
        test_predictions = test_predictions.flatten()

        # model.summary()
        test_mean = test_labels.mean(axis=0)  # 实测值的均值
        print("当前阶数:{}".format(degree))
        if is_nan(test_predictions[0]):  # 预测结果为非数值，表示预测异常,跳过此次结果
            print('预测结果包含Nan')
            continue
        MAE = round(abs(test_predictions - test_labels).mean(), 4)
        print('平均误差(MAE): ', MAE)
        MSE = round(mean_squared_error(test_labels, test_predictions), 4)
        print('均方误差(MSE): ', MSE)
        RMSE = round(((test_predictions - test_labels) ** 2).mean() ** 0.5, 4)
        print('均方根误差(RMSE): ', RMSE)
        R2 = round(r2_score(test_labels, test_predictions), 4)
        print('决定系数 (R2): ', R2)

        r21 = model.score(X_poly, train_labels)  # 训练集上的R2
        r22 = model.score(X_test_poly, test_labels)  # 测试集上的R2

        if R2 > max_r2:
            max_r2 = R2
            best_degree = degree
            best_model = model
            best_test_predictions = test_predictions
    model_param = {'degree': best_degree}
    save_regressor_result(prop_name, 'polyr', best_model, model_param, test_labels, best_test_predictions,
                          (test_dataset, test_labels))

def is_nan(num):
    return num != num

def split_dataset(prop_name, csv_file):
    # 导入数据
    df = pd.read_csv(csv_file)
    dataset = df.copy()

    train_dataset, test_dataset = train_test_split(dataset, test_size=TEST_DATASET_SIZE, random_state=RANDOM_STATE)

    pixel_bits = gdal_wrap.TIFF_PIXEL_BITS  # 8或32位无符号颜色
    max_pixel_value = 2 ** (pixel_bits) - 1
    # 获取训练集的目标变量
    train_labels = train_dataset.pop(prop_name)
    # 获取测试集的目标变量
    test_labels = test_dataset.pop(prop_name)

    # 处理类别属性
    if '母岩' in train_dataset.columns:
        train_dataset['母岩'] = train_dataset['母岩'].astype('category', copy=False)
    if '母质' in train_dataset.columns:
        train_dataset['母质'] = train_dataset['母质'].astype('category', copy=False)
    if 'TDLY' in train_dataset.columns:
        train_dataset['TDLY'] = train_dataset['TDLY'].astype('category', copy=False)

    return train_dataset, train_labels, test_dataset, test_labels