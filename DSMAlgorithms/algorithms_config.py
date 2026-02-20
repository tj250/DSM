import time
# 提高性能指标的可能方式：
# 1、贝叶斯换为随机搜索，所及搜索次数增大，
# 2、使用nmse作为搜索的目标参数，然后计算R2，注意随机搜索时的代码需要按照limeSoda方式进行更改，目前尚未更改
# 3、更改模型的最大迭代次数
# 4、更改入参的数据的预处理方式
DEBUG_STATE = False  # 当前是否处于调试状态（调试状态下，仅用少量的算法进行建模，制图时，也仅制图少量的行数，以加快运行速度）

TASK_MAX_DURATION = 1  # 多少分钟还没有处理完的任务是需要重新处理的
RANDOM_STATE = int(time.time())  # 用于可再现建模的结果，int(time.time()) 或 # None 或 # 42
KFOLD = 5  # 分位数的分箱数量或数据分割交叉验证的折数（即采用有放回抽样，每次用20%的数据进行验证）
USE_DATA_AUGMENTATION = False  # 是否使用数据增强以扩充样本数据量
if USE_DATA_AUGMENTATION:
    AUGUMENTATION_RATION = 1  # 数据增强的比例：以原始样点数（X）为基准，增强后的样点数量为：X*（1+AUGUMENTATION_RATION）
USE_BAYES_OPTI = True  # 是否使用贝叶斯优化，LimeSoda使用贝叶斯优化
BAYES_INIT_POINTS = 5  # 贝叶斯优化初始点数
BYEAS_ITER_TIMES = 10  # 贝叶斯优化迭代次数
MGWR_BAYES_INIT_POINTS = 3  # 针对MGWR的贝叶斯优化初始点数(MGWR运行速度过慢，且优化参数有限)
MGWR_BYEAS_ITER_TIMES = 3  # 针对MGWR的贝叶斯优化迭代次数(MGWR运行速度过慢，且优化参数有限)
RANDOM_ITER_TIMES = 200  # 使用随机搜索时，默认使用100次迭代
TEST_DATASET_SIZE = 0  # 测试集的比例（如0.2表示20%用作测试），如果值为0，则不分割(全部数据进行交叉验证)
VERBOSE = 1  # 输入日志的详细程度
USE_ADJUSTED_R2 = False  # 是否使用adjusted R2作为评估指标
USE_LIMESODA_KFOLD = False  # 使用LimeSoDa分割交叉验证,10折交叉验证，固定的分割方式,计算R2时，采用所有折汇聚后的结果一次性算出R2
RS_SCOROLING = { 'r2': 'r2'}  # 随机搜索时的评价指标（r2,neg_mean_squared_error）'nmse': 'neg_mean_squared_error','r2': 'r2'
RS_REFIT = True  # 是否使用最好的参数重新拟合模型
DF_GEOM_COL = 'geometry'
CSV_GEOM_COL_X = 'coordinates_x'  # .csv文件中存储几何体x坐标的列的名称
CSV_GEOM_COL_Y = 'coordinates_y'  # .csv文件中存储几何体y坐标的列的名称
DOCKER_MODE = False  # 是否在打包为Docker的方式下运行
DOCKER_DATA_PATH = "/data"  # docker模式运行下的样点数据文件路径
LOCAL_DATA_PATH = r"D:\PycharmProjects\DSM训练数据"  # 非docker模式运行下的样点数据文件路径，和computing_nodes表中的data_mapping_local_path列的内容保持一致
# PREDICTION_MEMORY_BLOCK_SIZE = 100*1024*1024.0  # 预测时的块尺寸大小（10M），由于在cokriging.py中的predict方法的self.uk.execute执行时可能会发生内存不足，因此需分块处理

LIMESODA_FOLDS_FILE_PATH = ''
DO_COVARS_ELLIMATION = True  # 建模时是否对协变量进行多重共线性剔除和特征重要性分析，此参数设置为False仅用于和LimeSoda基准进行对比,其余情况均应设置为True

# 针对不同算法的最大迭代次数的设置，因为不同算法对迭代次数的要求差异很大
MLP_MAX_ITER = 10000    # 100000
SVR_MAX_ITER = 50000    # 500000
GLM_MAX_ITER = 50000
EN_MAX_ITER = 2000
PLSR_MAX_ITER = 2000
