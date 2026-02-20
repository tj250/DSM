DEBUG = False
CSV_GEOM_COL_X = 'coordinates_x'         # .csv文件中存储几何体x坐标的列的名称
CSV_GEOM_COL_Y = 'coordinates_y'         # .csv文件中存储几何体y坐标的列的名称
DF_GEOM_COL = 'geometry'
DOCKER_MODE = False                      # 是否在打包为Docker的方式下运行

RANDOM_STATE = 42
LOCAL_DATA_PATH = r"D:\PycharmProjects\DSM训练数据"   # 非docker模式运行下的样点数据文件路径，和computing_nodes表中的data_mapping_local_path列的内容保持一致
KFOLD = 5  # 注意此配置需要和分布式节点中的KFOLD一致
# DEFAULT_SAMPLE_FILE = r"E:\pHEnv_b.csv"  # 默认样点文件
DEFAULT_SAMPLE_FILE = r"D:\PycharmProjects\DSM训练数据\土壤容重.csv"
DEFAULT_COVARIATES_PATH = r"D:\PycharmProjects\DSM训练数据\25个环境协变量"  # 默认协变量存储目录
DEFAULT_MAPPING_AREA_FILE = r"D:\PycharmProjects\DSM训练数据\template.tif"  # 默认制图区域文件
DEFAULT_VALIDATION_FILE = "" # r"E:\data\DSM_Test\validation.csv"  # 默认独立验证集数据文件

#
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\B.204\B.204_SOC.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\BB.30_1\BB.30_1_SOC.csv"
# --DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\BB.30_2\BB.30_2_pH.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\BB.51\BB.51_pH.csv" # 重试未成功
# --DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\BB.72\BB.72_SOC.csv"
# --DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\BB.250\BB.250_Clay.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\CV.98\CV.98_SOC.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\G.104\G.104_pH.csv"
#-- DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\G.150\G.150_Clay.csv"
#-- DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\H.138\H.138_SOC.csv"
#-- DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\MG.44\MG.44_SOC.csv"
#-- DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\MG.112\MG.112_SOC.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\MGS.101\MGS.101_Clay.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\MWP.36\MWP.36_Clay.csv"
#--DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\NRW.42\NRW.42_SOC.csv"
#--DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\NRW.62\NRW.62_Clay.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\NRW.115\NRW.115_Clay.csv"
#--DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\NSW.52\NSW.52_SOC.csv"
#--DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\O.32\O.32_SOC.csv" # 需手动制指定预测变量
#--DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\PC.45\PC.45_SOC.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\RP.62\RP.62_SOC.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\SA.112\SA.112_pH.csv"
#--DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\SC.50\SC.50_SOC.csv"
#--DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\SC.93\SC.93_SOC.csv"
#--DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\SL.125\SL.125_Clay.csv"
#--DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\SM.40\SM.40_Clay.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\SP.231\SP.231_pH.csv"
#-- DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\SSP.460\SSP.460_Clay.csv"
# DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\SSP.58\SSP.58_Clay.csv"
#-- DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\UL.120\UL.120_Clay.csv"
#-- DEFAULT_SAMPLE_FILE = r"D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\数据集\LimeSoDa\data\W.50\W.50_SOC.csv"


# DEFAULT_COVARIATES_PATH = r""  # 默认协变量存储目录
# DEFAULT_MAPPING_AREA_FILE = r""  # 默认制图区域文件
# DEFAULT_VALIDATION_FILE = r""  # 默认独立验证集数据文件
