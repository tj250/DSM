import os
import config
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from langchain_core.messages import HumanMessage
from agents.llm.llm_functions import call_llm_with_pydantic_output
from agents.data_structure.explore_data_pydantic import CategoricalVariables
from dataclasses import dataclass, field
from data_access.structure_data_dealer import StructureDataDealer
from .autocorrelation import AutoCorrelation
from .data_distribution import check_data_distribution, normality_distribution_test, normality_test_after_transormed
from .jenks_classifier import optimal_jenks_bins, classify_with_jenks
from DSMAlgorithms import DataExceptionTest
from .geodetector import factor_detector
from .multicollinearity import multicollinearity_elimination
from .feature_importance import select_important_covars_by_xgboost
from .linear_relation import linear_relation_test
from DSMAlgorithms import DataDistributionType


'''
esda分析结果数据结构
'''


@dataclass
class EsdaAnalysisResult():
    data_distribution: DataDistributionType = DataDistributionType.Unknown  # 响应变量的数据分布
    transformed_is_normality: bool = True  # 响应变量经过多类正态分布变换后是否服从正态分布
    transformed_lambda_val: float = 0  # 正态分布变换所采用的lambda值
    is_autocorrelation: bool = True  # 响应变量是否存在空间自相关
    transform_algorithms: str = ''  # 正态分布变换的方法
    have_heterogeneity: bool = True  # 是否存在空间异质性
    is_with_multicollinearity:bool = False # 是否具有多重共线性
    left_cols_after_rfe: list[str] = field(default_factory=list)  # 经过递归特征消除（RFE）以避免多重共线性后剩余的列
    left_cols_after_fia: list[str] = field(default_factory=list)  # 经过特征重要性筛选后,特种重要性不为0的列
    left_cols_after_rfe_fia: list[str] = field(default_factory=list)  # 经过递归特征消除（RFE）和特征重要性筛选后,特种重要性不为0的列
    # features_importance: dict[str, float] = field(default_factory=dict)  # 特征重要性信息
    have_linear_relation: bool = False  # 是否存在至少一个解释变量和响应变量间存在线性相关性的情况
    is_few_samples: bool = False  # 是否为小样本数据
    is_high_dimension: bool = False  # 是否为高维数据
    with_geometry:bool = False  # 数据中是否带有地理坐标,对于无地理坐标数据，不能使用地理加权回归，克里金等方法


'''
执行探索性空间数据分析
tabular_file:样点数据文件
prediction_variable：预测的土壤属性
continuous_variables：连续型的环境协变量
categorical_variables：类别型的环境协变量

返回分析结果
'''


def execute_analysis(tabular_file: str, prediction_variable: str, continuous_variables: list[str],
                     categorical_variables: list[str]) -> EsdaAnalysisResult:
    analysis_result = EsdaAnalysisResult()
    p_val = 0.05  # 显著性检测的阈值
    # 从文件读出数据，统一存储于GeoDataFrame中
    analysis_result.with_geometry, gdf = StructureDataDealer.read_tabular_data(tabular_file)

    # 1、先总体检测数据分布
    print("检测响应变量的数据分布...")
    analysis_result.data_distribution = check_data_distribution(gdf[prediction_variable].values, p_val)
    # 必要的响应变量异常值检测及剔除
    outliers, lower_bound, upper_bound = DataExceptionTest.exception_test(list(gdf[prediction_variable]),
                                                                          analysis_result.data_distribution == DataDistributionType.Normal)
    if len(outliers) > 0:  # 使用过滤掉异常值后的数据进行分析和检测
        filtered_gdf = gdf[(gdf[prediction_variable] >= lower_bound) & (gdf[prediction_variable] <= upper_bound)]
    else:
        filtered_gdf = gdf
    # 9、是否为高维数据(样本量/特征数<20)
    analysis_result.is_high_dimension = (len(filtered_gdf) / len(filtered_gdf.columns)) < 20  # 如何数据行数与列数的比值不超过20，则认为属于高维数据
    # --------注意，此后均采用剔除异常值以后的filtered_gdf进行运算 -----------

    # 2、响应变量正态分布检验及变换后正态分布检验，因为大多数线性模型均要求响应变量符合正态分布，以此检测结果甄选模型
    print("响应变量正态分布检验及变换后正态分布检验...")
    if analysis_result.data_distribution != DataDistributionType.Normal:  # 非正态分布，检测是否可以通过多种变换变换为正态分布
        transformed_data, analysis_result.transformed_lambda_val, analysis_result.transform_algorithms = normality_test_after_transormed(
            filtered_gdf[prediction_variable].values,
            False, p_val)
        analysis_result.transformed_is_normality = transformed_data is not None  # 如果有变换后数据，则表示经过变换后符合正态分布
    if analysis_result.with_geometry: # 如果数据中包含了空间坐标信息
        # 3、响应变量(全局)空间自相关分析
        print("响应变量(全局)空间自相关分析...")
        analysis_result.is_autocorrelation = AutoCorrelation.check(prediction_variable, filtered_gdf, config.DF_GEOM_COL, p_val)
    # 4、响应变量异质性分析(使用地理探测器的因子探测)
    print("响应变量异质性分析...")
    analysis_result.have_heterogeneity = False  # 默认假定不存在空间异质性
    gdf2 = filtered_gdf.copy()
    for column in gdf2.columns:  # 删除所有非定类的列
        if column == prediction_variable:
            continue
        if column not in categorical_variables:
            del gdf2[column]
    for column in continuous_variables:  # 对所有连续值进行类别化处理
        if column not in filtered_gdf.columns:  # 排除坐标值的列
            continue
        optimal_classes = optimal_jenks_bins(np.array(filtered_gdf[column]).reshape(-1, 1))  # 搜索最优类别数量
        classified = classify_with_jenks(filtered_gdf[[column]], optimal_classes)  # 按照最优类别数量进行类别化处理
        print(column + ': ' + str(optimal_classes))
        gdf2[column] = classified.ravel()
    # q的值域为[0, 1]，值越大说明Y的空间分异性越明显；如果分层是由自变量X生成的，则q值越大表示自变量X对属性Y的解释力越强，反之则越弱。
    # 极端情况下，q值为1表明因子X完全控制了Y的空间分布，q值为0则表明因子X与Y没有任何关系，q值表示X解释了100xq % 的Y。
    # 因子探测
    factor_df = factor_detector(gdf2, prediction_variable, [x for x in gdf2.columns if x not in [prediction_variable]])
    for column_name in factor_df.columns:
        if factor_df.at['p value', column_name] < p_val and factor_df.at[
            'q statistic', column_name] > 0.1:  # 因子检测结果显著且q值超过0.1
            analysis_result.have_heterogeneity = True  # 存在空间异质性
            break
    # 4、解释变量正态分布检验
    gdf3 = filtered_gdf.copy()
    if analysis_result.with_geometry:
        df = gdf3.drop(config.DF_GEOM_COL, axis=1)
    else:
        df = gdf3
    y = df[prediction_variable]
    X = df.drop(prediction_variable, axis=1)

    # 5、解释变量与响应变量的线性相关性分析
    print("解释变量与响应变量的线性相关性分析...")
    analysis_result.have_linear_relation = linear_relation_test(X, y, continuous_variables, p_val)

    # 6、多重共线性消除（针对无法处理多重共线性的模型）
    print("多重共线性检测与消除...")
    successful, test_result = multicollinearity_elimination(analysis_result.with_geometry, categorical_variables, filtered_gdf, prediction_variable)
    if successful:  # 如果检测成功（对于超高维度数据，可能存在完全共线性，无需进一步检测，尽快结束检测并返回）
        analysis_result.left_cols_after_rfe.extend(test_result)
    # 检测是否存在多重共线性,
    # 情况1：1代表去除的geometry列，如果原来的列和保留的列数量不一致，则认为进行了多重共线性剔除
    # 情况2：超高维度数据，存在完全共线性，后续建模应仅依靠特征重要性分析结果进行
    if len(filtered_gdf.columns) > len(analysis_result.left_cols_after_rfe) + 1 or not successful:
        analysis_result.is_with_multicollinearity = True
    else:
        analysis_result.is_with_multicollinearity = False
    # 7、变量特征重要性分析和非重要变量剔除
    print("变量特征重要性分析和非重要变量剔除...")
    analysis_result.left_cols_after_fia = select_important_covars_by_xgboost(X, y)
    # 如果存在多重共线性并且经过RFE后特征数量超过10个（如果特征过少，往往无法表达复杂地理环境和协变量之间的复杂关系，可不做特征重要性筛选），
    # 并且是高维数据，还需要计算先经过RFE，再经过FIA后剩余的列
    if analysis_result.is_with_multicollinearity and len(analysis_result.left_cols_after_rfe)>10 and analysis_result.is_high_dimension:
        analysis_result.left_cols_after_rfe_fia = select_important_covars_by_xgboost(X, y, analysis_result.left_cols_after_rfe)

    print(f"总特征数量：{len(filtered_gdf.columns)-2 if analysis_result.with_geometry else len(filtered_gdf.columns)}",
          f"递归特征消除(RFE)后的特征数量{len(analysis_result.left_cols_after_rfe)},"
          f"特征重要性分析(FIA)后剩余的特征数量{len(analysis_result.left_cols_after_fia)},"
          f"RFE和FIA后剩余的特征数量{len(analysis_result.left_cols_after_rfe_fia)}")

    # 8、是否为小样本数据(样本量<30 或 样本量/自变量<20)
    analysis_result.is_few_samples = (len(filtered_gdf) < 30 or (len(filtered_gdf) / len(filtered_gdf.columns) < 20))
    return analysis_result


'''
从解释变量中解析出类别型变量
'''


def parse_categorical_variables(interpretation_variables: list, ) -> (bool, list):
    categorical_variables_list = []
    categorical_variables_analyzing_prompt = """在进行数据挖掘回归建模时，已确定了如下的解释变量：{}
    请从这些解释变量中找出属于类别型的变量，并将其返回。
    """
    messages = [HumanMessage(
        f"{categorical_variables_analyzing_prompt}".format(','.join(interpretation_variables)))]

    response = call_llm_with_pydantic_output(messages, CategoricalVariables)  # 调用LLM分析变量类型
    if response['parsing_error'] == None:  # LLM成功响应
        variables_analyzing_result = response['parsed']  # LLM给出的类别型变量列表类
        if variables_analyzing_result.categorical_variables is not None:
            for var_name in variables_analyzing_result.categorical_variables:
                if var_name in interpretation_variables:  # 如果变量位于解释变量中
                    categorical_variables_list.append(var_name)
        return True, categorical_variables_list
    else:
        return False, None
