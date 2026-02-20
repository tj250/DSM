import numpy as np
import pandas as pd
from scipy.stats import f, ncf


def load_data(file_path, sheet_name):
    """加载Excel数据"""
    return pd.read_excel(file_path, sheet_name=sheet_name)


def merge_data(df_vars, df_wy):
    """合并自变量和因变量数据"""
    return pd.merge(df_vars, df_wy, on='Id', how='inner')


def check_data(df, y, factors):
    """检查数据集中是否包含必要的列和是否存在缺失值"""
    missing_columns = [col for col in [y] + factors if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the DataFrame: {', '.join(missing_columns)}")
    if df.isnull().any().any():
        raise ValueError("Data contains null values.")


def clean_data(df):
    """移除含有缺失值的行"""
    if df.isnull().any().any():
        print("Data contains null values, removing rows with nulls.")
        df = df.dropna()
    return df


def cal_ssw(df, y, factor):
    """计算组内和组间方差"""

    def _cal_ssw(group_df):
        length = group_df.shape[0]
        if length == 1:
            return 0, np.square(group_df[y].values[0]), group_df[y].values[0]
        else:
            return (length - 1) * group_df[y].var(ddof=1), np.square(group_df[y].mean()), np.sqrt(length) * group_df[
                y].mean()

    df_grouped = df.groupby(factor).apply(_cal_ssw)
    ssw_sum = df_grouped.sum(axis=0)
    return ssw_sum[0], ssw_sum[1], ssw_sum[2]


def cal_q(df, y, factor):
    """计算Q统计量和相关值"""
    ssw, lamda_1st_sum, lamda_2nd_sum = cal_ssw(df, y, factor)
    total_var = (df.shape[0] - 1) * df[y].var(ddof=1)
    q = 1 - ssw / total_var
    return q, lamda_1st_sum, lamda_2nd_sum


'''
地理探测器的因子检测
df:要检测的数据(DataFrame格式)
y:因变量
factors:自变量
'''


def factor_detector(df: pd.DataFrame, y: str, factors: list[str]):
    check_data(df, y, factors)
    results = pd.DataFrame(index=["q statistic", "p value"], columns=factors)
    for factor in factors:
        q, lamda_1st_sum, lamda_2nd_sum = cal_q(df, y, factor)
        y_variance = df[y].var(ddof=1)
        if y_variance == 0:
            print(f"{y} 的方差为零，可能导致无法进行有效的统计分析。")
            lamda = float('nan')
        else:
            lamda = (lamda_1st_sum - np.square(lamda_2nd_sum) / df.shape[0]) / y_variance
            if np.isinf(lamda) or np.isnan(lamda):
                print("无效的 lambda 值，检查数据是否含有异常值。")
                lamda = float('nan')
        F_value = df.shape[0] * q / (1 - q) if q != 1 else float('inf')
        p_value = ncf.sf(F_value, df[factor].nunique() - 1, df.shape[0] - df[factor].nunique(),
                         nc=lamda) if not np.isnan(lamda) else 1
        results.loc["q statistic", factor] = q
        results.loc["p value", factor] = p_value
    return results


def interaction_detection(df, y, factors):
    """进行交互作用探测"""
    results = pd.DataFrame(index=factors, columns=factors)
    for i, factor1 in enumerate(factors):
        for j, factor2 in enumerate(factors[i + 1:], i + 1):
            q1, _, _ = cal_q(df, y, factor1)
            q2, _, _ = cal_q(df, y, factor2)
            q_interact, _, _ = cal_q(df, y, [factor1, factor2])
            interaction_effect = q_interact - (q1 + q2)
            results.loc[factor1, factor2] = interaction_effect
            results.loc[factor2, factor1] = interaction_effect
    return results


def analyze_years(file_path, years, factors, dependent_vars):
    """分析多年数据"""
    for year in years:
        sheet_vars = f'zbl{year}'
        sheet_wy = f'wy_css_{year}'
        df_vars = load_data(file_path, sheet_vars)
        df_wy = load_data(file_path, sheet_wy)
        df_merged = merge_data(df_vars, df_wy)
        df_merged = clean_data(df_merged)

        for dv in dependent_vars:
            results = factor_detector(df_merged, dv, factors)
            interaction_results = interaction_detection(df_merged, dv, factors)

            # 保存结果
            with pd.ExcelWriter(f'F:/QH_EF/3/geo/results_{dv}_{year}.xlsx') as writer:
                results.to_excel(writer, sheet_name='Factor Detection')
                interaction_results.to_excel(writer, sheet_name='Interaction Detection')

# file_path = 'F:/QH_EF/3/gwr/yw6000.xlsx'
# years = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
# factors = ['TEM', 'PRE', 'NDVI', 'GI', 'GDP', 'POP', 'CLCD', 'Silt', 'Clay', 'DEM', 'Sand']
# dependent_vars = ['CSS', 'WY']
#
# analyze_years(file_path, years, factors, dependent_vars)
