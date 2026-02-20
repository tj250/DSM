from mcp.server.fastmcp import FastMCP
from mcp_regression_build_model import mcp_reg_build_model
import pandas as pd
from utility import print_caller_parameters

mcp_explore_data = FastMCP("ExploreData", port=8001)


@mcp_explore_data.tool()
async def get_feature_unique_values(structured_data_file: str, column_name: str, ) -> list:
    """
    获取结构化数据中某一列的唯一值。

    Args:
        structured_data_file (str):包含待分析列的结构化数据文件。
        column_name (str):要分析的列的名称。

    Returns:
        list: 唯一值的列表。
    """
    print_caller_parameters()
    # 读取 CSV 文件
    return pd.read_csv(structured_data_file, usecols=[column_name])[column_name].unique()


@mcp_explore_data.tool()
async def read_one_feature_values(structured_data_file: str, column_name: str, ) -> pd.Series:
    """
    获取结构化数据中某一列的所有值。

    Args:
        structured_data_file (str):要读取的结构化数据文件。
        column_name (str):要读取的列的名称。

    Returns:
        pd.Series: 读取出的列的值。
    """
    print_caller_parameters()
    # 读取 CSV 文件
    return pd.read_csv(structured_data_file, usecols=[column_name])[column_name]


@mcp_explore_data.tool()
async def read_multiple_features_values(structured_data_file: str, columns_name: list[str], ) -> pd.DataFrame:
    """
    获取结构化数据中若干列的值。

    Args:
        structured_data_file (str):要读取的结构化数据文件。
        columns_name (list[str]):要读取的多个列的名称集合。

    Returns:
        pd.DataFrame: 读取出的列的值。
    """
    print_caller_parameters()
    # 读取 CSV 文件
    return pd.read_csv(structured_data_file, usecols=columns_name)
