import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy import stats
import numpy as np
import config


class StructureDataDealer:
    '''
    获取结构化数据的schema
    datasource:数据源
    data_type:1-csv，2-shp,3-gdb
    '''

    @staticmethod
    def get_structed_data_schema(datasource, data_type: int):
        if data_type == 1:  # csv等结构化数据，非空间数据格式
            df = pd.read_csv(datasource)
            # 获取schema信息
            schema = {
                column: str(dtype)
                for column, dtype in df.dtypes.items()
            }
            return schema
        elif data_type == 2:
            pass

    '''
    读一个一个表格式文件中的内容,形成DataFrame
    datasource:数据源
    x_col_name:X坐标列的名称
    y_col_name:Y坐标列的名称
    
    返回值：
    是否为GeoDataFrame
    DataFrame
    '''

    @staticmethod
    def read_tabular_data(datasource):
        df = None
        if datasource.endswith('.csv'):
            df = pd.read_csv(datasource)
        if config.CSV_GEOM_COL_X in df.columns:
            # 创建包含坐标的几何对象列（假设列名为'X'和'Y'）
            df[config.DF_GEOM_COL] = df.apply(lambda row: Point(row[config.CSV_GEOM_COL_X], row[config.CSV_GEOM_COL_Y]), axis=1)
            del df[config.CSV_GEOM_COL_X]
            del df[config.CSV_GEOM_COL_Y]

            # 转换为GeoDataFrame
            gdf = gpd.GeoDataFrame(df.copy(), geometry=config.DF_GEOM_COL)
            return True, gdf
        else:
            return False, df

    '''
    进行数据清理
    tabular_file:数据所在的tabular文件
    continuous_variables:连续型变量的列表
    exclude_exception_data:是否需要剔除包含异常值的列
    
    返回值：
    是否为GeoDataFrame
    DataFrame
    '''

    @staticmethod
    def data_cleaning(tabular_file: str, continuous_variables: list[str], exclude_exception_data=True) -> gpd.GeoDataFrame:
        _, gdf = StructureDataDealer.read_tabular_data(tabular_file)
        # 1、缺失值处理
        gdf.dropna(inplace=True)
        # 2、异常值处理(可选)
        if exclude_exception_data:  # 需要剔除异常值
            selected_types = gdf[continuous_variables]
            z_scores = stats.zscore(selected_types)
            abs_z_scores = np.abs(z_scores)
            filtered_inds = (abs_z_scores < 3).all(axis=1)
            gdf = gdf[filtered_inds]

        # 3、对连续型的列进行z-score标准化处理
        for key in gdf.columns:
            if key in continuous_variables:  # 仅针对连续型环境变量处理
                gdf[key] = (gdf[key] - gdf[key].mean()) / gdf[key].std()
        return gdf
