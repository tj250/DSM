import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
import matplotlib.pyplot as plt
import config

'''
空间自相关性检测
'''
class AutoCorrelation():

    '''
    计算全局莫兰指数
    '''

    @staticmethod
    def compute_moran(colname: str, gdf: gpd.GeoDataFrame, geom_col: str = config.DF_GEOM_COL):
        # 1. 加载空间数据（示例为GeoJSON文件）
        # gdf = gpd.read_file('your_data.geojson')  # 替换为实际文件路径

        # 2. 构建空间权重矩阵（这里使用Queen邻接规则）
        w = Queen.from_dataframe(gdf, geom_col=geom_col,use_index=False)  # 也可用DistanceBand/其他权重

        # 3. 提取待检验的变量列（如房价）
        y = gdf[colname].values  # 替换为实际列名

        # 4. 计算全局Moran's I
        moran = Moran(y, w, permutations=999)  # 999次蒙特卡洛模拟

        # 5. 输出结果
        print(f"Moran's I 值: {moran.I:.4f}")
        print(f"P值: {moran.p_sim:.4f}")
        print(f"标准化统计量Z: {moran.z_sim:.4f}")

        # 6. 结果解读
        if moran.p_sim < 0.05:
            print("空间自相关显著（p < 0.05）")
        else:
            print("空间自相关不显著")
        return float(moran.I), float(moran.p_sim)

    '''
    计算局部莫兰指数
    '''

    def compute_moran_local(colname: str, gdf: gpd.GeoDataFrame):
        # 1. 加载空间数据（示例为GeoJSON文件）
        # gdf = gpd.read_file('your_data.geojson')  # 替换为实际文件路径

        # 2. 构建空间权重矩阵（这里使用Queen邻接规则）
        w = Queen.from_dataframe(gdf)  # 也可用DistanceBand/其他权重
        w.transform = 'r'  # 行标准化

        # 3. 计算局部莫兰指数（999次置换检验）
        moran_loc = Moran_Local(gdf[colname], w, permutations=999)

        # 4. 结果输出与可视化
        print("局部莫兰指数值:", moran_loc.Is)
        print("p值:", moran_loc.p_sim)
        print("聚类类型:", moran_loc.q)  # 1=HH, 2=LL, 3=HL, 4=LH

        # 绘制空间聚类地图
        fig, ax = plt.subplots(figsize=(12, 8))
        gdf.assign(cluster=moran_loc.q).plot(
            column='cluster',
            cmap='RdYlBu',
            edgecolor='k',
            legend=True,
            ax=ax
        )
        ax.set_title("Local Moran's I Cluster Map (GDP)")
        plt.show()

        return moran_loc.Is, moran_loc.p_sim, moran_loc.q

    '''
    检测数据是否具有空间自相关
    colname:gdf中的待检测数据列名称
    gdf:GeoDataFrame格式的数据
    '''

    @staticmethod
    def check(colname: str, gdf: gpd.GeoDataFrame, geom_col: str = config.DF_GEOM_COL, p=0.05) -> bool:
        moran, pvalue = AutoCorrelation.compute_moran(colname, gdf, geom_col)
        return pvalue < p

