from rasterio.sample import sample_gen
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from shapely.geometry import Point, Polygon
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import random
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import os
import rasterio

class SpatialDataAugmentation:
    """
    空间数据增强类，结合局部和全局特征进行智能数据增强
    """

    def __init__(self, points_gdf, target_columns):
        """
        初始化空间数据增强器

        参数:
            points_gdf: 原始点位GeoDataFrame
            target_columns: 目标属性列
        """
        self.points_gdf = points_gdf.copy()
        self.target_columns = target_columns
        self.scaler = StandardScaler()

        # 提取坐标和属性数据
        self.coords = np.array([[p.x, p.y] for p in points_gdf.geometry])

        # 只使用有效的数值属性进行分析
        valid_columns = []
        for col in target_columns:
            if col in points_gdf.columns and pd.api.types.is_numeric_dtype(points_gdf[col]):
                valid_columns.append(col)

        self.valid_columns = valid_columns
        if valid_columns:
            # 移除包含NaN的行进行聚类分析
            valid_data = points_gdf[valid_columns].dropna()
            if len(valid_data) > 0:
                self.attributes = self.scaler.fit_transform(valid_data)
                self.valid_indices = valid_data.index.tolist()
            else:
                self.attributes = None
                self.valid_indices = []
        else:
            self.attributes = None
            self.valid_indices = []

    def _global_clustering(self, n_clusters=5):
        """
        基于全局特征进行聚类分析
        """
        if self.attributes is None or len(self.attributes) < n_clusters:
            return None

        # 结合空间坐标和属性特征进行聚类
        valid_coords = self.coords[self.valid_indices]

        # 标准化坐标
        coord_scaler = StandardScaler()
        normalized_coords = coord_scaler.fit_transform(valid_coords)

        # 组合空间和属性特征
        combined_features = np.hstack([normalized_coords, self.attributes])

        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(combined_features)

        return cluster_labels, kmeans.cluster_centers_

    def _local_neighborhood_analysis(self, k=5):
        """
        基于局部邻域进行分析
        """
        # 使用KNN找到每个点的邻居
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(self.coords)), algorithm='ball_tree')
        nbrs.fit(self.coords)

        distances, indices = nbrs.kneighbors(self.coords)

        return distances, indices

    def _calculate_spatial_weights(self, point_idx, neighbor_indices, distances):
        """
        计算空间权重，距离越近权重越大
        """
        # 排除自身
        neighbor_indices = neighbor_indices[1:]
        distances = distances[1:]

        # 避免除零错误
        distances = np.maximum(distances, 1e-10)

        # 反距离权重
        weights = 1.0 / distances
        weights = weights / np.sum(weights)

        return neighbor_indices, weights

    def _generate_augmented_point(self, original_idx, cluster_info=None, neighbor_info=None,
                                  augmentation_factor=0.3, max_distance=100):
        """
        生成增强点位

        参数:
            original_idx: 原始点索引
            cluster_info: 聚类信息
            neighbor_info: 邻域信息
            augmentation_factor: 增强因子，控制新点与原点的相似度
            max_distance: 最大空间偏移距离
        """
        # random.seed(self.random_state)
        original_point = self.points_gdf.iloc[original_idx]
        original_coord = self.coords[original_idx]

        # 复制原始点属性
        new_point = original_point.copy()
        new_point.name = None

        # 空间偏移：结合局部和全局信息
        if neighbor_info is not None:
            neighbor_indices, weights = neighbor_info

            # 基于邻居位置计算偏移方向
            neighbor_coords = self.coords[neighbor_indices]
            weighted_center = np.average(neighbor_coords, weights=weights, axis=0)

            # 向邻居中心方向偏移，但加入随机性
            direction = weighted_center - original_coord
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 0:
                direction = direction / direction_norm
            else:
                direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                direction = direction / np.linalg.norm(direction)
        else:
            # 随机方向
            direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            direction = direction / np.linalg.norm(direction)

        # 随机距离，但受max_distance限制
        distance = random.uniform(10, max_distance) * augmentation_factor

        # 计算新坐标
        new_coord = original_coord + direction * distance
        new_point.geometry = Point(new_coord[0], new_coord[1])

        # 属性插值：结合局部邻域信息
        if neighbor_info is not None and self.valid_columns:
            neighbor_indices, weights = neighbor_info

            for col in self.valid_columns:
                if col in new_point.index and not pd.isna(original_point[col]):
                    # 获取邻居的属性值
                    neighbor_values = []
                    neighbor_weights = []

                    for i, neighbor_idx in enumerate(neighbor_indices):
                        neighbor_val = self.points_gdf.iloc[neighbor_idx][col]
                        if not pd.isna(neighbor_val):
                            neighbor_values.append(neighbor_val)
                            neighbor_weights.append(weights[i])

                    if neighbor_values:
                        neighbor_weights = np.array(neighbor_weights)
                        neighbor_weights = neighbor_weights / np.sum(neighbor_weights)

                        # 加权平均，结合原始值
                        interpolated_value = np.average(neighbor_values, weights=neighbor_weights)

                        # 与原始值混合
                        mixed_value = (1 - augmentation_factor) * original_point[col] + \
                                      augmentation_factor * interpolated_value

                        # 添加小量随机噪声
                        noise_factor = 0.05  # 5%的噪声
                        noise = random.uniform(-noise_factor, noise_factor) * abs(mixed_value)
                        new_point[col] = mixed_value + noise

        return new_point

    def extract_from_raster(self, raster_file):
        """单个栅格文件的值提取"""
        column_name = os.path.basename(raster_file).split('.')[0]

        with rasterio.open(raster_file) as src:
            # 坐标系统转换
            points_transformed = self.augmented_gdf

            # 提取栅格值
            values = []
            for point in points_transformed.geometry:
                x, y = point.x, point.y
                try:
                    val = list(sample_gen(src, [(x, y)]))[0][0]
                    # 保持原始值，不进行填补
                    values.append(val if not np.isnan(val) else np.nan)
                except:
                    values.append(np.nan)

            return column_name, values

    def augment_data(self, augmentation_ratio=2.0, max_distance=100, n_clusters=5, k_neighbors=5):
        """
        执行空间数据增强

        参数:
            augmentation_ratio: 增强比例（新点数量/原点数量）
            max_distance: 最大空间偏移距离
            n_clusters: 聚类数量
            k_neighbors: 邻居数量

        返回:
            augmented_gdf: 增强后的GeoDataFrame
        """
        # random.seed(self.random_state)
        print(f"开始智能空间数据增强，增强比例: {augmentation_ratio}")

        # 全局聚类分析
        cluster_result = self._global_clustering(n_clusters)

        # 局部邻域分析
        distances, neighbor_indices = self._local_neighborhood_analysis(k_neighbors)

        # 计算需要生成的新点数量
        n_original = len(self.points_gdf)
        n_augmented = int(n_original * augmentation_ratio)

        augmented_data = []

        # 添加原始点
        for idx, row in self.points_gdf.iterrows():
            augmented_data.append(row)

        # 生成增强点
        for i in tqdm(range(n_augmented), disable=True, desc="生成增强样本"):
            # 随机选择一个原始点作为基础
            base_idx = random.randint(0, n_original - 1)

            # 获取邻域信息
            neighbor_info = self._calculate_spatial_weights(
                base_idx, neighbor_indices[base_idx], distances[base_idx]
            )

            # 生成新点
            augmentation_factor = random.uniform(0.2, 0.5)  # 随机增强因子
            new_point = self._generate_augmented_point(
                base_idx,
                cluster_info=cluster_result,
                neighbor_info=neighbor_info,
                augmentation_factor=augmentation_factor,
                max_distance=max_distance
            )

            augmented_data.append(new_point)

        # 创建增强后的GeoDataFrame
        augmented_gdf = gpd.GeoDataFrame(augmented_data, crs=self.points_gdf.crs)

        print(f"数据增强完成：原始点 {n_original} 个，增强后 {len(augmented_gdf)} 个")
        self.augmented_gdf = augmented_gdf
        return augmented_gdf


