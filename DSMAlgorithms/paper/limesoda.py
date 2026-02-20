import os
import pandas as pd

base_dir = r'D:\论文&教材&书籍 编写等\地理数据挖掘智能体研究论文\dataset\LimeSoDa\data'

def load_dataset(dataset_name):
    dataset_file = os.path.join(os.path.join(base_dir, dataset_name), dataset_name + '_dataset.csv')
    coordinate_file = os.path.join(os.path.join(base_dir, dataset_name), dataset_name + '_coordinates.csv')
    ds_df = pd.read_csv(dataset_file)
    coord_df = pd.read_csv(coordinate_file)
    full_df = pd.concat([ds_df, coord_df], axis='columns')
    print(str(len(full_df)))

load_dataset('BB.250')