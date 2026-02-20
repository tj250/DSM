import datetime, psycopg2, json, os
import pickle,dill
from dataclasses import dataclass
from agents.utils.views import BMActiveState
from data_access.db import PostgresAccess
from data_access.task import TaskInfo
import time
import uuid
from agents.data_structure.base_data_structure import RegressionModelInfo
from DSMAlgorithms.base.dsm_base_model import AlgorithmsType
from DSMAlgorithms.base.base_data_structure import DataDistributionType
from agents.utils.views import UncertaintyType

'''
不确定性制图的数据访问
'''


class UncertaintyMappingDataAccess:
    '''
    查询某个算法的某类不确定性制图的ID
    '''

    @staticmethod
    def query_uncertainty_map_id(algorithms_id: str, map_type:UncertaintyType) -> str:
        conn = PostgresAccess.get_db_conn()
        cursor = conn.cursor()
        cursor.execute("select map_id from map where algorithms_id=%s and map_type=%s", (algorithms_id,map_type.name,))
        rows = cursor.fetchall()
        map_id = rows[0][0]
        cursor.close()
        conn.close()
        return map_id