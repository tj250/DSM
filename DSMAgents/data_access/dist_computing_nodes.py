from data_access.db import PostgresAccess

'''
获取所有的可用分布式计算节点
'''


@staticmethod
def get_computing_nodes() -> list[str]:
    conn = PostgresAccess.get_db_conn()
    cursor = conn.cursor()
    cursor.execute("select url,data_mapping_local_path from computing_nodes")
    rows = cursor.fetchall()
    urls = []
    mapping_paths = []
    for row in rows:
        urls.append(row[0])
        mapping_paths.append(row[1])
    cursor.close()
    conn.close()
    return urls, mapping_paths
