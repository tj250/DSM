import psycopg2

'''
访问postgres数据库
'''


class PostgresAccess:
    @staticmethod
    def get_db_conn():
        # Configure database connection parameters
        conn_params = {
            "dbname": "dsm",
            "user": "postgres",
            "password": "123456",
            "host": "192.168.1.71"
        }

        # Connecting to a database
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = False  # 关闭自动提交
        return conn
