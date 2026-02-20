import psycopg2
import algorithms_config

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
            "host": "host.docker.internal" if algorithms_config.DOCKER_MODE else "192.168.1.71"  # 如果处于通过Dockers发布的模式，则连接宿主机地址
        }

        # Connecting to a database
        conn = psycopg2.connect(**conn_params)
        return conn
