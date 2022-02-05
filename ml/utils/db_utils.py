import os

import pymysql.cursors
from sqlalchemy import create_engine


def connect():
    config = {
        "host": os.environ['MYSQL_HOST'],
        "port": int(os.environ['MYSQL_PORT']),
        "database": os.environ['MYSQL_DATABASE'],
        "user": os.environ['MYSQL_USER'],
        "password": os.environ['MYSQL_PASSWORD'],
    }
    conn = pymysql.connect(**config)
    engine = create_engine('mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8'.format(**config))
    return conn, engine


def execute(conn, sql):
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    except:
        print('execution failed')
        conn.rollback()


def fetch_all(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def load_from_csv(conn, table_name, file_path, columns):
    try:
        with conn.cursor() as cur:
            with open(f'{file_path}', mode='r', encoding='utf-8') as f:
                cur.copy_from(f, f'{table_name}', sep=',', columns=columns)
        conn.commit()
    except:
        conn.rollback()
