import os

import psycopg2
from sqlalchemy import create_engine


def connect(env):
    if env == "dev":
        pgconfig = {
            "host": "db",
            "port": os.environ["PG_PORT"],
            "database": os.environ["PG_DATABASE"],
            "user": os.environ["PG_USER"],
            "password": os.environ["PG_PASSWORD"],
        }
    elif env == "prd":
        pgconfig = {
            "host": "db",
            "port": os.environ["PG_PORT_PRD"],
            "database": os.environ["PG_DATABASE_PRD"],
            "user": os.environ["PG_USER_PRD"],
            "password": os.environ["PG_PASSWORD_PRD"],
        }
    engine = create_engine("postgresql://{user}:{password}@{host}:{port}/{database}".format(**pgconfig))
    return psycopg2.connect(**pgconfig), engine


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
