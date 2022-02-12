import os

import pymysql.cursors
from sqlalchemy import create_engine


def get_parameters(param_key):
    ssm = boto3.client("ssm", region_name="ap-northeast-1")
    response = ssm.get_parameters(
        Names=[
            param_key,
        ],
        WithDecryption=True,
    )
    return response["Parameters"][0]["Value"]


def connect():
    if os.environ["APP_ENV"] == "prd":
        import boto3

        config = {
            "host": get_parameters("MYSQL_HOST"),
            "port": int(get_parameters("MYSQL_PORT")),
            "database": get_parameters("MYSQL_DATABASE"),
            "user": get_parameters("MYSQL_USER"),
            "password": get_parameters("MYSQL_PASSWORD"),
        }
    else:
        config = {
            "host": os.environ["MYSQL_HOST"],
            "port": int(os.environ["MYSQL_PORT"]),
            "database": os.environ["MYSQL_DATABASE"],
            "user": os.environ["MYSQL_USER"],
            "password": os.environ["MYSQL_PASSWORD"],
        }
    conn = pymysql.connect(**config)
    engine = create_engine(
        "mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8".format(
            **config
        )
    )
    return conn, engine


def execute(conn, sql):
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    except:
        print("execution failed")
        conn.rollback()


def fetch_all(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def load_from_csv(conn, table_name, file_path, columns):
    try:
        with conn.cursor() as cur:
            with open(f"{file_path}", mode="r", encoding="utf-8") as f:
                cur.copy_from(f, f"{table_name}", sep=",", columns=columns)
        conn.commit()
    except:
        conn.rollback()
