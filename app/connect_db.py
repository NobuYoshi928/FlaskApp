import os

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

pgconfig = {
    "host": "db",
    "port": os.environ["PG_PORT"],
    "database": os.environ["PG_DATABASE"],
    "user": os.environ["PG_USER"],
    "password": "padawan12345",
}
dsl = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**pgconfig)


def main():
    # pd.read_sql用のコネクタ
    conn = psycopg2.connect(**pgconfig)
    # pd.to_sql用のcreate engine
    engine = create_engine(dsl)


if __name__ == "__main__":
    main()
