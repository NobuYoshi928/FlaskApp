import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import pymysql.cursors

conn = pymysql.connect(
    host=os.environ["MYSQL_HOST"],
    port=int(os.environ["MYSQL_PORT"]),
    user=os.environ["MYSQL_USER"],
    password=os.environ["MYSQL_PASSWORD"],
    db=os.environ["MYSQL_DB"],
    charset="utf8",
    cursorclass=pymysql.cursors.DictCursor,
)


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


def create_test_table():
    sql = """
    create table IF not exists `test_table`(
        `id`               INT(20) AUTO_INCREMENT,
        `name`             VARCHAR(20) NOT NULL,
        `created_at`       Datetime DEFAULT NULL,
        `updated_at`       Datetime DEFAULT NULL,
            PRIMARY KEY (`id`)
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
    """
    execute(conn, sql)


def count_test_table():
    sql = "select count(1) from test_table;"
    return str(fetch_all(conn, sql)[0]["count(1)"])


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash!!!"),
        html.Div(children=f"テストテーブルのレコード数は{count_test_table()}件です"),
        dcc.Graph(
            id="example-graph",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar", "name": "SF"},
                    {
                        "x": [1, 2, 3],
                        "y": [2, 4, 5],
                        "type": "bar",
                        "name": "Montréal",
                    },
                ],
                "layout": {"title": "Dash Data Visualization"},
            },
        ),
    ]
)

if __name__ == "__main__":
    create_test_table()
    app.run_server(host="0.0.0.0", port=5050, debug=True)
