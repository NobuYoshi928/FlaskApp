import datetime
import os
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

train_df = pd.read_csv("./ml/data/train1.csv")
userinput_df = pd.read_csv("./ml/data/train2.csv")
model = pickle.load(open("./ml/deploy/20220108055328/models/mlmodel.pkl", "rb"))
pgconfig = {
    "host": "db",
    "port": os.environ["PG_PORT"],
    "database": os.environ["PG_DATABASE"],
    "user": os.environ["PG_USER"],
    "password": os.environ["PG_PASSWORD"],
}
dsl = "postgresql://{user}:{password}@{host}:{port}/{database}".format(**pgconfig)
conn = psycopg2.connect(**pgconfig)


def preprocessing(df):
    X = df.iloc[:, :-1].drop("index", axis=1, inplace=False)
    y = df.iloc[:, -1]
    return X, y


def create_roc_curve():
    train_X, train_y = preprocessing(train_df)
    userinput_X, userinput_y = preprocessing(userinput_df)
    train_y_score = model.predict_proba(train_X)[:, 1]
    userinput_y_score = model.predict_proba(userinput_X)[:, 1]
    train_fpr, train_tpr, train_thresholds = roc_curve(train_y, train_y_score)
    userinput_fpr, userinput_tpr, userinput_thresholds = roc_curve(
        userinput_y, userinput_y_score
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train_fpr, y=train_tpr, name="train data", fill="tozeroy")
    )
    fig.add_trace(
        go.Scatter(
            x=userinput_fpr, y=userinput_tpr, name="predict data", fill="tozeroy"
        )
    )
    fig.update_layout(
        title="ROC curve",
        width=700,
        height=500,
    )
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return fig


def show_feature_importance():
    X, _ = preprocessing(train_df)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=X.columns, y=model.coef_.flatten()))
    fig.update_layout(
        title="変数重要度（回帰係数）",
        width=700,
        height=500,
    )
    fig.update_xaxes(title="features")
    return fig


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    children=[
        html.H1(children="糖尿病予測アプリ"),
        html.Div(
            children=[
                dcc.Graph(
                    id="example-graph1",
                    figure=show_feature_importance(),
                ),
                dcc.Graph(
                    id="example-graph",
                    figure=create_roc_curve(),
                ),
            ]
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=5050, debug=True)
