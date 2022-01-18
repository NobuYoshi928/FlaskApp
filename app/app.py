import json
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve
from utils import db_utils, ml_utils

# デプロイパラメタから対象リソースを取得
with open("./deploy_param.json") as f:
    deploy_param = json.load(f)
index_id = deploy_param["index_id"]
imputer_id = deploy_param["imputer_id"]
model_id = deploy_param["model_id"]
imputer = pickle.load(open(f"./deploy/model/imputer_{imputer_id}.pkl", "rb"))
model = pickle.load(open(f"./deploy/model/model_{model_id}.pkl", "rb"))

# DB接続
conn, engine = db_utils.connect("dev")
train_df = pd.read_sql(
    sql="SELECT * FROM diabetes_diagnosis_results WHERE is_trained is True;", con=conn
).drop("is_trained", axis=1, inplace=False)
not_train_df = pd.read_sql(
    sql="SELECT * FROM diabetes_diagnosis_results WHERE is_trained is False;", con=conn
).drop("is_trained", axis=1, inplace=False)
predict_result_df = pd.read_sql(sql="SELECT * FROM predict_results;", con=conn)
src_columns = (
    "index",
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
    "outcome",
    "is_trained",
)
predict_result_columns = (
    "index",
    "predict_result",
    "predict_probability",
    "true_result",
    "model_id",
)
results_temp_columns = (
    "index",
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
    "predict_result",
    "predict_probability",
)


def update_tables():
    with open(f"./deploy/data/src_index_{index_id}.txt", mode="r") as f:
        trained_indexes = f.readlines()[0]
    sql = f"""
        UPDATE diabetes_diagnosis_results SET is_trained = True 
        WHERE index in ({trained_indexes})
        """
    db_utils.execute(conn, sql)
    sql = f"""
        UPDATE diabetes_diagnosis_results SET is_trained = False 
        WHERE index not in ({trained_indexes})
        """
    db_utils.execute(conn, sql)
    sql = f"""
        DELETE FROM predict_results
        WHERE index in ({trained_indexes})
        """
    db_utils.execute(conn, sql)
    sql = f"""
        DELETE FROM results_temp;
        """
    db_utils.execute(conn, sql)


def create_roc_curve():
    train_X, train_y = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    train_X, _ = ml_utils.preprocess_train(train_X)
    train_y_score = model.predict_proba(train_X)[:, 1]
    train_fpr, train_tpr, _ = roc_curve(train_y, train_y_score)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train_fpr, y=train_tpr, name="train data", fill="tozeroy")
    )
    if len(predict_result_df) > 0:
        userinput_y = predict_result_df["true_result"].values
        userinput_y_score = predict_result_df["predict_probability"].values
        userinput_fpr, userinput_tpr, _ = roc_curve(userinput_y, userinput_y_score)
        fig.add_trace(
            go.Scatter(
                x=userinput_fpr, y=userinput_tpr, name="predict data", fill="tozeroy"
            )
        )
    fig.update_layout(
        title="ROC curve",
        width=500,
        height=300,
    )
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return fig


def create_label_counts():
    df = pd.read_sql(
        sql="SELECT index, outcome, is_trained FROM diabetes_diagnosis_results;",
        con=conn,
    )
    value_counts_df = df.groupby(["outcome", "is_trained"]).count().reset_index()
    value_counts_df["outcome"] = value_counts_df["outcome"].apply(
        lambda x: "陽性" if x == 1 else "陰性"
    )
    value_counts_df["is_trained"] = value_counts_df["is_trained"].apply(
        lambda x: "train data" if x else "predict data"
    )
    fig = px.histogram(
        value_counts_df, x="is_trained", y="index", color="outcome", barmode="group"
    )
    fig.update_layout(
        title="True Lavel Counts",
        width=500,
        height=300,
    )
    fig.update_xaxes(title="Data Type")
    fig.update_yaxes(title="Counts")
    return fig


def show_features_histgram():
    fig = make_subplots(rows=2, cols=4)
    columns = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
    ]
    for i, column in enumerate(columns):
        fig.add_trace(
            go.Histogram(x=train_df[column].values), row=(i // 4 + 1), col=(i % 4 + 1)
        )
        fig.update_xaxes(title=column, row=(i // 4 + 1), col=(i % 4 + 1))
    for i, column in enumerate(columns):
        fig.add_trace(
            go.Histogram(x=not_train_df[column].values),
            row=(i // 4 + 1),
            col=(i % 4 + 1),
        )
        fig.update_xaxes(title=column, row=(i // 4 + 1), col=(i % 4 + 1))
    fig.update_layout(
        title="Features Histgram",
        height=500,
        width=1000,
    )
    return fig


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    children=[
        html.H1(children="糖尿病予測アプリ"),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H3("測定値を入力してください"),
                        html.Div(
                            [
                                "index:",
                                dcc.Input(
                                    id="select-index",
                                    type="number",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                "pregnancies:",
                                dcc.Input(
                                    id="select-pregnancies",
                                    type="number",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                "glucose:",
                                dcc.Input(
                                    id="select-glucose",
                                    type="number",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                "blood_pressure:",
                                dcc.Input(
                                    id="select-blood_pressure",
                                    type="number",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                "skin_thickness:",
                                dcc.Input(
                                    id="select-skinthickness",
                                    type="number",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                "insulin:",
                                dcc.Input(
                                    id="select-insulin",
                                    type="number",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                "bmi:",
                                dcc.Input(
                                    id="select-bmi",
                                    type="number",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                "diabetes_pedigree_function:",
                                dcc.Input(
                                    id="select-diabetes_pedigree_function",
                                    type="number",
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                "age:",
                                dcc.Input(
                                    id="select-age",
                                    type="number",
                                ),
                            ]
                        ),
                        html.Button(
                            id="predict-button",
                            n_clicks=0,
                            children="予測を開始する",
                            style={"background": "#DDDDDD"},
                        ),
                        html.H3(""),
                        html.H3(id="predict_result", children="（ここに予測結果が表示されます）"),
                        html.H3(""),
                        html.H3(id="evaluate_result", children=""),
                        html.H3(""),
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id="select_result",
                                    options=[
                                        {"label": "はい", "value": "yes"},
                                        {"label": "いいえ", "value": "no"},
                                    ],
                                ),
                                html.Button(
                                    id="register-button",
                                    n_clicks=0,
                                    children="結果を登録する",
                                    style={"background": "#DDDDDD"},
                                ),
                            ]
                        ),
                        html.H3(""),
                        html.H3(id="register_done_message"),
                    ],
                    style={
                        "float": "left",
                        "background": "#FFE4B5",
                        "width": "400px",
                        "height": "800px",
                    },
                ),
            ],
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Graph(
                            id="roc_curve",
                            figure=create_roc_curve(),
                            style={"float": "left"},
                        ),
                        dcc.Graph(
                            id="true_label_counts",
                            figure=create_label_counts(),
                            style={"float": "right"},
                        ),
                    ],
                ),
                dcc.Graph(
                    id="histgram",
                    figure=show_features_histgram(),
                    style={"clear": "both"},
                ),
            ],
            style={
                "float": "right",
                "background": "#FFE4B5",
                "width": "1000px",
                "height": "800px",
            },
        ),
    ]
)


@app.callback(
    Output(component_id="predict_result", component_property="children"),
    [Input(component_id="predict-button", component_property="n_clicks")],
    [
        State(component_id="select-index", component_property="value"),
        State(component_id="select-pregnancies", component_property="value"),
        State(component_id="select-glucose", component_property="value"),
        State(component_id="select-blood_pressure", component_property="value"),
        State(component_id="select-skinthickness", component_property="value"),
        State(component_id="select-insulin", component_property="value"),
        State(component_id="select-bmi", component_property="value"),
        State(
            component_id="select-diabetes_pedigree_function", component_property="value"
        ),
        State(component_id="select-age", component_property="value"),
    ],
)
def predict(
    n_clicks,
    index,
    pregnancies,
    glucose,
    blood_pressure,
    skinthickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
):
    if n_clicks >= 1:
        input_dict = {
            "index": [index],
            "pregnancies": [pregnancies],
            "glucose": [glucose],
            "blood_pressure": [blood_pressure],
            "skin_thickness": [skinthickness],
            "insulin": [insulin],
            "bmi": [bmi],
            "diabetes_pedigree_function": [diabetes_pedigree_function],
            "age": [age],
        }
        input_df = pd.DataFrame(input_dict)
        input_y_proba, input_y_pred = ml_utils.predict(input_df, imputer, model)
        input_df["predict_result"] = input_y_pred[0]
        input_df["predict_probability"] = input_y_proba[0]
        input_df["index"] = index
        input_df.reindex(index=results_temp_columns)
        try:
            input_df.to_sql("results_temp", con=engine, if_exists="append", index=False)
            return (
                f'患者番号{index}は{"陽性" if input_y_pred[0] == 1 else "陰性"}です。予測結果は正しいですか？'
            )
        except:
            return (
                f'患者番号{index}は{"陽性" if input_y_pred[0] == 1 else "陰性"}です。予測結果は正しいですか？'
            )


@app.callback(
    Output(component_id="register_done_message", component_property="children"),
    [Input(component_id="register-button", component_property="n_clicks")],
    [
        State(component_id="select-index", component_property="value"),
        State(component_id="select_result", component_property="value"),
    ],
)
def register(n_clicks, index, is_prediction_true):
    if n_clicks >= 1:
        temp_df = pd.read_sql(
            sql=f"SELECT * FROM results_temp WHERE index = {index};", con=conn
        )
        predict_result = temp_df["predict_result"][0]
        if is_prediction_true == "no":
            true_result = 0 if predict_result == 1 else 1
        else:
            true_result = predict_result
        # diabetes_diagnosis_resultsテーブルにレコードを追加
        src_record_df = temp_df.copy()
        src_record_df.drop(
            ["predict_probability", "predict_result"], axis=1, inplace=True
        )
        src_record_df["outcome"] = true_result
        src_record_df["is_trained"] = False
        src_record_df.reindex(index=src_columns)
        # predict_resultsテーブルにレコードを追加
        temp_df = temp_df[["index", "predict_result", "predict_probability"]]
        temp_df["true_result"] = true_result
        temp_df["model_id"] = model_id
        temp_df.reindex(index=predict_result_columns)
        try:
            src_record_df.to_sql(
                "diabetes_diagnosis_results",
                con=engine,
                if_exists="append",
                index=False,
            )
            temp_df.to_sql(
                "predict_results", con=engine, if_exists="append", index=False
            )
            return f"登録しました"
        except:
            return "登録済みです"


if __name__ == "__main__":
    update_tables()
    app.run_server(host="0.0.0.0", port=5050, debug=True)
