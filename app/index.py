import pickle

import pandas as pd
from flask import Flask

app = Flask(__name__)


def predict(param):
    features = param["feature"]
    df = pd.DataFrame.from_dict(features)
    df.drop("index", axis=1, inplace=True)
    X = df.iloc[:, :-1]
    y_true = df.iloc[:, -1].to_list()[0]
    loaded_model = pickle.load(open("../ml/models/newest_model.pkl", "rb"))
    y_predict = loaded_model.predict(X)[0]
    return y_true, y_predict


@app.route("/")
def index():
    params = {
        "feature": {
            "index": [2181],
            "Pregnancies": [4],
            "Glucose": [120],
            "BloodPressure": [80],
            "SkinThickness": [0],
            "Insulin": [0],
            "BMI": [49.89446445],
            "DiabetesPedigreeFunction": [0.264578365],
            "Age": [29],
            "Outcome": [1],
        }
    }
    y_true, y_predict = predict(params)
    return f"<h1>正解: {y_true}  予測: {y_predict}</h1>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50, debug=True)
