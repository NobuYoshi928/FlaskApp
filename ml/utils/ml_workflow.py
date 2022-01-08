import datetime
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def transrate_abnormal_value(df):
    df["BloodPressure"] = df["BloodPressure"].apply(lambda x: np.nan if x == 0 else x)
    df["SkinThickness"] = df["SkinThickness"].apply(lambda x: np.nan if x == 0 else x)
    df["Insulin"] = df["Insulin"].apply(lambda x: np.nan if x == 0 else x)
    df["BMI"] = df["BMI"].apply(lambda x: np.nan if x < 1 else x)
    return df


def preprocess_train(df, output_dir=None):
    df.drop("index", axis=1, inplace=True)
    df = transrate_abnormal_value(df)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    imputer = SimpleImputer(strategy="median")
    df = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    df["Outcome"] = y.values
    imputer_output_path = f"{output_dir}models/imputer.pkl"
    train_output_path = f"{output_dir}data/preprocessed_train.pkl"
    pickle.dump(imputer, open(imputer_output_path, "wb"))
    df.to_pickle(train_output_path)
    return df, imputer


def preprocess_test(df, imputer, output_dir):
    df.drop("index", axis=1, inplace=True)
    df = transrate_abnormal_value(df)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    df = pd.DataFrame(imputer.transform(X), columns=X.columns)
    df["Outcome"] = y.values
    test_output_path = f"{output_dir}data/preprocessed_test.pkl"
    df.to_pickle(test_output_path)
    return df


def train_model(train_df, C, output_dir):
    train_X, train_y = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    model = LogisticRegression(C=C).fit(train_X, train_y)
    model_path = f"{output_dir}models/mlmodel.pkl"
    pickle.dump(model, open(model_path, "wb"))
    return model


def evaluate_model(model, test_df):
    test_X, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    pred_y = model.predict_proba(test_X)[:, 1]
    fpr, tpr, _ = roc_curve(test_y, pred_y)
    return auc(fpr, tpr)


def train_and_evaluate_model(df, C):
    try:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        evaluate_dir = f"./evaluate/{run_id}/"
        os.makedirs(f"{evaluate_dir}data/")
        os.makedirs(f"{evaluate_dir}models/")
        src_output_path = f"{evaluate_dir}data/src.pkl"
        df.to_pickle(src_output_path)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
        train_df, imputer = preprocess_train(train_df, evaluate_dir)
        test_df = preprocess_test(test_df, imputer, evaluate_dir)
        model = train_model(train_df, C, evaluate_dir)
        result = evaluate_model(model, test_df)
        print(f"ID: {run_id}, C: {C}, AUC score: {result}")
    except:
        raise Exception


def train_and_deploy_model(run_id, C):
    try:
        deploy_dir = f"./deploy/{run_id}/"
        os.makedirs(f"{deploy_dir}data/")
        os.makedirs(f"{deploy_dir}models/")
        src_df = pd.read_pickle(f"./evaluate/{run_id}/data/src.pkl")
        train_df, imputer = preprocess_train(src_df, deploy_dir)
        train_model(train_df, C, deploy_dir)
        print("deploy finished")
    except:
        raise Exception
