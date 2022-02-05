import datetime
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def transrate_abnormal_value(df):
    df["blood_pressure"] = df["blood_pressure"].apply(lambda x: np.nan if x == 0 else x)
    df["skin_thickness"] = df["skin_thickness"].apply(lambda x: np.nan if x == 0 else x)
    df["insulin"] = df["insulin"].apply(lambda x: np.nan if x == 0 else x)
    df["bmi"] = df["bmi"].apply(lambda x: np.nan if x < 1 else x)
    return df


def preprocess_train(train_X):
    train_X.drop("id", axis=1, inplace=True)
    train_X = transrate_abnormal_value(train_X)
    imputer = SimpleImputer(strategy="median")
    train_X = pd.DataFrame(imputer.fit_transform(train_X), columns=train_X.columns)
    return train_X, imputer


def preprocess_test(test_X, imputer):
    test_X.drop("id", axis=1, inplace=True)
    test_X = transrate_abnormal_value(test_X)
    test_X = pd.DataFrame(imputer.transform(test_X), columns=test_X.columns)
    return test_X


def train_model(train_X, train_y, C):
    model = LogisticRegression(C=C).fit(train_X, train_y)
    return model


def evaluate_model(test_X, test_y, model):
    pred_y = model.predict_proba(test_X)[:, 1]
    fpr, tpr, _ = roc_curve(test_y, pred_y)
    return auc(fpr, tpr)


def output(
    output_dir, src_df, train_X, model, imputer=None, test_X=None, train_indexes=None
):
    os.makedirs(f"{output_dir}data/")
    os.makedirs(f"{output_dir}models/")
    src_output_path = f"{output_dir}data/src.pkl"
    imputer_output_path = f"{output_dir}models/imputer.pkl"
    trainX_output_path = f"{output_dir}data/preprocessed_trainX.pkl"
    testX_output_path = f"{output_dir}data/preprocessed_testX.pkl"
    model_path = f"{output_dir}models/mlmodel.pkl"
    train_indexes_path = f"{output_dir}train_indexes.txt"
    src_df.to_pickle(src_output_path)
    train_X.to_pickle(trainX_output_path)
    pickle.dump(model, open(model_path, "wb"))
    if imputer is not None:
        pickle.dump(imputer, open(imputer_output_path, "wb"))
    if test_X is not None:
        test_X.to_pickle(testX_output_path)
    if train_indexes:
        with open(train_indexes_path, mode="w") as f:
            f.write(train_indexes)


def train_and_evaluate_model(src_df, C):
    try:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        evaluate_dir = f"./evaluate/{run_id}/"
        train_df, test_df = train_test_split(src_df, test_size=0.2, random_state=0)
        train_X, train_y = train_df.iloc[:, :-1], train_df.iloc[:, -1]
        test_X, test_y = test_df.iloc[:, :-1], test_df.iloc[:, -1]
        train_X, imputer = preprocess_train(train_X)
        test_X = preprocess_test(test_X, imputer)
        model = train_model(train_X, train_y, C)
        result = evaluate_model(test_X, test_y, model)
        output(evaluate_dir, src_df, train_X, model, imputer, test_X)
        print(f"ID: {run_id}, C: {C}, AUC score: {result}")
    except:
        raise Exception


def train_and_deploy_model(run_id, C):
    try:
        deploy_dir = f"./deploy/{run_id}/"
        src_df = pd.read_pickle(f"./evaluate/{run_id}/data/src.pkl")
        train_indexes = ",".join([str(i) for i in src_df["id"].values.tolist()])
        train_X, train_y = src_df.iloc[:, :-1], src_df.iloc[:, -1]
        train_X, imputer = preprocess_train(train_X)
        model = train_model(train_X, train_y, C)
        output(deploy_dir, src_df, train_X, model, imputer, train_indexes=train_indexes)
        print("deploy finished")
    except:
        raise Exception


def predict(input_X, imputer, model):
    input_X = preprocess_test(input_X, imputer)
    input_y_proba = model.predict_proba(input_X)[:, 1]
    input_y_pred = model.predict(input_X)
    return input_y_proba, input_y_pred
