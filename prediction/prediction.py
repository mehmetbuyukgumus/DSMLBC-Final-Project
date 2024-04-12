from joblib import load
import pandas as pd
####
from data_preparation.data_prep import *
from sklearn.model_selection import train_test_split
from data_preparation.data_prep import load_data

def final_decision(given_model, X_test, path, dataframe):
    estimators = load("modeling/prediction-table.joblib")
    if given_model == "OneVs":
        model = estimators["model"][0]
        result_onevs = model.predict(X_test)
        dataframe["result"] = pd.Series(result_onevs)
        dataframe.to_excel("prediction/data.xlsx")
        print(result_onevs)
        return result_onevs
    elif given_model == "Decision Tree":
        model = estimators["model"][1]
        result_dec_tree = model.predict(X_test)
        dataframe["result"] = pd.Series(result_dec_tree)
        dataframe.to_excel("prediction/data.xlsx")
        print(result_dec_tree)
        return result_dec_tree
    elif given_model == "KNN":
        model = estimators["model"][2]
        result_knn = model.predict(X_test)
        dataframe["result"] = pd.Series(result_knn)
        dataframe.to_excel("prediction/data.xlsx")
        print(result_knn)
        return result_knn
    elif given_model == "SVM":
        model = estimators["model"][3]
        result_svm = model.predict(X_test)
        dataframe["result"] = pd.Series(result_svm)
        dataframe.to_excel("prediction/data.xlsx")
        print(result_svm)
        return result_svm
    
    