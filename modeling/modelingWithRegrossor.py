import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def load_data():
    data = pd.read_csv("datasets/winequalityN.csv")
    df = data.copy()
    return df


def drop_na(dataframe):
    return dataframe.dropna(axis=0)


def grab_col_names(dataframe):
    float_cols = [col for col in dataframe.columns if dataframe[col].dtype == "float"]
    obj_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    int_cols = [col for col in dataframe.columns if dataframe[col].dtype == "int"]
    return float_cols, obj_cols, int_cols
    

def standarditaziton(dataframe, cols):
    scaler = StandardScaler()
    dataframe[cols] = scaler.fit_transform(dataframe[cols])
    return dataframe
    

def encoding(dataframe, col_name):
    labels = {"red":0, "white":1}
    dataframe[col_name] = dataframe[col_name].map(labels)
    return dataframe


def base_models(dataframe,cv=3):
    X = dataframe.drop("quality", axis=1)
    y = dataframe["quality"]
    metrics = ["neg_mean_squared_error","neg_mean_absolute_error", "r2"]
    print("Base Models....")
    classifiers = [('LR', LinearRegression()),
                   ("Ridge", Ridge(alpha=0.1)),
                   ("XGBM", XGBRegressor()),
                   ("LGBM", LGBMRegressor(verbose=-1)),
                   ("ElasticNey", ElasticNet()) 
                   ]
    for metric in metrics:
        for name, classifier in classifiers:
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=metric)
            print(f"{metric}: {round(cv_results['test_score'].mean(), 4)} ({name})")
