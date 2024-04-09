import pandas as pd
import numpy as np
from utils.helper_functions import grab_col_names, replace_with_thresholds
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def load_data():
    dataframe = pd.read_csv("datasets/winequalityN.csv")
    return dataframe


def label_encoder(dataframe, col_name):
    encoder = LabelEncoder()
    dataframe[col_name] = encoder.fit_transform(dataframe[col_name])
    return dataframe


def outlier_process(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    for col in num_cols:
        replace_with_thresholds(dataframe, col, 0.05, 0.95)
    return dataframe


def na_values(dataframe):
    dataframe.dropna(inplace=True)
    return dataframe


def new_features(dataframe):
    dataframe["total acidity"] = dataframe["fixed acidity"] + dataframe["volatile acidity"] + dataframe["citric acid"]

    dataframe["residul sugar levels"] = pd.cut(dataframe["residual sugar"],
                        bins=[0, 1, 17, 35, 120, 1000],
                        labels=["bone_dry", "dry", "off_dry", "medium_dry", "sweet"])

    dataframe["alcohol levels"] = pd.cut(dataframe["alcohol"], bins=[0, 12.5, 13.5, 14.5, 20],
                    labels=["low", "moderately_low", "high", "very_high"])

    dataframe["white perfect pH"] = np.where((dataframe["type"] == 1) & (dataframe["pH"] >= 3) & (dataframe["pH"] <= 3.3), 1, 0)

    dataframe["red perfect pH"] = np.where((dataframe["type"] == 0) & (dataframe["pH"] >= 3.3) & (dataframe["pH"] <= 3.6), 1, 0)

    dataframe["perfect ph"] = dataframe["white perfect pH"] + dataframe["red perfect pH"]

    dataframe["Cquality"] = pd.cut(dataframe["quality"], bins=[0, 5, 6, 10], labels=["bad", "good", "perfect"])

    dataframe["residul sugar levels"] = LabelEncoder().fit_transform(dataframe["residul sugar levels"])
    dataframe["alcohol levels"] = LabelEncoder().fit_transform(dataframe["alcohol levels"])
    dataframe["Cquality"] = LabelEncoder().fit_transform(dataframe["Cquality"])

    dataframe.drop(columns=["quality", "red perfect pH", "white perfect pH"], inplace=True, axis=1)
    return dataframe


def standardization(dataframe, minmax = True):
    if minmax:
        scaler = MinMaxScaler((0,1))
        dataframe = dataframe.iloc[:,:-1]
        dataframe = scaler.fit_transform(dataframe)
        return dataframe
    else:
        scaler = StandardScaler()
        dataframe = dataframe.iloc[:,:-1]
        dataframe = scaler.fit_transform(dataframe)
        return dataframe
        
        
        




