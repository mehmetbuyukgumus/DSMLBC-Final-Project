from data_preparation.data_prep import *
from modeling.models import base_models


def main():
    df = load_data()
    df = label_encoder(df, "type")
    df = outlier_process(df)
    df = na_values(df)
    df = new_features(df)
    X = df.drop("Cquality", axis=1)
    y = df["Cquality"]
    base_models(df, X, y)
    


if __name__ == "__main__":
    main()