from data_preparation.data_prep import *
from modeling.models import base_models, build_run_model
from sklearn.model_selection import train_test_split
from prediction.prediction import final_decision


def main():
    path = "datasets/winequalityN.csv"
    ### -- Verinin hazırlanması --
    df = load_data(path)
    df = label_encoder(df, "type")
    df = outlier_process(df)
    df = na_values(df)
    df = new_features(df)
    ### -- Bağımlı ve bağımsız değişkenlerin seçimi --
    X = df.drop("Cquality", axis=1)
    y = df["Cquality"]
    ### -- Test ve eğitim verilerinin hazırlanması --
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=42)
    ### -- Model Başarı Ölçümü --
    # base_models(X,y)
    ### -- Hiperparametre Optimizasyonu --
    # estimators = build_run_model(X_train, X_test, y_train, y_test)
    ### -- Final Tahmin --
    final_decision("OneVs", X, path, df)

    

if __name__ == "__main__":
    main()