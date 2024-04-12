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
    

# ########## KNN ##########
# Confusion Matrix
# ---------- Confusion Matrix ----------
# [[275 183  24]
#  [200 289  51]
#  [ 53 116 102]]
# ---------- Classification Report ----------
#               precision    recall  f1-score   support

#            0       0.52      0.57      0.54       482
#            1       0.49      0.54      0.51       540
#            2       0.58      0.38      0.46       271

#     accuracy                           0.52      1293
#    macro avg       0.53      0.49      0.50      1293
# weighted avg       0.52      0.52      0.51      1293


# ########## Decision Tree ##########
# Confusion Matrix
# ---------- Confusion Matrix ----------
# [[332 129  21]
#  [108 346  86]
#  [ 26  91 154]]
# ---------- Classification Report ----------
#               precision    recall  f1-score   support

#            0       0.71      0.69      0.70       482
#            1       0.61      0.64      0.63       540
#            2       0.59      0.57      0.58       271

#     accuracy                           0.64      1293
#    macro avg       0.64      0.63      0.64      1293
# weighted avg       0.64      0.64      0.64      1293


# ########## OneVs ##########
# Confusion Matrix
# ---------- Confusion Matrix ----------
# [[363 114   5]
#  [ 90 407  43]
#  [ 10 103 158]]
# ---------- Classification Report ----------
#               precision    recall  f1-score   support

#            0       0.78      0.75      0.77       482
#            1       0.65      0.75      0.70       540
#            2       0.77      0.58      0.66       271

#     accuracy                           0.72      1293
#    macro avg       0.73      0.70      0.71      1293
# weighted avg       0.73      0.72      0.72      1293


# ########## SVM ##########
# Confusion Matrix
# ---------- Confusion Matrix ----------
# [[247 235   0]
#  [181 359   0]
#  [ 55 216   0]]
# ---------- Classification Report ----------
#               precision    recall  f1-score   support

#            0       0.51      0.51      0.51       482
#            1       0.44      0.66      0.53       540
#            2       0.00      0.00      0.00       271

#     accuracy                           0.47      1293
#    macro avg       0.32      0.39      0.35      1293


########### HIPERPARAMETRE OPTIMIZASYONU ÇIKTISI ###########

#             name  accuracy  f1_score  recall_score  precision_score
# 0            KNN  0.566125  0.556063      0.566125         0.578750
# 1  Decision Tree  0.635731  0.636500      0.635731         0.638392
# 2          OneVs  0.703790  0.703747      0.703790         0.708548
# 3            SVM  0.466357  0.408131      0.466357         0.375697