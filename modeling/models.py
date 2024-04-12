import pandas as pd
from data_preparation.data_prep import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from modeling.classifiers import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from warnings import filterwarnings
from joblib import dump
filterwarnings("ignore")


def base_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.20)
    for name, classifier, params in classifiers:
        model_object = classifier
        model = model_object.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(" ")
        print(f"########## {name} ##########")
        print("Confusion Matrix")
        print("---------- Confusion Matrix ----------")
        print(confusion_matrix(y_test, y_pred))
        print("---------- Classification Report ----------")
        print(classification_report(y_test, y_pred))
        # print("---------- Accuracy Score ----------")
        # print(accuracy_score(y_test, y_pred))
        # print("---------- F1 Score ----------")
        # print(f1_score(y_test, y_pred, average="weighted"))
        # print("---------- Recall Score ----------")
        # print(recall_score(y_test, y_pred, average="weighted"))
        # print("---------- Precision Score ----------")
        # print(precision_score(y_test, y_pred, average="weighted"))
        # print("---------- Roc_Auc_Score Score ----------")
        # print(roc_auc_score(y_test, y_pred, multi_class="ovo"))


def build_run_model(X_train, X_test, y_train, y_test):
    table_name = []
    table_model_obj = []
    table_accuracy = []
    table_f1_score = []
    table_recall_score = []
    table_precision_score = []
    table_best_params = []
    for name, model_obj, params in classifiers:
        table_name.append(name)
        table_model_obj.append(model_obj)
        table_best_params
        model = model_obj
        model = model.fit(X_train,y_train)
        gs_best = GridSearchCV(model,params, cv=3, n_jobs=-1, verbose=0).fit(X_train, y_train)
        final_model = model.set_params(**gs_best.best_params_).fit(X_train,y_train)
        y_pred = final_model.predict(X_test)
        table_best_params.append(gs_best.best_params_)
        table_accuracy.append(accuracy_score(y_test, y_pred))
        table_f1_score.append(f1_score(y_test, y_pred, average="weighted"))
        table_recall_score.append(recall_score(y_test, y_pred, average="weighted"))
        table_precision_score.append(precision_score(y_test, y_pred, average="weighted"))
    table = pd.DataFrame(columns=["name", "accuracy","f1_score","recall_score", "precision_score"])
    table["name"] =table_name
    table["model"] = table_model_obj
    table["best_params"] = table_best_params
    table["accuracy"] = table_accuracy
    table["f1_score"] = table_f1_score
    table["recall_score"] = table_recall_score
    table["precision_score"] = table_precision_score
    table = table.sort_values(by= "f1_score", ascending=False)
    dump(table, "modeling/prediction-table.joblib")
    
    
