from data_preparation.data_prep import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from modeling.classifiers import classifiers
from warnings import filterwarnings
filterwarnings("ignore")


def base_models(dataframe, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.20)
    for name, classifier in classifiers:
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
        
        
    
        


