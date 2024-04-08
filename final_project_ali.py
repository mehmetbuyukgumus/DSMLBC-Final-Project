import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay,recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
import warnings
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("datasets/winequalityN.csv")

def check_df(dataframe, head=5):
    print(10*"#" + " Shape ".center(9) + 10*"#")
    print(dataframe.shape)
    print(10*"#" + " Types ".center(9) + 10*"#")
    print(dataframe.dtypes)
    print(10*"#" + " Head ".center(9) + 10*"#")
    print(dataframe.head(head))
    print(10*"#" + " Tail ".center(9) + 10*"#")
    print(dataframe.tail(head))
    print(10*"#" + " NA ".center(9) + 10*"#")
    print(dataframe.isnull().sum())
    print(10*"#" + " Quantiles ".center(9) + 10*"#")
    print(dataframe.describe([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]).T)
    print(10*"#" + " Unique Values ".center(9) + 10*"#")
    print(dataframe.nunique())
check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal degişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe:dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th:int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal degişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişen listesi
    num_cols: list
        Numerik degisken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un icerisinde.


    """


    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["object", "category"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]]
    num_cols = [col for col in dataframe.columns if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_car: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["type"] = LabelEncoder().fit_transform(df["type"])

# ----------------------------------------------------------------------------------------------------------------------
df.dropna(inplace=True)

y = df["quality"]
X = df.drop(["quality"], axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

RF=OneVsRestClassifier(RandomForestClassifier(max_features=0.2))
RF.fit(X_train,y_train)
y_pred = RF.predict(X_test)
pred_prob=RF.predict_proba(X_test)
pred_prob.shape

RF.score(X_test,y_test)

# ----------------------------------------------------------------------------------------------------------------------


def outlier_thresholds(dataframe, col_name, q1, q3):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, q1, q3):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col, 0.05, 0.95)

df.isnull().sum()

for col in df.columns:
    df[col].fillna(df[col].median(), inplace=True)

df["total acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]

df["residul sugar levels"] = pd.cut(df["residual sugar"],
                                    bins=[0, 1, 17, 35, 120, 1000],
                                    labels=["bone_dry", "dry", "off_dry", "medium_dry", "sweet"])

df["alcohol levels"] = pd.cut(df["alcohol"], bins=[0, 12.5, 13.5, 14.5, 20],
                              labels=["low", "moderately_low", "high", "very_high"])

df["white perfect pH"] = np.where((df["type"] == 1) & (df["pH"] >= 3) & (df["pH"] <= 3.3), 1, 0)

df["red perfect pH"] = np.where((df["type"] == 0) & (df["pH"] >= 3.3) & (df["pH"] <= 3.6), 1, 0)

df["perfect ph"] = df["white perfect pH"] + df["red perfect pH"]

df["Cquality"] = pd.cut(df["quality"], bins=[0, 5, 6, 10], labels=["bad", "good", "perfect"])

df["residul sugar levels"] = LabelEncoder().fit_transform(df["residul sugar levels"])
df["alcohol levels"] = LabelEncoder().fit_transform(df["alcohol levels"])
df["Cquality"] = LabelEncoder().fit_transform(df["Cquality"])

df.drop(columns=["quality", "red perfect pH", "white perfect pH"], inplace=True, axis=1)

y = df["Cquality"]
X = df.drop(["Cquality"], axis=1)

# ----------------------------------------------------------------------------------------------------------------------

MinMaxScaler = preprocessing.MinMaxScaler()
X = MinMaxScaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
ypred = knn_clf.predict(X_test)

result = confusion_matrix(y_test, ypred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, ypred)
print("Classification Report:",)
print(result1)
result2 = accuracy_score(y_test,ypred)
print("Accuracy:",result2)

# ----------------------------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=classifier.classes_)
disp.plot()
plt.show()
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('F1-Score : ', f1_score(y_test, y_pred, average = 'weighted'))
print('Recall: ', recall_score(y_test, y_pred ,average = 'micro'))

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print(result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

# ----------------------------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize SVM classifier
svm = SVC(kernel='linear', C=1.0)

# Fit the classifier to the training data
svm.fit(X_train, y_train)

# Predict on the testing data
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# Initialize Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the classifier to the training data
gb_classifier.fit(X_train, y_train)

# Predict on the testing data
y_pred = gb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

