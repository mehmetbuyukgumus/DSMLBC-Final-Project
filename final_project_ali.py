import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv('Final Project/winequalityN.csv')

df.head()
df.info()

df['quality'].nunique()
df['fixed acidity'].nunique()


df.groupby('type')["quality"].mean()
df.groupby('quality')["fixed acidity"].mean()
# 7 unique quality numbers but actually 10 in total
df["quality"].describe().T

def get_col_names(dataframe, cat_th=10, car_th = 20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == 'O']

    # Categorical Columns

    categorical_columns = cat_cols + num_but_cat
    categorical_columns = [col for col in categorical_columns if col not in categorical_columns]

    # Numerical Columns
    numerical_columns = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    numerical_columns = [col for col in numerical_columns if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'Number of Categorical Columns: {len(cat_cols)}')
    print(f'Number of Numerical Columns: {len(numerical_columns)}')
    print(f'Number of Hidden Cardinal Columns: {len(cat_but_car)}')
    print(f'Number of Categorical Columns But Stored As Numerical in Dataframe: {len(num_but_cat)}')
    print(f'Categorical Columns But Stored as Numerical in Dataframe: {num_but_cat}')
    return cat_cols, numerical_columns, cat_but_car


get_col_names(df)
# Check for missing values
df.isnull().sum()

# Filling missing values with their mean values
for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())

# Check for outliers

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

num_cols = df.drop(["type", "quality"], axis = 1)
for col in num_cols:
    if check_outlier(df, col) == True:
        grab_outliers(df, col)
        print(col, outlier_thresholds(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if check_outlier(df, col) == True:
        replace_with_thresholds(df, col)
        print(col, outlier_thresholds(df, col))

for col in num_cols:
     print(col, check_outlier(df, col))

#df.describe()

"""""
sns.set()
plt.figure(figsize=(20,10))
sns.boxplot(data=df,palette="Set3")
plt.show()
"""""

# len(df[df['free sulfur dioxide'] > 83])
# len(df[df['total sulfur dioxide'] > 283])
# len(df[df['residual sugar'] > 20])

df["type"].value_counts()
df.groupby(['type', 'quality']).agg({"pH": "mean"})

le = LabelEncoder()
df['type'] = le.fit_transform(df["type"])
le.inverse_transform([0, 1])

df.head()
df['quality'].value_counts()

y = df["quality"]
X = df.drop("quality", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
y_pred = dtree_model.predict(X_test)
cm = confusion_matrix(y_test, dtree_predictions)
print(cm)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test))}")
# ali