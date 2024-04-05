import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from warnings import filterwarnings
filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Data Prep
data = pd.read_csv("datasets/winequalityN.csv")
data["target"] = data["quality"].apply(lambda x: 1 if x >= 6 else 0)
df = data.drop("quality", axis=1)

# Mapping
mapping = {"red" : 0,"white": 1}
df["type"] = df["type"].map(mapping)

# N/A Values
df.dropna(inplace=True)

# Standardization
none_target_cols = [col for col in df.columns if col not in ["target"]]
scaler = StandardScaler()
df[none_target_cols] = scaler.fit_transform(df[none_target_cols])

# Modeling
X = df.drop("target", axis=1)
y = df["target"]
metrics = ["f1", "accuracy", "roc_auc", "precision"]

def base_models(X, y, scoring="roc_auc", cv=3):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('RF', RandomForestClassifier()),
                   ("GB", GradientBoostingClassifier()),
                   ("ADA", AdaBoostClassifier()),
                   ("XGBM", XGBClassifier()),
                   ("LGBM", LGBMClassifier(verbose=-1)),
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")

for metric in metrics:        
    base_models(X,y, metric, cv=3)
    
    
# İlk Sonuçlar

# Base Models....
# f1: 0.7449 (LR) 
# f1: 0.7342 (RF) 
# f1: 0.7441 (GB) 
# f1: 0.7244 (ADA) 
# f1: 0.7254 (XGBM) 
# f1: 0.7225 (LGBM) 
# Base Models....
# accuracy: 0.6947 (LR) 
# accuracy: 0.6995 (RF) 
# accuracy: 0.7022 (GB) 
# accuracy: 0.681 (ADA) 
# accuracy: 0.6873 (XGBM) 
# accuracy: 0.6876 (LGBM) 
# Base Models....
# roc_auc: 0.7981 (LR) 
# roc_auc: 0.7914 (RF) 
# roc_auc: 0.7994 (GB) 
# roc_auc: 0.7656 (ADA) 
# roc_auc: 0.7807 (XGBM) 
# roc_auc: 0.7871 (LGBM) 
# Base Models....
# precision: 0.8007 (LR) 
# precision: 0.821 (RF) 
# precision: 0.8128 (GB) 
# precision: 0.7939 (ADA) 
# precision: 0.8123 (XGBM) 
# precision: 0.82 (LGBM)
