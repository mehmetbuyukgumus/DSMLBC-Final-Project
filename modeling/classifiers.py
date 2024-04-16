from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from modeling.hyper_params import *

classifiers = [
            ("KNN", KNeighborsClassifier(), knn_params),
            ("Decision Tree", DecisionTreeClassifier(), decision_tree_params),
            ("OneVs", OneVsOneClassifier(RandomForestClassifier()), one_vs_params),
            ("SVM", SVC(), SVC_params)
            ]