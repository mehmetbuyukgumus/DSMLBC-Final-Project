from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

classifiers = [("KNN", KNeighborsClassifier()),
               ("Decision Tree", DecisionTreeClassifier()),
               ("Rest C", OneVsOneClassifier(RandomForestClassifier())),
               ("SVM", SVC())
               ]