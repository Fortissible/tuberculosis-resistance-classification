from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from code.DecisionTree import DecisionTree
from code.RFClassifier import RandomForest

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


def scratch_dt(x_train, x_test, y_train, y_test):
    clf = DecisionTree(max_depth=100)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    acc = accuracy(y_test, pred)
    print("from scratch dt accuracy is :", acc)

def scikit_learn_dt(x_train, x_test, y_train, y_test):
    lib_clf = DecisionTreeClassifier(max_depth=100)
    lib_clf.fit(x_train, y_train)
    lib_pred = lib_clf.predict(x_test)

    lib_acc = accuracy(y_test, lib_pred)
    print("from scikit_learn dt accuracy is :", lib_acc)

def scratch_rf(x_train, x_test, y_train, y_test):
    clf = RandomForest(max_depth=100)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    acc = accuracy(y_test, pred)
    print("from scratch dt accuracy is :", acc)

def scikit_learn_rf(x_train, x_test, y_train, y_test):
    lib_clf = RandomForestClassifier(max_depth=100)
    lib_clf.fit(x_train, y_train)
    lib_pred = lib_clf.predict(x_test)

    lib_acc = accuracy(y_test, lib_pred)
    print("from scikit_learn dt accuracy is :", lib_acc)

if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    #print(pd.DataFrame(data.data))
    #print(pd.DataFrame(data.target))

    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1234
    )

    print("------------------ DECISION TREE ------------------")
    scratch_dt(x_train, x_test, y_train, y_test)
    scikit_learn_dt(x_train, x_test, y_train, y_test)

    print("\n\n------------------ RANDOM FOREST ------------------")
    scratch_rf(x_train, x_test, y_train, y_test)
    scikit_learn_rf(x_train, x_test, y_train, y_test)
