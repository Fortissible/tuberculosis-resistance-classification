import pandas as pd
import numpy as np
import random
import math
import time
import copy
import collections
import pickle

from joblib import Parallel, delayed

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn import tree as sk_tree
from sklearn.tree import export_graphviz
from subprocess import call

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import roc_curve, roc_auc_score

from code.method.MultiLabelRandomForest import RandomForestClassifier as MLRF
from code.method.ReinforcedRandomForest import RandomForestClassifier as RRF
from code.metric.Metric import *

if __name__ == "__main__":
    dataFrame2 = pd.read_csv(f'amr_datasets_all_class_bin.csv', sep=",")
    dataFrame2 = dataFrame2.drop('!accession', axis=1)
    dataFrame2 = dataFrame2.sample(frac=1).reset_index(drop=True)
    x_multilabel, y_multilabel = dataFrame2.iloc[:, :-4], dataFrame2.iloc[:, -4:] #multilabel
    X_train, X_test, y_train, y_test = train_test_split(x_multilabel, y_multilabel, test_size=0.3)
    model_10_10_20 = None
    model_25_20_15 = None
    model_50_30_10 = None

    with open('../saved_model/model_GSO_w_KCV_param_10_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_10_10_20 = pickle.load(f)

    with open('../saved_model/model_GSO_w_KCV_param_25_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_25_20_15 = pickle.load(f)

    with open('../saved_model/model_GSO_w_KCV_param_50_25_15.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_50_30_10 = pickle.load(f)

    model_list = [model_10_10_20, model_25_20_15, model_50_30_10]
    model_list_name = ["model_10_10_20", "model_25_20_15", "model_50_30_10"]

    for model_idx, model in enumerate(model_list) :
        train_pred_res, train_pred_rmg_list, train_pred_leaf_loc_list, train_tree_feature_path = model.predict(X_train)
        test_pred_res, test_pred_rmg_list, test_pred_leaf_loc_list, test_tree_feature_path = model.predict(X_test)

        all_res_train_acc = []
        all_res_test_acc = []

        all_res_train_cmp = []
        all_res_test_cmp = []

        for col_idx, column in enumerate(y_train.columns):
            print(f"\n\n-------Resistance {column} Model {model_list_name[model_idx]}-------")

            # TRAIN
            n_fold_gridsearch_train_acc = accuracy_score(y_train[column],train_pred_res[:, col_idx])
            print(f"Train acc : {n_fold_gridsearch_train_acc}")
            print("actual:\t", np.array(y_train[column]))
            print("pred:\t", np.array(train_pred_res[:, col_idx]))
            print(n_fold_gridsearch_train_acc)
            all_res_train_acc.append(n_fold_gridsearch_train_acc)
            all_res_train_cmp.append([y_train[column],
                                      train_pred_res[:, col_idx]
                                      ]
                                     )

            metric_confusion_matrix(
                y_train[column],
                train_pred_res[:, col_idx],
                f"Train {column} Model {model_list_name[model_idx]}"
            )
            metric_all_score(
                y_train[column],
                train_pred_res[:, col_idx]
            )

            # TEST
            n_fold_gridsearch_test_acc = accuracy_score(y_test[column], test_pred_res[:, col_idx])
            print(f"\nTest acc {n_fold_gridsearch_test_acc}")
            # print("actual:\t", np.array(y_test[column]))
            # print("pred:\t", np.array(test_pred_res[:, col_idx]))
            all_res_test_acc.append(n_fold_gridsearch_test_acc)
            all_res_test_cmp.append([y_test[column],
                                     test_pred_res[:, col_idx]
                                     ]
                                    )

            metric_confusion_matrix(
                y_test[column],
                test_pred_res[:, col_idx],
                f"Test-{column}-Model-{model_list_name[model_idx]}",
                visualize=True
            )

            metric_all_score(
                y_train[column],
                train_pred_res[:, col_idx]
            )

        mean_test_pred_proba = metric_get_predict_proba(test_pred_leaf_loc_list, test_pred_rmg_list)

        metric_roc_auc(y_test,mean_test_pred_proba, model_list_name[model_idx])

        if model_idx+1 != len(model_list):
            print(f"\n\n--------- Next Model {model_list_name[model_idx+1]} ---------")
