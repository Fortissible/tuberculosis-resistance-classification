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
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import tree as sk_tree
from sklearn.tree import export_graphviz
from subprocess import call

from code.method.MultiLabelRandomForest import RandomForestClassifier as MLRF
from code.method.ReinforcedRandomForest import RandomForestClassifier as RRF
from code.metric.Metric import *

def train_sklearnRF_GridSO_w_KFoldCV(x_singlelabel, y_singlelabel,
                n_est=None,
                n_max_depth=None,
                n_min_samples_split=None,
                fold=5):

    if n_min_samples_split is None:
        n_min_samples_split = [20, 20, 15]
    if n_max_depth is None:
        n_max_depth = [10, 10, 25]
    if n_est is None:
        n_est = [10, 25, 50]

    opt_n_estimators = n_est
    opt_max_depth = n_max_depth
    opt_min_samples_split = n_min_samples_split

    # loop for grid search optimization
    for p1, p2, p3 in zip(opt_n_estimators, opt_max_depth, opt_min_samples_split):

        print(f"\n-------Param Model - {p1, p2, p3}-------\n")

        clf = RandomForestClassifier(
                   n_estimators=p1,
                   max_depth=p2,
                   min_samples_split=p3,
                   min_samples_leaf=2,
                   random_state=77
                   )

        """---------K-Fold_Crossvalidation w Multilabel---------"""

        kfold_cv = KFold(n_splits=fold, random_state=100, shuffle=True)

        # n-Fold Loop
        for idx, (train_index, test_index) in enumerate(kfold_cv.split(x_singlelabel)):
            x_train, x_test = x_singlelabel.iloc[train_index, :], x_singlelabel.iloc[test_index, :]
            y_train, y_test = y_singlelabel.iloc[train_index], y_singlelabel.iloc[test_index]

            clf.fit(x_train, y_train)

            from sklearn import metrics
            print(f"-------MODEL {y_singlelabel.name} FOLD - {idx}-------")
            train_pred = clf.predict(x_train)
            test_pred = clf.predict(x_test)

            n_fold_gridsearch_train_acc = metrics.accuracy_score(y_train, train_pred)
            print("train acc", n_fold_gridsearch_train_acc)

            n_fold_gridsearch_test_acc = metrics.accuracy_score(y_test, test_pred)
            print("test acc", n_fold_gridsearch_test_acc)

        # Saving the model with current parameter grid search:
        filename = f"model_{y_singlelabel.name}_sklearn_RF_GSO_w_KCV_param_{p1}_{p2}_{p3}.sav"
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(clf, f)

    # # Saving the results:
    # with open('train_sklearn_RF_GSO_w_KCV_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([total_train_acc, total_train_probs, total_train_cmp], f)
    #
    # with open('testing_sklearn_RF_GSO_w_KCV_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([total_test_acc, total_test_probs, total_test_cmp], f)

def trainGridSO_w_KFoldCV(x_multilabel, y_multilabel,
                n_est=None,
                n_max_depth=None,
                n_min_samples_split=None,
                fold=5):

    if n_min_samples_split is None:
        n_min_samples_split = [20, 20, 15]
    if n_max_depth is None:
        n_max_depth = [10, 10, 25]
    if n_est is None:
        n_est = [10, 25, 50]

    opt_n_estimators = n_est
    opt_max_depth = n_max_depth
    opt_min_samples_split = n_min_samples_split

    total_train_acc = []
    total_test_acc = []

    total_train_probs = []
    total_test_probs = []

    total_train_cmp = []
    total_test_cmp = []

    # loop for grid search optimization
    for p1, p2, p3 in zip(opt_n_estimators, opt_max_depth, opt_min_samples_split):

        print(f"\n-------Param Model - {p1, p2, p3}-------\n")

        clf = MLRF(n_estimators=p1,
                   max_depth=p2,
                   min_samples_split=p3,
                   min_samples_leaf=2,
                   min_split_gain=0.0005,
                   colsample_bytree="sqrt",
                   subsample=0.8,
                   random_state=77,
                   classifier_type="gini",
                   multilabel=True
                   )

        """---------K-Fold_Crossvalidation w Multilabel---------"""

        kfold_cv = KFold(n_splits=fold, random_state=100, shuffle=True)

        # n-Fold Loop
        for idx, (train_index, test_index) in enumerate(kfold_cv.split(x_multilabel)):
            x_train, x_test = x_multilabel.iloc[train_index, :], x_multilabel.iloc[test_index, :]
            y_train, y_test = y_multilabel.iloc[train_index], y_multilabel.iloc[test_index]

            x_multi_datasets_train = []
            x_multi_datasets_test = []

            # copy dataset into n-label similar dataset
            for _ in y_multilabel.columns:
                x_multi_datasets_train.append(x_train)
                x_multi_datasets_test.append(x_test)

            clf.fit(x_multi_datasets_train, y_train)

            from sklearn import metrics
            print(f"-------MODEL FOLD - {idx}-------")
            train_pred_res, train_pred_rmg_list, train_pred_leaf_loc_list, train_tree_feature_path = clf.predict(x_train)
            test_pred_res, test_pred_rmg_list, test_pred_leaf_loc_list, test_tree_feature_path = clf.predict(x_test)


            # all phenotype resistance accuracy loop
            for col_idx, column in enumerate(y_train.columns):
                print(f"-------Resistance {column}-------")

                n_fold_gridsearch_train_acc = metrics.accuracy_score(y_train[column], train_pred_res[:, col_idx])
                print("train acc", n_fold_gridsearch_train_acc)

                n_fold_gridsearch_test_acc = metrics.accuracy_score(y_test[column], test_pred_res[:, col_idx])
                print("test acc", n_fold_gridsearch_test_acc)

        # Saving the model with current parameter grid search:
        filename = f"model_GSO_w_KCV_param_{p1}_{p2}_{p3}.sav"
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(clf, f)

    # # Saving the results:
    # with open('train_GSO_w_KCV_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([total_train_acc, total_train_probs, total_train_cmp], f)
    #
    # with open('testing_GSO_w_KCV_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([total_test_acc, total_test_probs, total_test_cmp], f)

def trainGridSO(x_multilabel, y_multilabel,
                n_est=None,
                n_max_depth=None,
                n_random_state=None,
                n_min_samples_split=None):

    if n_min_samples_split is None:
        n_min_samples_split = [10, 20, 30]
    if n_max_depth is None:
        n_max_depth = [10, 25, 50]
    if n_random_state is None:
        n_random_state = [2, 8, 16]
    if n_est is None:
        n_est = [5, 10, 25]

    opt_n_estimators = n_est
    opt_max_depth = n_max_depth
    opt_random_state = n_random_state
    opt_min_samples_split = n_min_samples_split

    arr_acc_train = []
    arr_acc_test = []

    for p1, p2, p3, p4 in zip(
            opt_n_estimators,
            opt_max_depth,
            opt_random_state,
            opt_min_samples_split
    ):

        # clf_singlelabel = RandomForestClassifier(n_estimators=p1,
        #                             max_depth=p2,
        #                             min_samples_split=p4,
        #                             min_samples_leaf=2,
        #                             min_split_gain=0.0,
        #                             colsample_bytree="sqrt",
        #                             subsample=0.8,
        #                             random_state=p3,
        #                             classifier_type="information_gain",
        #                             multilabel = False
        #                             )

        # x_single_train, x_single_test, y_single_train, y_single_test = train_test_split(
        #     x_singlelabel, y_singlelabel, test_size=0.25, random_state=random.randint(1, 25)
        # )

        # label_name = y_singlelabel.to_frame().columns

        # clf_singlelabel.fit(x_single_train, y_single_train)

        # from sklearn import metrics
        # #SINGLE LABEL
        # print(f"------ SINGLELABEL {label_name} w param {p1},{p2},{p3},{p4} ------\n")
        # for idx,label_name in enumerate(y_singlelabel.to_frame().columns):
        #   print(f"RESISTANCE FOR LABEL {label_name} with parameter {p1},{p2},{p3},{p4}")
        #   print(metrics.accuracy_score(y_single_train.to_frame()[label_name],
        #                                clf_singlelabel.predict(x_single_train,label_num=idx)
        #                                )
        #   )
        #   print(metrics.accuracy_score(y_single_test.to_frame()[label_name],
        #                                clf_singlelabel.predict(x_single_test,label_num=idx)
        #                                )
        #   )
        # print(clf_singlelabel.feature_importances_,"\n\n")

        # ----------------- MULTILABEL -----------------

        clf_multilabel = RandomForestClassifier(n_estimators=p1,
                                                max_depth=p2,
                                                min_samples_split=p4,
                                                min_samples_leaf=2,
                                                min_split_gain=0.0,
                                                colsample_bytree="sqrt",
                                                subsample=0.8,
                                                random_state=p3,
                                                classifier_type="gini",
                                                multilabel=True
                                                )

        x_multi_train, x_multi_test, y_multi_train, y_multi_test = train_test_split(
            x_multilabel, y_multilabel, test_size=0.25, random_state=random.randint(1, 25)
        )

        x_multi_datasets_train = []

        for i in y_multi_train.columns:
            x_multi_datasets_train.append(x_multi_train)

        clf_multilabel.fit(x_multi_datasets_train, y_multi_train)

        # MULTI LABEL
        print(f"-------- MULTILABEL w param {p1},{p2},{p3},{p4} --------\n")
        for idx, label_name in enumerate(y_multilabel.columns):
            print(f"RESISTANCE FOR LABEL {label_name} with parameter {p1},{p2},{p3},{p4}")
            print(metrics.accuracy_score(y_multi_train[label_name],
                                         clf_multilabel.predict(x_multi_train, label_num=idx)
                                         )
                  )
            print(metrics.accuracy_score(y_multi_test[label_name],
                                         clf_multilabel.predict(x_multi_test, label_num=idx)
                                         )
                  )

        print(clf_multilabel.feature_importances_, "\n\n")

        filename = f"model_GSO_param_{p1}_{p2}_{p3}.sav"
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(clf, f)



if __name__ == '__main__':
    dataFrame = pd.read_csv(f'dataset_all_nolineage_nospecies.csv',sep=",")
    dataFrame = dataFrame.drop('!accession', axis=1)
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

    #fit and transform the categorical variable
    # encoded = pd.get_dummies(dataFrame["lineage"], prefix="lineage")
    # print(encoded)

    #Drop lineage feature if exist (optional to drop)
    # classes = dataFrame.iloc[:, -4:]
    # print(classes)
    #
    # dataFrame = dataFrame.drop('lineage', axis=1)
    # dataFrame = dataFrame.drop('phen_inh', axis=1)
    # dataFrame = dataFrame.drop('phen_rif', axis=1)
    # dataFrame = dataFrame.drop('phen_emb', axis=1)
    # dataFrame = dataFrame.drop('phen_pza', axis=1)
    # dataFrame_new = pd.concat([dataFrame, encoded, classes], axis=1, join='inner', copy=True)
    # print(dataFrame_new)

    #MULTILABEL
    x_multilabel, y_multilabel = dataFrame.iloc[:, :-4], dataFrame.iloc[:, -4:] #multilabel
    trainGridSO_w_KFoldCV(x_multilabel, y_multilabel)

    #SINGLELABEL
    dataFrameSingleLabel = dataFrame.copy()
    dataFrameSingleLabel = dataFrameSingleLabel.drop('phen_inh', axis=1)
    dataFrameSingleLabel = dataFrameSingleLabel.drop('phen_rif', axis=1)
    dataFrameSingleLabel = dataFrameSingleLabel.drop('phen_emb', axis=1)
    dataFrameSingleLabel = dataFrameSingleLabel.drop('phen_pza', axis=1)
    for i in range(-4,0):
        y_singlelabel = dataFrame.iloc[:, i]  #singlelabel
        train_sklearnRF_GridSO_w_KFoldCV(dataFrameSingleLabel, y_singlelabel)

