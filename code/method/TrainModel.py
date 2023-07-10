import pandas as pd
import numpy as np
import random
import math
import time
import copy
import collections
import pickle

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn import tree as sk_tree

from code.method.MultiLabelRandomForest import RandomForestClassifier as MLRF
from code.method.ReinforcedRandomForest import RandomForestClassifier as RRF

def trainGridSO_w_KFoldCV(x_multilabel, y_multilabel,
                n_est=None,
                n_max_depth=None,
                n_min_samples_split=None,
                fold=5):

    if n_min_samples_split is None:
        n_min_samples_split = [20, 20, 15, 10]
    if n_max_depth is None:
        n_max_depth = [10, 10, 25, 50]
    if n_est is None:
        n_est = [10, 25, 50, 100]

    opt_n_estimators = n_est
    opt_max_depth = n_max_depth
    opt_min_samples_split = n_min_samples_split

    arr_acc_train_single = []
    arr_acc_test_single = []
    result_type_gsearch_kcv_test_single = []
    result_type_gsearch_kcv_train_single = []

    arr_acc_train_multi = []
    arr_acc_test_multi = []
    result_type_gsearch_kcv_test_multi = []
    result_type_gsearch_kcv_train_multi = []

    total_train_acc = []
    total_test_acc = []

    total_train_probs = []
    total_test_probs = []

    total_train_cmp = []
    total_test_cmp = []

    # loop for grid search optimization
    for p1, p2, p3 in zip(opt_n_estimators, opt_max_depth, opt_min_samples_split):

        print(f"\n-------Param Model - {p1, p2, p3}-------\n")

        fold_train_acc = []
        fold_test_acc = []

        fold_train_prob = []
        fold_test_prob = []

        fold_train_cmp = []
        fold_test_cmp = []

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
            for i in y_multilabel.columns:
                x_multi_datasets_train.append(x_train)
                x_multi_datasets_test.append(x_test)

            clf.fit(x_multi_datasets_train, y_train)

            from sklearn import metrics
            print(f"-------MODEL FOLD - {idx}-------")
            train_pred_res, train_pred_probs = clf.predict(x_train)
            test_pred_res, test_pred_probs = clf.predict(x_test)

            all_res_train_acc = []
            all_res_test_acc = []

            all_res_train_cmp = []
            all_res_test_cmp = []

            # all phenotype resistance accuracy loop
            for col_idx, column in enumerate(y_train.columns):
                print(f"-------Resistance {column}-------")

                print("train acc")
                print("actual:\t", np.array(y_train[column]))
                print("pred:\t", np.array(train_pred_res[:, col_idx]))
                n_fold_gridsearch_train_acc = metrics.accuracy_score(y_train[column], train_pred_res[:, col_idx])
                print(n_fold_gridsearch_train_acc)
                all_res_train_acc.append(n_fold_gridsearch_train_acc)
                all_res_train_cmp.append([y_train[column],
                                          train_pred_res[:, col_idx]
                                          ]
                                         )

                print("test acc")
                print("actual:\t", np.array(y_test[column]))
                print("pred:\t", np.array(test_pred_res[:, col_idx]))
                n_fold_gridsearch_test_acc = metrics.accuracy_score(y_test[column], test_pred_res[:, col_idx])
                print(n_fold_gridsearch_test_acc)
                all_res_test_acc.append(n_fold_gridsearch_test_acc)
                all_res_test_cmp.append([y_test[column],
                                         test_pred_res[:, col_idx]
                                         ]
                                        )

            fold_train_acc.append(all_res_train_acc)
            fold_test_acc.append(all_res_test_acc)

            fold_train_prob.append(train_pred_probs)
            fold_test_prob.append(test_pred_probs)

            fold_train_cmp.append(all_res_train_cmp)
            fold_test_cmp.append(all_res_test_cmp)

        # Saving the model with current parameter grid search:
        filename = f"model_GSO_w_KCV_param_{p1}_{p2}_{p3}.sav"
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(clf, f)

        total_train_acc.append(fold_train_acc)
        total_test_acc.append(fold_test_acc)

        total_train_probs.append(fold_train_prob)
        total_test_probs.append(fold_test_prob)

        total_train_cmp.append(fold_train_cmp)
        total_test_cmp.append(fold_test_cmp)

    # Saving the results:
    with open('train_GSO_w_KCV_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([total_train_acc, total_train_probs, total_train_cmp], f)

    with open('testing_GSO_w_KCV_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([total_test_acc, total_test_probs, total_test_cmp], f)

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
    # dataFrame = pd.read_csv(f'amr_datasets_r_inh.csv',sep=",")
    # dataFrame = dataFrame.drop('!accession', axis=1)
    # dataFrame = dataFrame.drop('line_age',axis=1)
    # dataFrame = dataFrame.drop('phen_r_inh',axis=1)

    dataFrame2 = pd.read_csv(f'amr_datasets_all_class_bin.csv',sep=",")
    dataFrame2 = dataFrame2.drop('!accession', axis=1)
    # dataFrame2 = dataFrame2.drop('line_age',axis=1)

    # dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)
    dataFrame2 = dataFrame2.sample(frac=1).reset_index(drop=True)

    # dataFrame = pd.DataFrame({
    #     "Outlook":
    #     ["S","S","O","R","R","R","O","S","S","R","S","O","O","R"],
    #     "Temp":
    #     ["H","H","H","M","C","C","C","M","C","M","M","M","H","M"],
    #     "Humidity":
    #     ["H","H","H","H","N","N","N","H","N","N","N","H","N","H"],
    #     "Windy":
    #     ["F","T","F","F","F","T","T","F","F","F","T","T","F","T"],
    #     "Play":
    #     ["F","F","T","T","T","F","T","F","T","T","T","T","T","F"]
    # })

    # create LabelEncoder object
    # label_encoder = LabelEncoder()

    # fit and transform the categorical variable

    # for header in column_headers:
    # for header in dataFrame.columns:
    #   dataFrame[header] = label_encoder.fit_transform(dataFrame[header])

    x_multilabel, y_multilabel = dataFrame2.iloc[:, :-4], dataFrame2.iloc[:, -4:] #multilabel
    # x_singlelabel, y_singlelabel = dataFrame.iloc[:, :-1], dataFrame.iloc[:, -1]  #singlelabel

    print(x_multilabel,y_multilabel,"\n-------------------------------\n")
    # print(x_singlelabel,y_singlelabel)

    trainGridSO_w_KFoldCV(x_multilabel,y_multilabel)