import pandas as pd
import numpy as np
import random
import math
import time
import copy
import collections
import pickle
import ast

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
import graphviz

import pydot
from IPython.display import Image

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import roc_curve, roc_auc_score

from code.method.MultiLabelRandomForest import RandomForestClassifier as MLRF
from code.method.ReinforcedRandomForest import RandomForestClassifier as RRF
from code.metric.Metric import *

def create_MLRFtree_visualization(data, graph, parent_node=None, left_id=0, right_id=0):
    if 'leaf_value' in data:
        leaf_node = pydot.Node(
            str(left_id)+str(right_id)+"\n"+
            str(data['leaf_value'][0])+
            str(data['leaf_value'][1])+"\n"+
            str(data['leaf_value'][2])+
            str(data['leaf_value'][3]))
        graph.add_node(leaf_node)
        if parent_node:
            graph.add_edge(pydot.Edge(parent_node, leaf_node))
    else:
        split_feature = data['split_feature']
        split_value = data['split_value']
        node_label = f"{left_id}{right_id}\n{split_feature}\n<= {split_value}"

        node = pydot.Node(node_label, shape='box')
        graph.add_node(node)
        if parent_node:
            graph.add_edge(pydot.Edge(parent_node, node))

        left_tree = data['left_tree']
        right_tree = data['right_tree']
        create_MLRFtree_visualization(left_tree, graph, node, left_id=left_id+1, right_id=right_id)
        create_MLRFtree_visualization(right_tree, graph, node, left_id=left_id, right_id=right_id+1)

def dataSplitting(dataFrame):
    x_multilabel, y_multilabel = dataFrame.iloc[:, :-4], dataFrame.iloc[:, -4:]  # multilabel
    X_train, X_test, y_train, y_test = train_test_split(x_multilabel, y_multilabel, test_size=0.3)
    return X_train, X_test, y_train, y_test

def useSLRF(X_train, X_test, y_train, y_test):
    model_10_10_20 = []
    model_25_10_20 = []
    model_50_25_15 = []
    for i in range(0,4):
        print(y_train.columns[i])
        with open(f'../saved_model/model_{y_train.columns[i]}_sklearn_RF_GSO_w_KCV_param_10_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
            model_10_10_20.append(pickle.load(f))
        with open(f'../saved_model/model_{y_train.columns[i]}_sklearn_RF_GSO_w_KCV_param_25_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
            model_25_10_20.append(pickle.load(f))
        with open(f'../saved_model/model_{y_train.columns[i]}_sklearn_RF_GSO_w_KCV_param_50_25_15.sav', 'rb') as f:  # Python 3: open(..., 'rb')
            model_50_25_15.append(pickle.load(f))

    model_list = [model_10_10_20, model_25_10_20, model_50_25_15]
    model_list_name = ["model_10_10_20", "model_25_10_20", "model_50_25_15"]

    for model_idx, model in enumerate(model_list):
        model_pred_proba = []
        model_y_test = []
        for i in range(0, 4):

            train_pred = model[i].predict(X_train)
            test_pred = model[i].predict(X_test)

            print(f"\n\n-------Resistance {y_train.columns[i]} Model {model_list_name[model_idx]}-------")

            # ---TRAIN---

            # print("actual:\t", np.array(y_train))
            # print("pred:\t", np.array(train_pred))

            print("\nTRAINING\n")
            metric_confusion_matrix(
                y_train[y_train.columns[i]],
                train_pred,
                f"Train-{y_train.columns[i]}-Model-SLRF-{model_list_name[model_idx]}"
            )
            metric_all_score(
                y_train[y_train.columns[i]],
                train_pred
            )

            # ---TEST---

            # print("actual:\t", np.array(y_test))
            # print("pred:\t", np.array(test_pred))

            print("\nTESTING\n")
            metric_confusion_matrix(
                y_test[y_test.columns[i]],
                test_pred,
                f"Test-{y_test.columns[i]}-Model-SLRF-{model_list_name[model_idx]}",
                visualize=True
            )
            metric_all_score(
                y_test[y_test.columns[i]],
                test_pred
            )

            pred_proba = model[i].predict_proba(X_test)[:, 1]
            model_pred_proba.append(pred_proba)
            model_y_test.append(y_test[y_test.columns[i]].to_numpy())

        mean_test_pred_proba = np.vstack((model_pred_proba[0],
                                          model_pred_proba[1],
                                          model_pred_proba[2],
                                          model_pred_proba[3])
                                         ).T

        vstack_model_y_test = np.vstack((model_y_test[0],
                                         model_y_test[1],
                                         model_y_test[2],
                                         model_y_test[3])
                                        ).T
        y_test_new = pd.DataFrame(data=vstack_model_y_test,
                              columns=["phen_inh", "phen_rif", "phen_emb", "phen_pza"])

        metric_roc_auc(y_test_new,mean_test_pred_proba, model_list_name[model_idx], model_type="SLRF")

def printSLRFDecisionTree(X_train, X_test, y_train, y_test):
    model_10_10_20 = []
    model_25_10_20 = []
    model_50_25_15 = []

    for i in range(0,4):
        with open(f'../saved_model/model_{y_train.columns[i]}_sklearn_RF_GSO_w_KCV_param_10_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
            model_10_10_20.append(pickle.load(f))
        with open(f'../saved_model/model_{y_train.columns[i]}_sklearn_RF_GSO_w_KCV_param_25_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
            model_25_10_20.append(pickle.load(f))
        with open(f'../saved_model/model_{y_train.columns[i]}_sklearn_RF_GSO_w_KCV_param_50_25_15.sav', 'rb') as f:  # Python 3: open(..., 'rb')
            model_50_25_15.append(pickle.load(f))

    model_list = [model_10_10_20, model_25_10_20, model_50_25_15]
    model_list_name = ["model_10_10_20", "model_25_10_20", "model_50_25_15"]

    for model_idx, model in enumerate(model_list):
        for i in range(0,4):
            export_graphviz(model_list[model_idx][i].estimators_[0], out_file='tree.dot',
                               feature_names=X_test.columns,
                               max_depth=10,
                               class_names=["0","1"],
                               rounded=True, proportion=False,
                               filled=True)
            call(['dot', '-Tpng', 'tree.dot', '-o', f'DT0 SLRF {y_train.columns[i]} Model {model_list_name[model_idx]}.png', '-Gdpi=600'])

def printSLRFFeatureImportances(X_train, X_test, y_train, y_test):
    model_10_10_20 = []
    model_25_10_20 = []
    model_50_25_15 = []
    for i in range(0, 4):
        print(y_train.columns[i])
        with open(f'../saved_model/model_{y_train.columns[i]}_sklearn_RF_GSO_w_KCV_param_10_10_20.sav',
                  'rb') as f:  # Python 3: open(..., 'rb')
            model_10_10_20.append(pickle.load(f))
        with open(f'../saved_model/model_{y_train.columns[i]}_sklearn_RF_GSO_w_KCV_param_25_10_20.sav',
                  'rb') as f:  # Python 3: open(..., 'rb')
            model_25_10_20.append(pickle.load(f))
        with open(f'../saved_model/model_{y_train.columns[i]}_sklearn_RF_GSO_w_KCV_param_50_25_15.sav',
                  'rb') as f:  # Python 3: open(..., 'rb')
            model_50_25_15.append(pickle.load(f))

    model_list = [model_10_10_20, model_25_10_20, model_50_25_15]
    model_list_name = ["model_10_10_20", "model_25_10_20", "model_50_25_15"]

    for model_idx, model in enumerate(model_list):
        for i in range(0, 4):
            print(model[i].feature_importances_)
            print("\n\n")

def useMLRF(X_train, X_test, y_train, y_test):

    model_10_10_20 = None
    model_25_10_20 = None
    model_50_25_15 = None

    with open('../saved_model/model_GSO_w_KCV_param_10_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_10_10_20 = pickle.load(f)

    with open('../saved_model/model_GSO_w_KCV_param_25_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_25_10_20 = pickle.load(f)

    with open('../saved_model/model_GSO_w_KCV_param_50_25_15.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_50_25_15 = pickle.load(f)

    model_list = [model_10_10_20, model_25_10_20, model_50_25_15]
    model_list_name = ["model_10_10_20", "model_25_10_20", "model_50_25_15"]

    for model_idx, model in enumerate(model_list) :
        train_pred_res, train_pred_rmg_list, train_pred_leaf_loc_list, train_tree_feature_path = model.predict(X_train)
        test_pred_res, test_pred_rmg_list, test_pred_leaf_loc_list, test_tree_feature_path = model.predict(X_test)

        for col_idx, column in enumerate(y_train.columns):
            print(f"\n\n-------Resistance {column} Model {model_list_name[model_idx]}-------")

            # ---TRAIN---

            # print("actual:\t", np.array(y_train[column]))
            # print("pred:\t", np.array(train_pred_res[:, col_idx]))

            print("\nTRAINING\n")
            metric_confusion_matrix(
                y_train[column],
                train_pred_res[:, col_idx],
                f"Train-{column}-Model-MLRF-{model_list_name[model_idx]}"
            )
            metric_all_score(
                y_train[column],
                train_pred_res[:, col_idx]
            )

            # ---TEST---

            # print("actual:\t", np.array(y_test[column]))
            # print("pred:\t", np.array(test_pred_res[:, col_idx]))

            print("\nTESTING\n")
            metric_confusion_matrix(
                y_test[column],
                test_pred_res[:, col_idx],
                f"Test-{column}-Model-MLRF-{model_list_name[model_idx]}",
                visualize=True
            )

            metric_all_score(
                y_test[column],
                test_pred_res[:, col_idx]
            )

        mean_test_pred_proba = metric_get_predict_proba(test_pred_leaf_loc_list, test_pred_rmg_list)

        metric_roc_auc(y_test, mean_test_pred_proba, model_list_name[model_idx], model_type="MLRF")

        if model_idx+1 != len(model_list):
            print(f"\n\n--------- Next Model {model_list_name[model_idx+1]} ---------")

def printMLRFDecisionTree(X_train, X_test, y_train, y_test):
    model_10_10_20 = None
    model_25_10_20 = None
    model_50_25_15 = None

    with open('../saved_model/model_GSO_w_KCV_param_10_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_10_10_20 = pickle.load(f)

    with open('../saved_model/model_GSO_w_KCV_param_25_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_25_10_20 = pickle.load(f)

    with open('../saved_model/model_GSO_w_KCV_param_50_25_15.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_50_25_15 = pickle.load(f)

    model_list = [model_10_10_20, model_25_10_20, model_50_25_15]
    model_list_name = ["model_10_10_20", "model_25_10_20", "model_50_25_15"]

    for model_idx, model in enumerate(model_list):
        template_model = MLRF(n_estimators=0,
                   max_depth=0,
                   min_samples_split=0,
                   min_samples_leaf=2,
                   min_split_gain=0.0005,
                   colsample_bytree="sqrt",
                   subsample=0.8,
                   random_state=77,
                   classifier_type="gini",
                   multilabel=True
                   )
        template_model.n_estimators = model.n_estimators
        template_model.max_depth = model.max_depth
        template_model.min_samples_split = model.min_samples_split
        template_model.min_samples_leaf = model.min_samples_leaf
        template_model.min_split_gain = model.min_split_gain
        template_model.colsample_bytree = model.colsample_bytree
        template_model.subsample = model.subsample
        template_model.random_state = model.random_state
        template_model.trees = model.trees
        template_model.feature_importances_ = model.feature_importances_
        template_model.classifier_type = model.classifier_type
        template_model.multilabel = model.multilabel

        graph = pydot.Dot(graph_type="digraph")

        data = template_model.trees[0].describe_tree()
        create_MLRFtree_visualization(data, graph)

        # Save the graph as a PNG file
        print(graph,"\n\n")
        graph.write_png(f'MLRF_Model_{model_list_name[model_idx]}_tree_example.png')

def printMLRFFeatureImportances(X_train, X_test, y_train, y_test):
    model_10_10_20 = None
    model_25_10_20 = None
    model_50_25_15 = None

    with open('../saved_model/model_GSO_w_KCV_param_10_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_10_10_20 = pickle.load(f)

    with open('../saved_model/model_GSO_w_KCV_param_25_10_20.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_25_10_20 = pickle.load(f)

    with open('../saved_model/model_GSO_w_KCV_param_50_25_15.sav', 'rb') as f:  # Python 3: open(..., 'rb')
        model_50_25_15 = pickle.load(f)

    model_list = [model_10_10_20, model_25_10_20, model_50_25_15]
    model_list_name = ["model_10_10_20", "model_25_10_20", "model_50_25_15"]

    for model_idx, model in enumerate(model_list):
        print(model.feature_importances_)

if __name__ == "__main__":
    dataFrame = pd.read_csv(f'amr_datasets_all_class_bin.csv', sep=",")
    dataFrame = dataFrame.drop('!accession', axis=1)
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

    X_train, X_test, y_train, y_test = dataSplitting(dataFrame)

    printMLRFDecisionTree(X_train, X_test, y_train, y_test)

    # useSLRF(X_train, X_test, y_train, y_test)
    # printSLRFDecisionTree(X_train, X_test, y_train, y_test)
    # useMLRF(X_train, X_test, y_train, y_test)
