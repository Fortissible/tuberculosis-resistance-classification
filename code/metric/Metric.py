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

def metric_get_predict_proba(test_pred_leaf_loc_list, test_pred_rmg_list):
    test_pred_proba = []
    for _ in range(len(test_pred_leaf_loc_list[0])):
        test_pred_proba.append([0.0, 0.0, 0.0, 0.0])
    test_pred_proba = np.array(test_pred_proba)
    for tree_idx, tree_leafs in enumerate(test_pred_leaf_loc_list):
        unique_leaf, unique_leaf_counts = np.unique(np.array(tree_leafs), return_counts=True)
        for leaf in unique_leaf:
            leaf_idx = np.where(np.array(tree_leafs) == leaf)  # get all index of the unique leaf location
            leaf_value = np.array(test_pred_rmg_list)[tree_idx][leaf_idx]  # masih list isi 4 (resisten multilabel)
            sliced_leaf_value = np.array(leaf_value)[:, :, 1]
            inh_leaf_value = np.array(sliced_leaf_value[:, 0])
            inh_leaf_prob = np.count_nonzero(inh_leaf_value == 1) / len(inh_leaf_value)
            rif_leaf_value = np.array(sliced_leaf_value[:, 1])
            rif_leaf_prob = np.count_nonzero(rif_leaf_value == 1) / len(rif_leaf_value)
            emb_leaf_value = np.array(sliced_leaf_value[:, 2])
            emb_leaf_prob = np.count_nonzero(emb_leaf_value == 1) / len(emb_leaf_value)
            pza_leaf_value = np.array(sliced_leaf_value[:, 3])
            pza_leaf_prob = np.count_nonzero(pza_leaf_value == 1) / len(pza_leaf_value)
            leaf_pred_proba = [inh_leaf_prob, rif_leaf_prob, emb_leaf_prob, pza_leaf_prob]
            for idx, i in enumerate(leaf_pred_proba):
                test_pred_proba[leaf_idx, idx] += i
    mean_test_pred_proba = test_pred_proba / len(test_pred_leaf_loc_list)
    return mean_test_pred_proba

def metric_confusion_matrix(y_actual_class, y_pred_class, figure_name, visualize=False):
    # TRAINING METRIC

    conf_matrix_train = confusion_matrix(y_true=y_actual_class,
                                         y_pred=y_pred_class)
    TN_train = conf_matrix_train[0][0]
    FN_train = conf_matrix_train[1][0]
    TP_train = conf_matrix_train[1][1]
    FP_train = conf_matrix_train[0][1]

    print(f"TP {TP_train}, TN {TN_train}, FP {FP_train}, FN {FN_train}")

    if visualize:
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.matshow(conf_matrix_train, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(conf_matrix_train.shape[0]):
            for j in range(conf_matrix_train.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix_train[i, j],
                        va='center', ha='center', size='xx-large'
                        )

        plt.xlabel('Predictions', fontsize=11)
        plt.ylabel('Actuals', fontsize=11)
        plt.title(f'Confussion matrix {figure_name}', fontsize=11)
        plt.savefig(f'Confussion matrix {figure_name}.png')
        plt.show()

def metric_all_score(y_actual_class, y_pred_class):
    print('Accuracy: %.3f' % accuracy_score(y_actual_class,
                                            y_pred_class))
    print('Precision: %.3f' % precision_score(y_actual_class,
                                              y_pred_class))
    print('Recall: %.3f' % recall_score(y_actual_class,
                                        y_pred_class))
    print('F1 Score: %.3f' % f1_score(y_actual_class,
                                      y_pred_class))

    print('Fbeta-macro Score: %.3f' % fbeta_score(y_actual_class,
                                                  y_pred_class,
                                                  average='macro', beta=0.5))
    print('Fbeta-micro Score: %.3f' % fbeta_score(y_actual_class,
                                                  y_pred_class,
                                                  average='micro', beta=0.5))
    print('Fbeta-weighted Score: %.3f' % fbeta_score(y_actual_class,
                                                     y_pred_class,
                                                     average='weighted', beta=0.5))

def metric_roc_auc(y_test, y_pred_prob, model_name="", model_type="MLRF"):
    figure, axis = plt.subplots(1, 1)

    for idx, col in enumerate(y_test.columns):
        fpr, tpr, thresholds = roc_curve(y_test[col], y_pred_prob[:,idx], pos_label=1)
        roc_auc = roc_auc_score(y_test[col], y_pred_prob[:,idx])

        # Plot the ROC curve
        axis.plot(fpr, tpr, label=f'{model_type} {col} ROC curve (area = %0.3f)' % roc_auc)
        # roc curve for tpr = fpr

    axis.plot([0, 1], [0, 1], 'k--', label='Random classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title(f'{model_type}-{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_type}-{model_name} ROC Curve.png')
    plt.show()
