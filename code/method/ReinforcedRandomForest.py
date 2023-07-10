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

from .DecisionTree import Tree

class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree=None, subsample=0.8, random_state=None):
        """
        Random Forest Parameters
         ----------
         n_estimators:
              number of trees
         max_depth:
              tree depth, -1 means unlimited depth
         min_samples_split:
              The minimum number of samples required for node splitting,
              the node terminates splitting if it is less than this value
         min_samples_leaf:
              The minimum sample number of leaf nodes,
              less than this value leaves are merged
         min_split_gain:
              The minimum gain required for splitting,
              less than this value the node terminates the split
         colsample_bytree:
              Column sampling setting, which can be [sqrt, log2].
              sqrt means randomly selecting sqrt(n_features) features,
              log2 means to randomly select log(n_features) features,
              if set to other, column sampling will not be performed
         subsample:
              line sampling ratio
         random_state:
              Random seed, after setting,
              the n_estimators sample sets generated each time will not change,
              ensuring that the experiment can be repeated
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.random_state = random_state
        self.trees = None
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        """Model training entry"""

        if targets.unique().__len__() < 2 :
          return None,None,None,None,None

        # assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # Two column sampling methods
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)

        # Build multiple decision trees in parallel
        self.trees = Parallel(n_jobs=-1, verbose=0, backend="threading")(
            delayed(self._parallel_build_trees)(dataset, targets, random_state)
                for random_state in random_state_stages)

    def _parallel_build_trees(self, dataset, targets, random_state):
        """
        bootstrap has put back sampling to
        generate a training sample set and build a decision tree
        """
        subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
        dataset_stage = dataset.sample(n=int(self.subsample * len(dataset)), replace=True,
                                        random_state=random_state).reset_index(drop=True)
        dataset_stage = dataset_stage.loc[:, subcol_index]
        targets_stage = targets.sample(n=int(self.subsample * len(dataset)), replace=True,
                                        random_state=random_state).reset_index(drop=True)

        tree = self._build_single_tree(dataset_stage, targets_stage, depth=0)

        # -------------- PRINT BEST NODE --------------
        # print(tree.describe_tree())

        return tree

    def _build_single_tree(self, dataset, targets, depth):
        """Recursively build a decision tree"""
        # If the categories of the node
        # are all the same/the samples are less than
        # the minimum number of samples required for splitting,
        # select the category with the most occurrences.
        # Termination of division/split
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            # If after the parent node is split,
            # the left leaf node/right leaf node sample is less than
            # the set minimum number of leaf node samples,
            # the parent node will terminate the split
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:
                # If this feature is used when splitting,
                # the importance of this feature will be increased by 1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._build_single_tree(left_dataset, left_targets, depth+1)
                tree.tree_right = self._build_single_tree(right_dataset, right_targets, depth+1)
                return tree
        # If the depth of the tree exceeds the preset value, terminate the split
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    def choose_best_feature(self, dataset, targets):
        """
        Find the best data set division method,
        find the optimal split feature,
        split threshold, split gain
        """
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            # If the dimension feature has too many values,
            # select the 100th percentile value as the split threshold to be selected
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            # Calculate the splitting gain for the possible splitting thresholds,
            # and select the threshold with the largest gain
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(left_targets['label'], right_targets['label'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def calc_leaf_value(targets):
        """
        Select the category with the most occurrences
        in the sample as the value of the leaf node
        """
        label_counts = collections.Counter(targets)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    @staticmethod
    def calc_gini(left_targets, right_targets):
        """
        The classification tree uses the Gini index as an
        indicator to select the optimal split point
        """
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            # Count how many samples are in each category,
            # and then calculate gini
            label_counts = collections.Counter(targets)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        """
        Divide the sample into left and right parts according to the
        characteristics and threshold, the left is less than or
        equal to the threshold, and the right is greater than the threshold
        """
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, dataset, labels):
        """Input sample, predict category"""
        res = []
        tree_margin_list = np.zeros(len(self.trees))
        n_data_margin_list = []
        tree_rmg_list = []

        missclassified_data_idx = []

        for _ in range(len(self.trees)):
          tree_rmg_list.append([])
        for idx, row in dataset.iterrows():
            pred_list = []

            # Count the prediction results of each tree,
            # and select the result with the most occurrences
            # as the final category
            for tree_idx, tree in enumerate(self.trees):
                pred = tree.calc_predict_value(row)
                tree_rmg_list[tree_idx].append(pred)
                # count tree margin
                if pred == labels[idx]:
                  tree_margin_list[tree_idx] += 1
                else :
                  tree_margin_list[tree_idx] -= 1

                pred_list.append(pred)
            unique, counts = np.unique(np.array(pred_list), return_counts=True)

            # data point margin on the current forest
            # and append data point margin into n-data margin
            if len(unique)==1 and unique[0] == labels[idx] :
              n_data_margin_list.append(1.0)
            elif len(unique)==1 and unique[0] != labels[idx] :
              n_data_margin_list.append(-1.0)
            else :
              prob = counts[labels[idx]]/len(pred_list)
              margin = prob - (1-prob)
              n_data_margin_list.append(margin)

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])

            # Check missclassified data here
            if labels[idx] != pred_label[1] :
              missclassified_data_idx.append(idx)

        # make a copy of actual labels into n_tree x m_data_labels array
        labels_copy = np.tile(labels,(len(self.trees),1))

        # xnor function (if label == pred, value = 1) ==> same as rmg per tree function
        # the result is arr of resulting weight
        # (if same as label==1 else==0, per n_data for all tree in forest)
        tree_rmg_list = ~(np.logical_xor(np.array(tree_rmg_list),labels_copy))
        tree_rmg_list = tree_rmg_list.astype(int)

        return np.array(res), n_data_margin_list, tree_margin_list, tree_rmg_list, missclassified_data_idx