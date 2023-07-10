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
                 min_split_gain=0.005, colsample_bytree=None, subsample=0.5, random_state=None,
                 classifier_type="gini", multilabel=False):
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
         classifier_type:
              Select method for node splitting "gini" or "information_gain"
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
        self.classifier_type = classifier_type
        self.multilabel = multilabel

    def fit(self, datasets, targets):
        """Model training entry"""
        #Check if the targets is 1D Array (Series data type)
        #and convert into dataframe type
        if type(targets)==pd.core.series.Series:
          assert targets.unique().__len__() >= 2, "There must be two class for targets!"
          targets = targets.to_frame()

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # Two column sampling methods
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(datasets[0].columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(datasets[0].columns)))
        else:
            self.colsample_bytree = len(datasets[0].columns)

        # Build multiple decision trees in parallel
        self.trees = Parallel(n_jobs=-1, verbose=0, backend="loky")(
            delayed(self._parallel_build_trees)(datasets, targets, random_state)
                for random_state in random_state_stages)

    def _parallel_build_trees(self, datasets, targets, random_state):
        """
        bootstrap has put back sampling to
        generate a training sample set and build a decision tree
        """
        subcol_index = random.sample(datasets[0].columns.tolist(), self.colsample_bytree)

        datasets_stage = []
        for dataset in datasets :
            dataset_stage = dataset.sample(n=int(self.subsample * len(dataset)), replace=True,
                                          random_state=random_state).reset_index(drop=True)
            dataset_stage = dataset_stage.loc[:, subcol_index]
            datasets_stage.append(dataset_stage)

        targets_stage = targets.sample(n=int(self.subsample * len(dataset)), replace=True,
                                        random_state=random_state).reset_index(drop=True)

        tree = self._build_single_tree(datasets_stage, targets_stage, depth=0)

        # -------------- PRINT BEST NODE --------------
        # print(tree.describe_tree())

        return tree

    def _build_single_tree(self, datasets, targets, depth):
        """Recursively build a decision tree"""
        # If the categories of the node
        # are all the same/the samples are less than
        # the minimum number of samples required for splitting,
        # select the category with the most occurrences.
        # Termination of division/split
        if targets.shape[1]==1:
          if targets.shape[1]==1 and \
          ( len(targets[targets.columns[0]].unique()) <= 1 or
           datasets[0].__len__() <= self.min_samples_split ):
              tree = Tree()
              # tree.leaf_value = self.calc_leaf_value(targets[targets.columns[0]],self.multilabel)
              tree.leaf_value = self.calc_leaf_value(targets,self.multilabel)
              return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(datasets, targets)
            left_datasets, right_datasets, left_targets, right_targets = \
                self.split_dataset(datasets, targets, best_split_feature, best_split_value)

            tree = Tree()
            # If after the parent node is split,
            # the left leaf node/right leaf node sample is less than
            # the set minimum number of leaf node samples,
            # the parent node will terminate the split
            if left_datasets[0].__len__() <= self.min_samples_leaf or \
                    right_datasets[0].__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                # tree.leaf_value = self.calc_leaf_value(targets[targets.columns[0]],self.multilabel)
                tree.leaf_value = self.calc_leaf_value(targets,self.multilabel)
                return tree
            else:
                # If this feature is used when splitting,
                # the importance of this feature will be increased by 1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._build_single_tree(left_datasets, left_targets, depth+1)
                tree.tree_right = self._build_single_tree(right_datasets, right_targets, depth+1)
                return tree
        # If the depth of the tree exceeds the preset value, terminate the split
        else:
            tree = Tree()
            # tree.leaf_value = self.calc_leaf_value(targets[targets.columns[0]],self.multilabel)
            tree.leaf_value = self.calc_leaf_value(targets,self.multilabel)
            return tree

    def choose_best_feature(self, datasets, targets):
        """
        Find the best data set division method,
        find the optimal split feature,
        split threshold, split gain
        """
        best_split_gain = 1 * len(targets.columns)
        best_split_infogain = 0
        best_split_feature = None
        best_split_value = None

        if self.classifier_type == "gini":
            for feature in datasets[0].columns:
                split_gains = 0
                for idx, target_column in enumerate(targets.columns):
                    if datasets[idx][feature].unique().__len__() <= 100:
                        unique_values = sorted(datasets[idx][feature].unique().tolist())
                    # If the dimension feature has too many values,
                    # select the 100th percentile value as the split threshold to be selected
                    else:
                        unique_values = np.unique([np.percentile(datasets[idx][feature], x)
                                                  for x in np.linspace(0, 100, 100)]
                                                  )

                    # Calculate the splitting gain for the possible splitting thresholds,
                    # and select the threshold with the largest gain


                    # get subset of targets that <= split value into left_targets
                    left_targets = targets[datasets[idx][feature] <= 0]
                    # get subset of targets that > split value into right_targets
                    right_targets = targets[datasets[idx][feature] > 0]

                    # print(feature,
                    #       targets.columns[1],
                    #       np.array(targets).shape,
                    #       np.array(right_targets).shape,
                    #       np.array(left_targets).shape,
                    #       )

                    split_gain = self.calc_gini(left_targets[target_column], right_targets[target_column])
                    split_gains += split_gain

                if split_gains < best_split_gain:
                    best_split_feature = feature
                    best_split_value = 0
                    best_split_gain = split_gains


                    # for split_value in unique_values:
                    #     # get subset of targets that <= split value into left_targets
                    #     left_targets = targets[dataset[feature] <= split_value]
                    #     # get subset of targets that > split value into right_targets
                    #     right_targets = targets[dataset[feature] > split_value]
                    #     split_gain = self.calc_gini(left_targets[targets.columns[0]], right_targets[targets.columns[0]])

                    #     if split_gain < best_split_gain:
                    #         best_split_feature = feature
                    #         best_split_value = split_value
                    #         best_split_gain = split_gain
                    #     print(best_split_value)




        #################
        #      FIX      #
        #     JOINT     #
        #   INFO_GAIN   #
        #    DIBAWAH    #
        #################
        else :
            # Get marginal entropy for each class
            mgn_entropys = self.get_marginal_ent(targets)

            joint_cond_entropys = []
            joint_val_cond_entropys = []

            # Get conditional entropy for each class label
            for class_header_idx,class_header in enumerate(targets.columns):
              #y_label = y[class_header].value_counts()
              y_label = targets[class_header]
              cond_entropys,val_cond_entropys =\
                  self.get_conditional_entropy(datasets,y_label)

              joint_cond_entropys.append(
                  cond_entropys
              )

              joint_val_cond_entropys.append(
                  val_cond_entropys
              )

            # Get Joint Information gain for all attribute/feature
            joint_information_gains = []
            for cond_idx, cond_entropys in enumerate(joint_cond_entropys[0]):
              joint_information_gain = 0
              for mgn_idx,mgn_entropy in enumerate(mgn_entropys):
                joint_information_gain += mgn_entropy-joint_cond_entropys[mgn_idx][cond_idx]
              joint_information_gains.append(joint_information_gain)

            np_joint_information_gains = np.array(joint_information_gains)
            max_ig = np.argmax(np_joint_information_gains)
            best_split_feature = datasets.columns[max_ig]
            best_split_infogain = np_joint_information_gains[max_ig]
            best_split_gain = best_split_infogain
            best_split_value = np.min(joint_val_cond_entropys[0][max_ig])

        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def calc_leaf_value(targets, multilabel):
        """
        Select the category with the most occurrences
        in the sample as the value of the leaf node
        """
        # label_counts = collections.Counter(targets)
        # major_label = max(zip(label_counts.values(), label_counts.keys()))
        # print("major_label",major_label)
        # print("major_label[1]",major_label[1])
        # # return value dari resistance x (true/false)
        # return major_label[1]


        all_major_label = []
        for target_label in targets:
          label_counts = collections.Counter(targets[target_label])
          major_label = max(zip(label_counts.values(), label_counts.keys()))
          all_major_label.append(major_label)
        # # return list value dari multilabel resistance (list(true/false))
        return all_major_label

    # Information Gain METHOD - Conditional Entropy
    @staticmethod
    def get_conditional_entropy(x,y):

        cond_entropys = []
        val_cont_entropys = []
        attrs = x.columns
        class_label_name = y.name

        for attr in attrs:

          # count all unique value of an attribute
          attr_vals = x[attr].value_counts()
          cond_entropy = []

          for attr_val in range(len(attr_vals)):

            # print attribute name and the selected value
            # print(attr, attr_val)

            # get index of specific attribute value e.g idx of (Outlook == Sunny)
            val_idx = x[x[attr]==attr_val].index
            # print(val_idx)

            # total count of instance with specific attribute value
            val_cnt = len(val_idx)
            val_prob = -1*val_cnt/len(y)

            # get information of class label based on specific attr value idx
            val_labels = y[val_idx].value_counts()
            # print(val_labels)
            tmp = 0

            for val_label_idx, val_label in enumerate(val_labels):
              if (val_label == val_cnt):
                tmp += 0
              else :
                val_label_prob = val_label/val_cnt
                tmp += (val_label_prob* math.log2(val_label_prob))

              # count total label for each attr value
              # print(f"label {val_label_idx} count" \
              #   f" in attr {attr}-{attr_val} : {val_label}"
              # )

            #################
            #     SEARCH    #
            #      FOR      #
            #     LOWEST    #
            #################
            attr_val_ent = val_prob * tmp

            cond_entropy.append(attr_val_ent)
            # print(f"{val_prob * tmp}\n")

          #print(f"conditional entropy for {class_label_name}|{attr}:",np.sum(np.array(cond_entropy)))
          #print("\n")
          val_cont_entropys.append(cond_entropy)
          cond_entropys.append(np.sum(np.array(cond_entropy)))

        #print(cond_entropys)
        return cond_entropys,val_cont_entropys

    # Information Gain METHOD - Marginal Entropy
    @staticmethod
    def get_marginal_ent(y):
        mgn_entropys = []
        for class_header in y.columns:
          y_label = y[class_header].value_counts()
          mgn_entropy = 0
          for idx in range(len(y_label)):
            prob = y_label[idx]/len(y)
            mgn_entropy += (prob * math.log2(prob))
          mgn_entropys.append(-1*mgn_entropy)
          #print(f"marginal entropy {class_header}: {mgn_entropy}\n")
        return mgn_entropys

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
    def split_dataset(datasets, targets, split_feature, split_value):
        """
        Divide the sample into left and right parts according to the
        characteristics and threshold, the left is less than or
        equal to the threshold, and the right is greater than the threshold
        """
        left_datasets = []
        right_datasets = []

        for dataset in datasets:
          left_dataset = dataset[dataset[split_feature] <= split_value]
          left_datasets.append(left_dataset)
          right_dataset = dataset[dataset[split_feature] > split_value]
          right_datasets.append(right_dataset)

        left_targets = targets[dataset[split_feature] <= split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_datasets, right_datasets, left_targets, right_targets

    def predict(self, dataset):
        """Input sample, predict category"""

        res = []
        probs = []
        for _, row in dataset.iterrows():
            preds_list = []
            # Count the prediction results of each tree,
            # and select the result with the most occurrences
            # as the final category
            for tree in self.trees:
                preds_list.append(tree.calc_predict_value(row))

            preds_list = np.array(preds_list)
            temp_res = []
            for label_num in range(len(preds_list[0])):
                pred_list = preds_list[:,label_num]
                separated_pred_label_unique, separated_pred_label_counts = np.unique(pred_list, return_counts=True)
                pred_label = separated_pred_label_unique[
                    separated_pred_label_counts == separated_pred_label_counts.max()
                    ]
                temp_res.append(pred_label[0].item())
            res.append(temp_res)
            probs.append(preds_list)
        return np.array(res), probs