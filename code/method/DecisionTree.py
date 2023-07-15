# -*- coding: utf-8 -*-
"""
@Env: Python2.7
@Time: 2019/10/24 13:31
@Author: zhaoxingfeng
@Function：Random Forest（RF
@Version: V1.2
[1] UCI. wine[DB/OL].https://archive.ics.uci.edu/ml/machine-learning-databases/wine.
"""

class Tree(object):
    """Define a decision tree"""
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def calc_predict_value(self, dataset, depth=0, leaf_loc=0, feature_path=None):
        """Find the leaf node of the sample through the recursive decision tree"""
        if feature_path is None:
            feature_path = []
        if self.leaf_value is not None:
            return leaf_loc, self.leaf_value, feature_path
        elif dataset[self.split_feature] <= self.split_value:
            leaf_loc -= 1*depth
            depth += 1
            return self.tree_left.calc_predict_value(dataset, depth, leaf_loc, feature_path)
        else:
            leaf_loc += 1*depth
            depth += 1
            feature_path.append(self.split_feature)
            return self.tree_right.calc_predict_value(dataset, depth, leaf_loc, feature_path)

    def describe_tree(self):
        """
        Print the decision tree in json form,
        which is convenient for viewing the tree structure
        """
        if not self.tree_left and not self.tree_right:
            leaf_info = {"leaf_value": self.leaf_value}
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = {"split_feature": str(self.split_feature),
                          "split_value": + self.split_value,
                          "left_tree": left_info,
                          "right_tree": right_info}
        return tree_structure

