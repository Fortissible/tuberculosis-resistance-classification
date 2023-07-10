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

    def calc_predict_value(self, dataset):
        """Find the leaf node of the sample through the recursive decision tree"""
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    def describe_tree(self):
        """
        Print the decision tree in json form,
        which is convenient for viewing the tree structure
        """
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure