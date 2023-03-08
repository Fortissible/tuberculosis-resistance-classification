import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, l_node=None, r_node=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.l_node = l_node
        self.r_node = r_node
        self.value = value

    def is_a_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_features=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, x, y):
        self.n_features = x.shape[1] if not self.n_features else min(x.shape[1], self.n_features)
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x, y, depth=0):
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        # cek stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # find best split using gain
        best_feature, best_thresh = self._best_split(x, y, feat_idxs)

        # create child Node
        l_idxs, r_idxs = self._split(x[:, best_feature], best_thresh)
        left = self._grow_tree(x[l_idxs, :], y[l_idxs], depth + 1)
        right = self._grow_tree(x[r_idxs, :], y[r_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, x, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            x_col = x[:, feat_idx]
            thresholds = np.unique(x_col)
            for threshold in thresholds:
                # calc gain
                gain = self._information_gain(y, x_col, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def _information_gain(self, y, x_col, threshold):
        # parent node entropy
        parent_entropy = self._entropy(y)

        # create child node
        l_idxs, r_idxs = self._split(x_col, threshold)
        if len(l_idxs) == 0 or len(r_idxs) == 0:
            return 0

        # calc weighted avg. of children node entropy
        n = len(y)
        n_l, n_r = len(l_idxs), len(r_idxs)
        e_l, e_r = self._entropy(y[l_idxs]), self._entropy(y[r_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calc the information gain
        info_gain = parent_entropy - child_entropy
        return info_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _split(self, x_col, split_threshold):
        l_idxs = np.argwhere(x_col <= split_threshold).flatten()
        r_idxs = np.argwhere(x_col > split_threshold).flatten()
        return l_idxs, r_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, xs):
        return np.array([self._traverse_tree(x, self.root) for x in xs])

    def _traverse_tree(self, x, node):
        if node.is_a_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.l_node)
        return self._traverse_tree(x, node.r_node)
