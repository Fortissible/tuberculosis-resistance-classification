from code.DecisionTree import DecisionTree
import numpy as np
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_sample_split=self.min_samples_split,
                                n_features=self.n_features)
            x_sample, y_sample = self._bootstrap_samples(x, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, x, y):
        n_samples = x.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return x[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, x):
        preds = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(preds, 0, 1)
        preds = np.array([self._most_common_label(tree_pred) for tree_pred in tree_preds])
        return preds
