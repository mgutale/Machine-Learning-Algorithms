import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def predict(self, X):
        predictions = [self._predict(x, self.tree) for x in X]
        return np.array(predictions)
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if depth == self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return {'leaf': True, 'value': leaf_value}
        else:
            feature_indices = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
            best_feature_idx, best_threshold = self._find_best_split(X, y, feature_indices)
            left_idxs = np.where(X[:, best_feature_idx] <= best_threshold)[0]
            right_idxs = np.where(X[:, best_feature_idx] > best_threshold)[0]
            if len(left_idxs) == 0 or len(right_idxs) == 0:
                leaf_value = self._most_common_label(y)
                return {'leaf': True, 'value': leaf_value}
            else:
                left_subtree = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
                right_subtree = self._build_tree(X[right_idxs], y[right_idxs], depth+1)
                return {'leaf': False, 'feature_idx': best_feature_idx, 'threshold': best_threshold,
                        'left': left_subtree, 'right': right_subtree}
    
    def _find_best_split(self, X, y, feature_indices):
        best_gini = 1
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
                right_idxs = np.where(X[:, feature_idx] > threshold)[0]
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                else:
                    gini = self._gini_index(y[left_idxs], y[right_idxs])
                    if gini < best_gini:
                        best_gini = gini
                        best_feature_idx = feature_idx
                        best_threshold = threshold
        return best_feature_idx, best_threshold
    
    def _gini_index(self, y_left, y_right):
        n_left, n_right = len(y_left), len(y_right)
        gini_left = 1 - np.sum((np.unique(y_left, return_counts=True)[1] / n_left)**2)
        gini_right = 1 - np.sum((np.unique(y_right, return_counts=True)[1] / n_right)**2)
        gini = (n_left * gini_left + n_right * gini_right) / (n_left + n_right)
        return gini
    
    def _most_common_label(self, y):
        return np.bincount(y).argmax()
    
    def _predict(self, x, tree):
        if tree['leaf']:
            return tree['value']
        else:
            if x[tree['feature_idx']] <= tree['threshold']:
                return self._predict(x, tree['left'])
            else:
                return self._predict(x, tree['right'])

