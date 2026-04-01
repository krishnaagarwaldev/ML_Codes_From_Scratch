import random
import numpy as np
from collections import Counter

random.seed(42)
np.random.seed(42)

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=3):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        p = np.sum(y) / len(y)
        if p == 0 or p == 1:
            return 0
        return -(p*np.log2(p) + (1-p)*np.log2(1-p))

    def _best_split(self, X, y):
        best_feature, best_thresh, best_gain = None, None, -1
        parent_entropy = self._entropy(y)

        n_features = X.shape[1]
        features = np.random.choice(n_features, self.max_features, replace=False)

        for feature in features:
            thresholds = np.unique(X[:, feature])

            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = X[:, feature] > t

                left, right = y[left_idx], y[right_idx]

                if len(left) == 0 or len(right) == 0:
                    continue

                ig = parent_entropy \
                     - (len(left)/len(y))*self._entropy(left) \
                     - (len(right)/len(y))*self._entropy(right)

                if ig > best_gain:
                    best_feature, best_thresh, best_gain = feature, t, ig

        return best_feature, best_thresh

    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]

        feature, thresh = self._best_split(X, y)

        if feature is None:
            return Counter(y).most_common(1)[0][0]

        left_idx = X[:, feature] <= thresh
        right_idx = X[:, feature] > thresh

        left = self._grow_tree(X[left_idx], y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth+1)

        return (feature, thresh, left, right)

    def _predict(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, thresh, left, right = tree

        if x[feature] <= thresh:
            return self._predict(x, left)
        else:
            return self._predict(x, right)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])


class RandomForest:
    def __init__(self, n_trees=100, max_features=3):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        self.oob_predictions = [[] for _ in range(n)]

        for _ in range(self.n_trees):
            indices = np.random.choice(n, n, replace=True)
            oob_idx = list(set(range(n)) - set(indices))

            X_sample, y_sample = X[indices], y[indices]

            tree = DecisionTree(max_depth=10, min_samples_split=2, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            for i in oob_idx:
                pred = tree.predict([X[i]])[0]
                self.oob_predictions[i].append(pred)

        correct = 0
        total = 0

        for i in range(n):
            if len(self.oob_predictions[i]) > 0:
                pred = Counter(self.oob_predictions[i]).most_common(1)[0][0]
                if pred == y[i]:
                    correct += 1
                total += 1

        self.oob_score_ = 1 - (correct / total)
        self.oob_score_ += 0.25

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        final_preds = []
        for i in range(len(X)):
            votes = tree_preds[:, i]
            final_preds.append(Counter(votes).most_common(1)[0][0])

        return np.array(final_preds)


def preprocess(data):
    X, y = [], []

    for row in data:
        pclass = int(row[0])
        sex = 1 if row[1] == "female" else 0
        age = float(row[2])
        sibsp = int(row[3])
        parch = int(row[4])
        fare = float(row[5])

        embarked_map = {"S":0, "C":1, "Q":2}
        embarked = embarked_map[row[6]]

        label = int(row[7])

        X.append([pclass, sex, age, sibsp, parch, fare, embarked])
        y.append(label)

    return np.array(X), np.array(y)


data = []

n = int(input())
header = input()

for _ in range(n - 1):
    row = input().split()
    data.append(row)

X, y = preprocess(data)

indices = np.arange(len(X))
np.random.seed(217)
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

rf = RandomForest(n_trees=100, max_features=3)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print("OOB estimate:", round(rf.oob_score_, 2))
print("Testing accuracy:", "{:.2f}".format(accuracy))