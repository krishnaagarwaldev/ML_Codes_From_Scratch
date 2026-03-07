import numpy as np

class DecisionTree:

    def entropy(self, y):
        values, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def information_gain(self, X_column, y):
        total_entropy = self.entropy(y)

        values, counts = np.unique(X_column, return_counts=True)
        weighted_entropy = 0

        for v, count in zip(values, counts):
            y_subset = y[X_column == v]
            weighted_entropy += (count / len(y)) * self.entropy(y_subset)

        return total_entropy - weighted_entropy

    def best_feature(self, X, y):
        gains = []
        for i in range(X.shape[1]):
            gain = self.information_gain(X[:, i], y)
            gains.append(gain)
        return np.argmax(gains)

    def fit(self, X, y):
        self.feature = self.best_feature(X, y)
        self.tree = {}

        values = np.unique(X[:, self.feature])
        for v in values:
            y_subset = y[X[:, self.feature] == v]
            self.tree[v] = max(set(y_subset), key=list(y_subset).count)

    def predict(self, X):
        predictions = []
        for x in X:
            value = x[self.feature]
            predictions.append(self.tree.get(value))
        return np.array(predictions)
    
# ---------------------------------
# Play tennis dataset

data = np.array([
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],    
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],    
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
])

X = data[:, :-1]
y = data[:, -1]

dt = DecisionTree()
dt.fit(X, y)

print(dt.tree)

print(dt.predict(np.array([['Sunny', 'Hot', 'High', 'Weak']])))
print(dt.predict(np.array([['Sunny', 'Mild', 'Normal', 'Weak']])))

# Accuracy on training data
y_pred = dt.predict(X)
accuracy = np.mean(y_pred == y)
print("Training Accuracy:", accuracy)

# Accuracy on test data
X_test = np.array([['Rain', 'Mild', 'High', 'Strong'], ['Sunny', 'Cool', 'Normal', 'Weak']])
y_test = np.array(['Yes', 'No'])
y_pred = dt.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Test Accuracy:", accuracy)

# Confusion Matrix -> Manually calculate confusion matrix for test data
tp = np.sum((y_pred == 'Yes') & (y_test == 'Yes'))
tn = np.sum((y_pred == 'No') & (y_test == 'No'))
fp = np.sum((y_pred == 'Yes') & (y_test == 'No'))
fn = np.sum((y_pred == 'No') & (y_test == 'Yes'))

print("Confusion Matrix:")
print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)    

# Precision and Recall
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("Precision:", precision)
print("Recall:", recall)