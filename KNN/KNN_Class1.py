import numpy as np

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X):
        predictions = []

        for x in X:
            distances = []
            for i in range(len(self.X_train)):
                d = self.distance(x, self.X_train[i])
                distances.append((d, self.y_train[i]))

            distances.sort()
            neighbors = distances[:self.k]

            values = [label for (d, label) in neighbors]
            predictions.append(max(set(values), key=values.count))

        return np.array(predictions)
    
    
# ---------------------------------
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]   # use only 2 features for easy visualization
y = iris.target

# Split dataset into train and test -- 75% train, 25% test
n = X.shape[0]
indices = np.random.permutation(n)
split = int(0.75 * n)
X_train = X[indices[:split]]
y_train = y[indices[:split]]

X_test = X[indices[split:]]
y_test = y[indices[split:]]

# ---------------------------------
knn = KNN(k=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Predictions:", y_pred)
print("True labels:", y_test)
print("Accuracy:", accuracy)
