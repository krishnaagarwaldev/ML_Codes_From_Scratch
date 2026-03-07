import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def add_bias(self, X):
        n = X.shape[0]
        ones = np.ones((n,1))
        return np.hstack((ones, X))

    def fit(self, X, y):
        X = self.add_bias(X)
        n, m = X.shape
        self.w = np.zeros(m)

        for _ in range(self.epochs):
            z = np.dot(X, self.w)
            y_pred = self.sigmoid(z)
            gradient = np.dot(X.T, (y_pred - y)) / n
            self.w -= self.lr * gradient

    def predict(self, X):
        X = self.add_bias(X)
        z = np.dot(X, self.w)
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)

    def binary_cross_entropy(self, X, y):
        X = self.add_bias(X)
        n = X.shape[0]
        y_pred = self.sigmoid(np.dot(X, self.w))
        epsilon = 1e-5
        loss = -(1/n) * np.sum(y*np.log(y_pred + epsilon) + (1-y)*np.log(1-y_pred + epsilon))
        return loss

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ---------------------------------
# Generate Gaussian Dataset
# ---------------------------------

np.random.seed(0)

n = 400
half = n // 2

# Class 0
class0 = np.random.randn(half, 2) + np.array([-2, -2])

# Class 1
class1 = np.random.randn(half, 2) + np.array([2, 2])

X = np.vstack((class0, class1))
y = np.hstack((np.zeros(half), np.ones(half)))

# Shuffle properly
indices = np.random.permutation(n)
X = X[indices]
y = y[indices]

# ---------------------------------
# Train-Test Split (75-25)
# ---------------------------------

train_size = int(0.75 * n)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ---------------------------------
# Train Model
# ---------------------------------

model = LogisticRegression(lr=0.1, epochs=2000)
model.fit(X_train, y_train)

# ---------------------------------
# Evaluate
# ---------------------------------

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
loss = model.binary_cross_entropy(X_test, y_test)

print("Accuracy:", accuracy)
print("Loss:", loss)


# ---------------------------------
# Load Iris dataset and test
# ---------------------------------

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]   # use only 2 features for easy visualization
y = iris.target
print(X[:5], y[:5])

# Keep only class 0 and 1 because logistic regression is binary -> 2 classes, but in iris we have 3 classes (0,1,2)
mask = y != 2
X = X[mask]
y = y[mask]

model = LogisticRegression(lr=0.01, epochs=2000)
model.fit(X, y)
pred = model.predict(X)
accuracy = np.mean(pred == y)
loss = model.binary_cross_entropy(X, y)

print("\nIris Dataset Accuracy:", accuracy)
print("Iris Dataset Loss:", loss)