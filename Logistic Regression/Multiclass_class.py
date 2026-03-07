import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class SoftmaxRegression:

    def __init__(self, lr=0.01, epochs=3000):
        self.lr = lr
        self.epochs = epochs

    def add_bias(self, X):
        n = X.shape[0]
        return np.hstack((np.ones((n,1)), X))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]

    def fit(self, X, y):
        X = self.add_bias(X)
        n, m = X.shape
        num_classes = len(np.unique(y))

        self.W = np.zeros((m, num_classes))
        y_onehot = self.one_hot(y, num_classes)

        for _ in range(self.epochs):
            scores = X @ self.W
            probs = self.softmax(scores)
            gradient = (X.T @ (probs - y_onehot)) / n
            self.W -= self.lr * gradient

    def predict(self, X):
        X = self.add_bias(X)
        scores = X @ self.W
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)


# ---------------------------------
# Load Iris Dataset (3 classes)
# ---------------------------------

iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Train model
model = SoftmaxRegression(lr=0.01, epochs=5000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)

print("Multiclass Accuracy:", accuracy)