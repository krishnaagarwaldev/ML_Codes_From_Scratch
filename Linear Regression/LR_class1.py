import numpy as np

class LinearRegression:

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, m = X.shape
        
        # Add bias column
        ones = np.ones((n,1))
        X = np.hstack((ones, X))
        
        # Initialize weights
        self.w = np.zeros(m+1)

        # Gradient Descent
        for i in range(self.epochs):
            y_pred = np.dot(X, self.w)
            error = y_pred - y
            gradient = np.dot(X.T, error) / n
            self.w = self.w - self.lr * gradient

    def predict(self, X):
        n = X.shape[0]
        ones = np.ones((n,1))
        X = np.hstack((ones, X))
        return np.dot(X, self.w)
    
    def compute_cost(self, X, y):
        m = len(y)
        y_pred = self.predict(X)
        return (1/(2*m)) * np.sum((y_pred - y)**2)



# --------------------------
# Generate Random Data
# --------------------------

np.random.seed(0)

n = 100        # samples
m = 3          # features

X = np.random.rand(n, m)

# Generate random real weights (unknown to model)
real_w = np.random.randn(m + 1)
# real_w = np.zeros(m + 1)

# Add bias column to create y
X_bias = np.hstack((np.ones((n,1)), X))

# Generate y using matrix multiplication
y = np.dot(X_bias, real_w) + np.random.randn(n) * 0.01 # Add some noise


# --------------------------
# Train Model
# --------------------------

model = LinearRegression(lr=0.001, epochs=20000)
model.fit(X, y)

print("Real Weights:   ", real_w)
print("Learned Weights:", model.w)


# --------------------------
# R2 Score
# --------------------------

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

pred = model.predict(X)
print("R2 Score:", r2_score(y, pred)) # Should be close to 1


# --------------------------
# Compute Cost
# --------------------------

print("Cost:", model.compute_cost(X, y)) # Should be close to 0 if model is good



# --------------------------
# Load iris dataset and test
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]   # use only 2 features for easy visualization
y = iris.target

model = LinearRegression(lr=0.01, epochs=20000)
model.fit(X, y)
pred = model.predict(X)
print("\nR2 Score on Iris:", r2_score(y, pred))
print("Cost on Iris:", model.compute_cost(X, y))
