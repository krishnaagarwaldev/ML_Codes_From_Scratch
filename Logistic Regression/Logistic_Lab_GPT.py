import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Step 1: Set Random Seed
# ------------------------------------------------
np.random.seed(82)

# ------------------------------------------------
# Step 2: Generate Synthetic Dataset (Gaussian)
# ------------------------------------------------

n = 400
half = n // 2

# Class 0 centered at (-2, -2)
class0 = np.random.randn(half, 2) + np.array([-2, -2])

# Class 1 centered at (2, 2)
class1 = np.random.randn(half, 2) + np.array([2, 2])

# Stack data
X = np.vstack((class0, class1))

# Create labels
y = np.vstack((np.zeros((half,1)), np.ones((half,1))))

# Shuffle dataset properly
indices = np.random.permutation(n)
X = X[indices]
y = y[indices]

# ------------------------------------------------
# Step 3: Train-Test Split (75%-25%)
# ------------------------------------------------

split = int(0.75 * n)

X_train = X[:split]
X_test  = X[split:]

y_train = y[:split]
y_test  = y[split:]

# Add bias column
X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
X_test  = np.hstack((np.ones((X_test.shape[0],1)), X_test))

# ------------------------------------------------
# Step 4: Initialize Parameters
# ------------------------------------------------

w = np.zeros((X_train.shape[1], 1))
alpha = 0.01
epochs = 2000
loss_history = []

# ------------------------------------------------
# Step 5: Sigmoid Function
# ------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ------------------------------------------------
# Step 6: Binary Cross Entropy Loss
# ------------------------------------------------

def binary_cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    epsilon = 1e-5
    loss = -(1/m) * np.sum(
        y_true*np.log(y_pred + epsilon) +
        (1 - y_true)*np.log(1 - y_pred + epsilon)
    )
    return loss

# ------------------------------------------------
# Step 7: Gradient Descent Training
# ------------------------------------------------

for i in range(epochs):

    y_hat = sigmoid(X_train @ w)

    loss = binary_cross_entropy(y_train, y_hat)
    loss_history.append(loss)

    m = X_train.shape[0]
    gradient = (1/m) * (X_train.T @ (y_hat - y_train))

    w = w - alpha * gradient

# ------------------------------------------------
# Step 8: Plot Learning Curve
# ------------------------------------------------

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.show()

# ------------------------------------------------
# Step 9: Evaluate Model
# ------------------------------------------------

y_pred = sigmoid(X_test @ w)
y_pred = (y_pred >= 0.5).astype(int)

accuracy = np.mean(y_pred == y_test) * 100

print("Learned Weights:\n", w)
print("Test Accuracy:", accuracy, "%")