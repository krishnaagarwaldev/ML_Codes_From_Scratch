import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Step 1: Generate two separable classes
x1 = np.random.randn(200,2) + np.array([-2,2])   # Class 0
x2 = np.random.randn(200,2) + np.array([2,-2])   # Class 1

# Combine
X = np.vstack((x1,x2))

# Labels
y1 = np.zeros((200,1))
y2 = np.ones((200,1))
y = np.vstack((y1,y2))

# Shuffle properly
data = np.hstack((X,y))
np.random.shuffle(data)

X = data[:,:2]
y = data[:,2].reshape(-1,1)

# Train Test Split
X_train = X[:300]
X_test = X[300:]
y_train = y[:300]
y_test = y[300:]

# Initialize
w = np.zeros((2,1))
alpha = 0.001
epochs = 2000
loss_hist = []

# Sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Binary Cross Entropy
def loss_function(y, y_hat):
    n = len(y)
    loss = -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
    return loss.mean()

# Training
for i in range(epochs):
    y_hat = sigmoid(X_train @ w)
    
    loss = loss_function(y_train, y_hat)
    loss_hist.append(loss)
    
    gradient = (1/len(X_train)) * (X_train.T @ (y_hat - y_train))
    w = w - alpha * gradient

# Plot loss
plt.plot(loss_hist)
plt.title("Loss vs Epochs")
plt.show()

# Testing
y_pred = sigmoid(X_test @ w)
y_pred = (y_pred > 0.75)

accuracy = (y_pred == y_test).mean()*100
print("Accuracy:", accuracy)