# import math
# import numpy as np

# # Dataset
# dataset = [
#     (35,40,5,0),(42,50,6,0),(50,52,7,0),(60,65,8,0),(67,70,9,1),
#     (75,78,10,1),(80,85,12,1),(90,88,14,1),(95,90,15,1),(100,92,16,1),
#     (110,100,17,1),(120,105,18,1),(130,110,19,1),(140,115,20,1),(150,118,22,1),
#     (160,120,24,1),(175,125,25,1),(190,128,26,1),(210,130,28,1),(230,135,30,1)
# ]

# num_samples = len(dataset)

# # Convert to NumPy array
# data = np.array(dataset, dtype=float)

import numpy as np

# 🔹 Input
n = int(input().strip())
dataset = []

for _ in range(n):
    row = list(map(float, input().split()))
    dataset.append(row)

data = np.array(dataset)

# Split features and labels
X = data[:, :3]
y = data[:, 3]

# 🔹 First 5 rows
print("First 5 rows:")
print("   exam1  exam2  hours_study  admitted")

for i in range(min(5, n)):
    print(f"{i}     {int(X[i][0])}     {int(X[i][1])}           {int(X[i][2])}         {int(y[i])}")

# 🔹 Shape
print(f"\nShape (N, d): ({n}, 4)")

# 🔹 Summary stats
means = np.mean(X, axis=0)
# stds = np.std(X, axis=0) #wrong answer because this is sample data
stds_2 = np.std(X, axis=0, ddof=1)
min = np.min(X, axis=0)
max = np.max(X, axis=0)

print("\nSummary statistics:")

feature_names = ["exam1", "exam2", "hours_study"]

# for i, name in enumerate(feature_names):
#     print(
#         f"{name} -> Min: {int(np.min(X[:,i]))}, "
#         f"Max: {int(np.max(X[:,i]))}, "
#         f"Mean: {means[i]:.2f}, "
#         f"Std: {stds_2[i]:.2f}"
#     )
#or
for i in range(X.shape[1]):
    print(
        f"{feature_names[i]} -> Min: {int(np.min(X[:,i]))}, "
        f"Max: {int(np.max(X[:,i]))}, "
        f"Mean: {means[i]:.2f}, "
        f"Std: {stds_2[i]:.2f}"
    )

stds = np.std(X, axis=0) #Do not use sample std for standardization, use population std (ddof=0)
# 🔹 Standardization
X_norm = (X - means) / stds
X_bias = np.c_[np.ones(n), X_norm]

# 🔹 Sigmoid
def sigmoid(z):
    # z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 🔹 Gradient Descent
theta = np.zeros(4)
lr = 0.01

for _ in range(1500):
    preds = sigmoid(X_bias @ theta)
    error = preds - y
    grad = (X_bias.T @ error) / n # or grad = np.dot(X_bias.T, error)/n
    theta -= lr * grad  

# 🔹 Loss
eps = 1e-15
preds = sigmoid(X_bias @ theta)
preds = np.clip(preds, eps, 1 - eps)

loss = -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))

# print(f"\nFinal theta: {[round(t,2) for t in theta.tolist()]}") # [0.0, 0.0, 0.0, 0.0]
print(f"\nFinal theta: {theta[0]:.2f} {theta[1]:.2f} {theta[2]:.2f} {theta[3]:.2f}")
print(f"Final loss: {loss:.2f}")

# 🔹 Predictions
test_cases = np.array([
    [72, 80, 11],
    [150, 118, 20]
], dtype=float)

test_norm = (test_cases - means) / stds
test_norm = np.c_[np.ones(len(test_norm)), test_norm]

probs = sigmoid(test_norm @ theta)

print(f"\nPrediction for (exam1=72, exam2=80, hours_study=11): {probs[0]:.2f}")
print(f"Prediction for (exam1=150, exam2=118, hours_study=20): {probs[1]:.2f}")