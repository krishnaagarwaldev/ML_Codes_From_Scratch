import numpy as np

# # Dataset
# dataset = [
#     (35,0),(42,0),(50,0),(60,0),(67,1),(75,1),(80,1),(90,1),(95,1),(100,1),
#     (110,1),(120,1),(130,1),(140,1),(150,1),(160,1),(175,1),(190,1),(210,1),(230,1)
# ]

# # Separate X and y
# scores = np.array([row[0] for row in dataset], dtype=float)
# labels = np.array([row[1] for row in dataset], dtype=float)

# N = len(scores)

# 🔹 Take input
N = int(input())

scores = []
labels = []

for _ in range(N):
    x, y = map(int, input().split())
    scores.append(x)
    labels.append(y)

scores = np.array(scores)
labels = np.array(labels)

# 🔹 EDA
# print("First 5 rows:")
# print("exam_score admitted")
# for i in range(min(5, N)):
#     print(scores[i], labels[i])
# print(f"\nShape (N, d): ({N}, 2)")

print("First 5 rows:")
print("    exam_score  admitted")
for i in range(min(5, N)):
    # print(f"{i}           {scores[i]:<10} {labels[i]}")  # <10 for left-alignment
    print(f"{i}           {scores[i]}         {labels[i]}")

print(f"\nShape (N, d): ({N}, 2)")

print("\nSummary statistics:")
print(f"Min: {scores.min():.0f}") # or int(scores.min())
print(f"Max: {scores.max():.0f}") # or int(scores.max())
print(f"Mean: {scores.mean():.2f}")
print(f"Std: {scores.std():.2f}")

# 🔹 Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
  
# 🔹 Initialize parameters
theta0 = 0.0
theta1 = 0.0

alpha = 0.01
epochs = 1000

# 🔹 Gradient Descent
for _ in range(epochs):
    z = theta0 + theta1 * scores
    y_hat = sigmoid(z)

    error = y_hat - labels

    # gradients
    d_theta0 = np.sum(error) / N   # N = len(scores)
    d_theta1 = np.sum(error * scores) / N  # or np.mean(error * scores)

    # update
    theta0 -= alpha * d_theta0
    theta1 -= alpha * d_theta1

# 🔹 Loss
epsilon = 1e-15
y_hat = sigmoid(theta0 + theta1 * scores)
y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

loss = -np.mean(labels * np.log(y_hat) + (1 - labels) * np.log(1 - y_hat))

print(f"\nFinal theta0: {theta0:.2f}")
print(f"Final theta1: {theta1:.2f}")
print(f"Final loss: {loss:.2f}")
print()

# 🔹 Predictions
for x in [65, 155]:
    prob = sigmoid(theta0 + theta1 * x)
    print(f"Prediction for exam_score={x}: {prob:.2f}")