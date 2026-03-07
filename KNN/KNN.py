import numpy as np

# Sample Training Data (2 features)
X_train = np.array([
    [2, 3],
    [1, 1],
    [4, 5],
    [6, 7],
    [3, 4]
])

y_train = np.array([0, 0, 1, 1, 1])  # Labels

# Test point
X_test = np.array([3, 3])

k = 3

# Step 1: Calculate Euclidean distance
distances = []

for i in range(len(X_train)):
    distance = np.sqrt(np.sum((X_train[i] - X_test)**2))
    distances.append((distance, y_train[i]))

# Step 2: Sort distances
distances.sort()

# Step 3: Take first k labels
k_nearest = distances[:k]

labels = []

for item in k_nearest:
    labels.append(item[1])

# Step 4: Majority voting
prediction = max(set(labels), key=labels.count)

print("Predicted class:", prediction)