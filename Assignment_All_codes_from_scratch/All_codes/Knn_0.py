import numpy as np

n = int(input())

features = []
labels = []

for _ in range(n):
    row = input().split()
    features.append([float(row[i]) for i in range(4)])
    labels.append(row[4])

# Encode labels
unique_labels, y = np.unique(labels, return_inverse=True)

X = np.array(features)
y = np.array(y)

# Shuffle
np.random.seed(42)
indices = np.arange(n)
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# Euclidean distance
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

eps = 1e-8

def assign_weight(X, sample):
    dist = np.array([euclidean(sample, x) for x in X])
    w = 1 / (dist + eps)
    return w

def predict(X_train, y_train, sample, k):
    weight = assign_weight(X_train, sample)
    neigh_idx = np.argsort(-weight)[:k]

    neigh_labels = y_train[neigh_idx]
    neigh_weights = weight[neigh_idx]

    class_scores = {}
    for label, w in zip(neigh_labels, neigh_weights):
        class_scores[label] = class_scores.get(label, 0) + w

    return max(class_scores, key=class_scores.get)

def accuracy(X_train, y_train, X_val, y_val, k):
    preds = [predict(X_train, y_train, sample, k) for sample in X_val]
    return np.mean(y_val == preds)

# 5-fold CV
folds = 5
fold_size = len(X) // folds

acc_k = {}
k_val = [1, 3, 5, 7, 9]

for k in k_val:
    scores = []
    
    for i in range(folds):
        start = i * fold_size
        end = start + fold_size

        X_val = X[start:end]
        y_val = y[start:end]

        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        # ✅ FIX 1: Normalize using ONLY training data (no data leakage)
        min_val = X_train.min(axis=0)
        max_val = X_train.max(axis=0)

        # ✅ FIX 2: Avoid division by zero
        X_train_norm = (X_train - min_val) / (max_val - min_val + eps)
        X_val_norm = (X_val - min_val) / (max_val - min_val + eps)

        acc = accuracy(X_train_norm, y_train, X_val_norm, y_val, k)
        scores.append(acc)

    acc_k[k] = np.mean(scores)

# Find best k
best_k = None
best_acc = 0

for k1, value in acc_k.items():
    if value > best_acc:
        best_acc = value
        best_k = k1

print(f"Best k={best_k}: CV Mean Accuracy={best_acc:.2f}")