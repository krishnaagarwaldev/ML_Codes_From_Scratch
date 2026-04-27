import numpy as np

n = int(input())
features = []
labels = []

for _ in range(n):
    row = input().split()
    features.append(list(map(float, row[:4])))
    labels.append(row[4])
    

X = np.array(features)

unique_labels, y = np.unique(labels, return_inverse=True)

np.random.seed(42)
indices = np.arange(n)
np.random.shuffle(indices)  

X = X[indices]
y = y[indices]

idx = int(0.8*n)

X_train = X[:idx]
X_test = X[idx:]

y_train = y[:idx]
y_test = y[idx:]

def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

eps = 1e-5  # added for safety

# ✅ ONLY THIS FUNCTION CHANGED
def predict(X_train, y_train, sample, k):
    dist = np.sqrt(np.sum((X_train - sample)**2, axis = 1))
    
    nearest_indices = np.argsort(dist)[:k]
    labels = y_train[nearest_indices]
    nearest_dist = dist[nearest_indices]
    
    # 🔥 weighted voting
    weights = 1 / (nearest_dist + eps)
    
    values = np.unique(labels)
    weight_sum = []
    
    for v in values:
        total = 0
        for label, w in zip(labels, weights):
            if label == v:
                total += w
        weight_sum.append(total)
    
    return values[np.argmax(weight_sum)]

def accuracy(X_train, y_train, X_test, y_test, k):
    preds = [predict(X_train, y_train, sample, k) for sample in X_test]
    return np.mean(preds == y_test)    

folds = 5
fold_size = len(X_train) // folds

k_values = [1,3,5,7,9]
k_scores = {}

for k in k_values:
    fold_accuracies = []
    
    for i in range(folds):
        start = i*fold_size
        end = start + fold_size
        
        X_validation = X_train[start:end]
        y_validation = y_train[start:end]
        
        X_train_new = np.concatenate((X_train[:start], X_train[end:]))
        y_train_new = np.concatenate((y_train[:start], y_train[end:]))
        
        acc = accuracy(X_train_new, y_train_new, X_validation, y_validation, k)
        fold_accuracies.append(acc)
    
    k_scores[k] = np.mean(fold_accuracies)

best_score = max(k_scores.values())
candidates = [k for k in k_scores if k_scores[k] == best_score]

if len(candidates) > 1 and 1 in candidates:
    candidates.remove(1)
    
best_k = min(candidates)

test_accuracy = accuracy(X_train, y_train, X_test, y_test, best_k)

print("FINAL EVALUATION ON TEST SET")
print(f"Test set accuracy with k={best_k}: {test_accuracy:.2f}")