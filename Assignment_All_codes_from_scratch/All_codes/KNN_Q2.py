import numpy as np  

n = int(input())
features = []
labels = []

for _ in range(n):
    row = input().split() #row will be a list of strings, eg., if the input is "85 90 5 1" then row will be ['85', '90', '5', '1']
    features.append(list(map(float, row[:-1])))
    labels.append(row[-1])
    
X = np.array(features)

unique_labels, y = np.unique(labels, return_inverse = True)
# y = [0 if label == 'setosa' else 1 if label == 'versicolor' else 2 for label in labels]

X_min = X.min(axis=0)
X_max = X.max(axis=0)
  
X = (X - X_min)/(X_max - X_min + 1e-8) # add small value to avoid division by zero

np.random.seed(42)
indices = np.random.permutation(n)

X, y = X[indices], y[indices]

split =int(0.8*n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


def predict(X_train, y_train, sample, k):
    dist = np.sum(np.abs(X_train - sample), axis=1) #manhattan distance
    
    nearest = np.argsort(dist)[:k]
    labels = y_train[nearest]
    
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]
    
def accuracy(X_train, y_train, x_test, y_test, k):
    preds = [predict(X_train, y_train, sample, k) for sample in x_test]
    return np.mean(preds == y_test)
    

folds = 5
fold_size = len(X_train) // folds

k_values = [1, 3, 5, 7, 9]
k_scores = {}

for k in k_values:
    scores = []
    for i in range(folds):
        start = i*fold_size
        end = start + fold_size
        
        X_validation = X_train[start:end]
        y_validation = y_train[start:end]
        
        X_train_new = np.concatenate((X_train[:start], X_train[end:]))
        y_train_new = np.concatenate((y_train[:start], y_train[end:]))
        
        acc = accuracy(X_train_new, y_train_new, X_validation, y_validation, k)
        scores.append(acc)
        
    k_scores[k] = np.mean(scores)

best_scores = max(k_scores.values())
candidates = [k for k in k_scores if k_scores[k] == best_scores]

if len(candidates) > 1 and 1 in candidates:
    candidates.remove(1)
    
best_k = min(candidates)

test_accuracy = accuracy(X_train, y_train, X_test, y_test, best_k)
print("FINAL EVALUATION ON TEST SET")
print(f"Test set accuracy with k={best_k}: {test_accuracy:.2f}")
    
    
    
    
    
        
        
        
        
        
        
        
        















