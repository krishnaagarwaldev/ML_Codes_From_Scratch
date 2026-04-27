
import numpy as np

def entropy(y):
    if len(y) == 0:
        return 0
    unique_labels, counts = np.unique(y, return_counts = True)
    p = counts/len(y)
    return -np.sum(p*np.log2(p + 1e-9))
    

def gini(y):
    if len(y) == 0:
        return 0
    unique_labels, counts = np.unique(y, return_counts = True)
    p = counts/len(y)
    return 1 - np.sum(p**2)

def build_tree(X, y, depth, max_depth, min_leaf, criterion):
    # Base Condition
    if len(np.unique(y)) == 1 or depth >= max_depth or len(y) <= min_leaf:
        values, counts = np.unique(y, return_counts = True)
        return values[np.argmax(counts)]
        
    impurity = entropy(y) if criterion == "e" else gini(y)
    
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    n_samples, n_features = X.shape
    
    for f in range(n_features):
        values = np.unique(X[:, f])
        
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i+1]) / 2
            
            left_mask = X[:, f] <= threshold
            right_mask = X[:, f] > threshold
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            if criterion == 'e':
                imp_left = entropy(y_left)
                imp_right = entropy(y_right)
            else:
                imp_left = gini(y_left)
                imp_right = gini(y_right)
                
            weighted_imp = (len(y_left)/n_samples) * imp_left + (len(y_right)/n_samples)*imp_right
            
            gain = impurity - weighted_imp
            
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = threshold
                
    
    if best_feature is None:
        values, counts = np.unique(y, return_counts = True)
        return values[np.argmax(counts)]
        
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = X[:, best_feature] > best_threshold
    
    left_tree = build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_leaf, criterion)
    right_tree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth, min_leaf, criterion)
    
    return (best_feature, best_threshold, left_tree, right_tree)
    
    
def predict_one(tree, x):
    if not isinstance(tree, tuple):
        return tree
    
    feature, threshold, left, right = tree
    
    if x[feature] <= threshold:
        return predict_one(left, x)
    else:
        return predict_one(right, x)
        
def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])
    
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)



n = int(input())
data = []
        
for _ in range(n):
    row = list(map(float, input().split()))
    data.append(row)
    
data = np.array(data)
X = data[:, :-1]
y = data[:, -1].astype(int)

n1 = int(0.7*n)
n2 = int(0.85*n)

X_train, y_train = X[:n1], y[:n1]
X_val, y_val = X[n1:n2], y[n1:n2]
X_test, y_test = X[n2:], y[n2:]

for criterion in ["e", "g"]:
    best_val_acc = -1
    best_depth = 0
    best_leaf = 0
    best_tree = None
    
    for depth in [2,3,4,5,6]:
        for min_leaf in [1,3,5,10]:
            
            tree = build_tree(X_train, y_train, 0, depth, min_leaf, criterion)
            
            val_pred = predict(tree, X_val)
            val_acc = accuracy(y_val, val_pred)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_depth = depth
                best_leaf = min_leaf
                best_tree = tree
                
    test_pred = predict(best_tree, X_test)
    test_acc  = accuracy(y_test, test_pred)
    
    name = "entropy" if criterion == "e" else "gini"
    print(f"Best ({name}): depth={best_depth}, minleaf={best_leaf}, val_acc={best_val_acc:.2f}, test_acc={test_acc:.2f}")                
                

    