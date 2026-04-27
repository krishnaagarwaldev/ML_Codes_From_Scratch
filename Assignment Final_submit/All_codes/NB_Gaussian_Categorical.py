"""
6
25 50000 | Sunny Hot | 0
30 60000 | Sunny Hot | 0
22 45000 | Overcast Mild | 1
35 80000 | Rain Cool | 1
28 52000 | Sunny Mild | 1
40 90000 | Rain Hot | 0
25 50000 | Sunny Mild | 0
"""

import numpy as np

# =========================
# INPUT
# =========================
# Format:
# num1 num2 ... | cat1 cat2 ... | label

n = int(input())
dataset = []

for _ in range(n):
    line = input().split('|')
    
    nums = list(map(float, line[0].strip().split()))
    cats = line[1].strip().split()
    label = int(line[2].strip())
    
    dataset.append((nums, cats, label))


# =========================
# SPLIT
# =========================
idx = int(0.7 * n)
train_data = dataset[:idx]
test_data = dataset[idx:]

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}\n")


# =========================
# PRIORS
# =========================
y_train = [label for _, _, label in train_data]
classes = list(set(y_train))

class_counts = {}
class_priors = {}

for cls in classes:
    count = y_train.count(cls)
    class_counts[cls] = count
    class_priors[cls] = count / len(train_data)
    print(f"Class {cls} Prior: {class_priors[cls]:.2f}")


# =========================
# NUMERICAL (GAUSSIAN)
# =========================
X_train = np.array([d[0] for d in train_data])
y_train_np = np.array(y_train)

mean_per_class = {}
std_per_class = {}

for cls in classes:
    data = X_train[y_train_np == cls]
    mean_per_class[cls] = np.mean(data, axis=0)
    std_per_class[cls] = np.std(data, axis=0)


def gaussian(x, mean, std):
    std = np.maximum(std, 1e-9)
    exp = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return exp / (np.sqrt(2 * np.pi) * std)


# =========================
# CATEGORICAL (VALUES)
# =========================
num_cat = len(train_data[0][1])

cat_values = []
for i in range(num_cat):
    vals = []
    for _, cats, _ in train_data:
        if cats[i] not in vals:
            vals.append(cats[i])
    cat_values.append(vals)


# =========================
# COUNT FUNCTION
# =========================
def count_cat(cls, index, value):
    count = 0
    for _, cats, label in train_data:
        if label == cls and cats[index] == value:
            count += 1
    return count


# =========================
# PREDICT
# =========================
def predict(num_feat, cat_feat):
    
    best_class = None
    best_score = -1e9
    
    for cls in classes:
        
        log_prob = np.log(class_priors[cls])
        
        # ---- NUMERICAL PART ----
        prob_num = gaussian(np.array(num_feat), mean_per_class[cls], std_per_class[cls])
        log_prob += np.sum(np.log(prob_num + 1e-9))
        
        # ---- CATEGORICAL PART ----
        for i in range(num_cat):
            count = count_cat(cls, i, cat_feat[i])
            total = class_counts[cls]
            vocab_size = len(cat_values[i])
            
            prob = (count + 1) / (total + vocab_size)
            log_prob += np.log(prob)
        
        if log_prob > best_score:
            best_score = log_prob
            best_class = cls
    
    return best_class


# =========================
# TESTING
# =========================
predictions = []
actual = []

for nums, cats, label in test_data:
    pred = predict(nums, cats)
    predictions.append(pred)
    actual.append(label)

print("\nPredictions:", predictions)
print("Actual:     ", actual)


# =========================
# METRICS
# =========================
tp = tn = fp = fn = 0

for i in range(len(actual)):
    if actual[i] == 1 and predictions[i] == 1:
        tp += 1
    elif actual[i] == 1 and predictions[i] == 0:
        fn += 1
    elif actual[i] == 0 and predictions[i] == 1:
        fp += 1
    elif actual[i] == 0 and predictions[i] == 0:
        tn += 1

accuracy = (tp + tn) / len(actual) if len(actual) else 0
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print(f"\nAccuracy={accuracy:.2f}")
print(f"Precision={precision:.2f}")
print(f"Recall={recall:.2f}")
print(f"F1={f1_score:.2f}")