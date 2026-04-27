"""
10
Sunny Hot High Weak 0
Sunny Hot High Strong 0
Overcast Hot High Weak 1
Rain Mild High Weak 1
Rain Cool Normal Weak 1
Rain Cool Normal Strong 0
Overcast Cool Normal Strong 1
Sunny Mild High Weak 0
Sunny Cool Normal Weak 1
Rain Mild Normal Weak 1
"""

import numpy as np

n = int(input())
dataset = []

for _ in range(n):
    row = input().split()
    dataset.append(row)

dataset = np.array(dataset)

# Features and label
X = dataset[:, :-1]
y = dataset[:, -1]

# Train-test split
idx = int(0.7 * len(X))
X_train, X_test = X[:idx], X[idx:]
y_train, y_test = y[:idx], y[idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}\n")

# Unique classes
classes = np.unique(y_train)

# Prior probabilities
class_priors = {}
for cls in classes:
    class_priors[cls] = np.sum(y_train == cls) / len(y_train)

print("Class Priors:")
for cls in classes:
    print(f"{cls}: {class_priors[cls]:.2f}")

# Likelihoods: P(feature_value | class)
likelihoods = {}

for cls in classes:
    X_cls = X_train[y_train == cls]
    likelihoods[cls] = {}

    for col in range(X.shape[1]): # for each feature column
        feature_values = np.unique(X[:, col]) # eg. for col=0 (Outlook), feature_values will be ['Overcast', 'Rain', 'Sunny']
        likelihoods[cls][col] = {} # eg. likelihoods['Yes'][0] will store the likelihoods for feature column 0 (Outlook) for class 'Yes'

        for val in feature_values:
            # Laplace smoothing
            count = np.sum(X_cls[:, col] == val)
            likelihoods[cls][col][val] = (count + 1) / (len(X_cls) + len(feature_values)) # eg. if for class 'Yes' and feature column 0 (Outlook), count of 'Sunny' is 2, count of 'Overcast' is 3, count of 'Rain' is 5, then likelihoods['Yes'][0]['Sunny'] will be (2+1)/(10+3) = 3/13, likelihoods['Yes'][0]['Overcast'] will be (3+1)/(10+3) = 4/13, likelihoods['Yes'][0]['Rain'] will be (5+1)/(10+3) = 6/13

# Prediction
predictions = []

for sample in X_test:
    class_scores = {}

    for cls in classes:
        prob = np.log(class_priors[cls])

        for col in range(len(sample)):
            val = sample[col]
            prob += np.log(likelihoods[cls][col].get(val, 1e-9))

        class_scores[cls] = prob

    predicted_class = max(class_scores, key=class_scores.get)
    predictions.append(predicted_class)

print("\nPredictions:", predictions)
print("Actual:     ", list(y_test))

# Evaluation (binary: Yes/No)
tp = tn = fp = fn = 0

for i in range(len(y_test)):
    if y_test[i] == "Yes" and predictions[i] == "Yes":
        tp += 1
    elif y_test[i] == "Yes" and predictions[i] == "No":
        fn += 1
    elif y_test[i] == "No" and predictions[i] == "Yes":
        fp += 1
    elif y_test[i] == "No" and predictions[i] == "No":
        tn += 1

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print(f"\nAccuracy={accuracy:.2f}")
print(f"Precision={precision:.2f}")
print(f"Recall={recall:.2f}")
print(f"F1={f1:.2f}")