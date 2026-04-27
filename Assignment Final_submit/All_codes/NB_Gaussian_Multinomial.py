"""
6
25 50000 | good product | 1
30 60000 | bad quality | 0
22 45000 | excellent item | 1
35 80000 | worst product | 0
28 52000 | nice and good | 1
40 90000 | very bad | 0
"""
import numpy as np
import math
from collections import Counter

# =========================
# INPUT SECTION
# =========================
# Format per row:
# num1 num2 ... numN | text sentence | label
# Example:
# 25 50000 | good product nice | 1

n = int(input())
dataset = []

for _ in range(n):
    line = input().split('|')
    
    # Numerical features
    num_features = list(map(float, line[0].strip().split()))
    
    # Text features
    text = line[1].strip().lower().replace(',', '').split()
    
    # Label
    label = int(line[2].strip())
    
    dataset.append((num_features, text, label))


# =========================
# TRAIN-TEST SPLIT
# =========================
idx = int(0.7 * n)
train_data = dataset[:idx]
test_data = dataset[idx:]

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}\n")


# =========================
# SEPARATE FEATURES
# =========================
X_train = np.array([d[0] for d in train_data])
y_train = np.array([d[2] for d in train_data])

texts_train = [d[1] for d in train_data]


# =========================
# PRIOR PROBABILITIES
# =========================
classes = np.unique(y_train)
class_priors = {}

for cls in classes:
    class_priors[cls] = np.sum(y_train == cls) / len(y_train)
    print(f"Class {cls} Prior: {class_priors[cls]:.2f}")


# =========================
# GAUSSIAN (NUMERICAL PART)
# =========================
mean_per_class = {}
std_per_class = {}

for cls in classes:
    class_data = X_train[y_train == cls]
    mean_per_class[cls] = np.mean(class_data, axis=0)
    std_per_class[cls] = np.std(class_data, axis=0)


def gaussian(x, mean, std):
    std = np.maximum(std, 1e-9)  # avoid division by zero
    exp = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return exp / (np.sqrt(2 * np.pi) * std)


# =========================
# MULTINOMIAL (TEXT PART)
# =========================
class_texts = {cls: [] for cls in classes}

for num, txt, label in train_data:
    class_texts[label].append(txt)

# Build vocabulary
vocab = set(word for txts in class_texts.values() for txt in txts for word in txt)
vocab_size = len(vocab)


def word_count(texts):
    all_words = [word for txt in texts for word in txt]
    freq = Counter(all_words)
    total = sum(freq.values())
    return freq, total


word_stats = {}
for cls in classes:
    word_stats[cls] = word_count(class_texts[cls])


# =========================
# PREDICTION FUNCTION
# =========================
def predict(num_features, text_words):
    
    class_scores = {}
    
    for cls in classes:
        
        # 1. Prior
        log_prob = math.log(class_priors[cls])
        
        # 2. Numerical (Gaussian)
        prob_num = gaussian(num_features, mean_per_class[cls], std_per_class[cls])
        log_prob += np.sum(np.log(prob_num + 1e-9))
        
        # 3. Text (Multinomial with Laplace smoothing)
        word_freq, total_words = word_stats[cls]
        
        probs_text = [
            (word_freq.get(w, 0) + 1) / (total_words + vocab_size)
            for w in text_words
        ]
        
        log_prob += np.sum(np.log(probs_text))
        
        class_scores[cls] = log_prob
    
    return max(class_scores, key=class_scores.get)


# =========================
# TESTING
# =========================
predictions = []
actual = []

for num, txt, label in test_data:
    pred = predict(np.array(num), txt)
    predictions.append(pred)
    actual.append(label)

print("\nPredictions:", predictions)
print("Actual:     ", actual)


# =========================
# METRICS (Binary)
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