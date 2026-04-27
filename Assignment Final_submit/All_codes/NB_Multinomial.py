import numpy as np
import math
from collections import Counter

num_samples = int(input())
dataset = []

for _ in range(num_samples):
    line = input().rsplit(maxsplit = 1) # rsplit means split from the right, so if the input is "word1 word2 word3 0", then line will be ['word1 word2 word3', '0']
    text = line[0].strip().lower().replace(',', '').split()
    labels = int(line[1])
    dataset.append((text, labels))
    
    
idx = int(0.7*num_samples)
train_data = dataset[:idx]
test_data = dataset[idx:]

print(f"Training samples: {idx}")
print(f"Test samples: {len(test_data)}")

class_0_txt = [txt for txt, label in train_data if label==0] #eg. if train_data = [(['word1', 'word2'], 0), (['word3', 'word4'], 1)], then class_0_txt will be [['word1', 'word2']]
class_1_txt = [txt for txt, label in train_data if label==1]

prior_0 = len(class_0_txt) / idx #eg. if train_data = [(['word1', 'word2'], 0), (['word3', 'word4'], 1)], then idx = 2, len(class_0_txt) = 1, so prior_0 will be 0.5
prior_1 = len(class_1_txt) / idx

print(f"Class 0 Prior: {prior_0:.2f}")  
print(f"Class 1 Prior: {prior_1:.2f}")

vocab = set(word for txt, label in train_data for word in txt) #eg. if train_data = [(['word1', 'word2'], 0), (['word3', 'word4'], 1)], then vocab will be {'word1', 'word2', 'word3', 'word4'}
vocab_size = len(vocab)


def word_count(text):
    all_words = [word for txt in text for word in txt]
    word_freq = Counter(all_words) #eg. if all_words = ['word1', 'word2', 'word1'], then word_freq = {'word1': 2, 'word2': 1}
    total_words = sum(word_freq.values()) #eg. if word_freq = {'word1': 2, 'word2': 1}, then total_words will be 3
    return word_freq, total_words 

word_count_0, total_words_0 = word_count(class_0_txt)
word_count_1, total_words_1 = word_count(class_1_txt)

def predict(text):
    words = np.array(text)
    
    # word_count_0.get(w,0) eg. if word_count_0 = {'word1': 2, 'word2': 1} and w = 'word1', then word_count_0.get(w,0) will return 2, but if w = 'word3' then word_count_0.get(w,0) will return 0
    probs_0 = np.array([ (word_count_0.get(w,0) + 1) / (total_words_0 + vocab_size) for w in words])
    probs_1 = np.array([ (word_count_1.get(w,0) + 1) / (total_words_1 + vocab_size) for w in words])
    
    # log_prob_0 = math.log(prior_0) + np.sum(np.log(probs_0)) eg. if prior_0 = 0.5 and probs_0 = [0.2, 0.3], then log_prob_0 will be log(0.5) + log(0.2) + log(0.3)
    # Actual formula without log is P(C) * P(x1|C) * P(x2|C) * ... * P(xn|C), but we take log to avoid underflow and to convert multiplication to addition, so log(P(C)) + log(P(x1|C)) + log(P(x2|C)) + ... + log(P(xn|C))
    log_prob_0 = math.log(prior_0) + np.sum(np.log(probs_0))
    log_prob_1 = math.log(prior_1) + np.sum(np.log(probs_1))  #math.log is used to calculate log of a number, np.log is used to calculate log of each element in an array, so if probs_1 = [0.2, 0.3], then np.log(probs_1) will return [-1.60943791, -1.2039728]
    
    return 0 if log_prob_0 > log_prob_1 else 1
    
    
predictions = [predict(txt) for txt, label in test_data]
actual_labels = [label for txt, label in test_data]

pred_arr = np.array(predictions)
actual_arr = np.array(actual_labels)

accuracy = np.mean(pred_arr == actual_arr) #eg. if pred_arr = [0, 1, 0, 1] and actual_arr = [0, 0, 0, 1], then pred_arr == actual_arr will be [True, False, True, True], so np.mean(pred_arr == actual_arr) will be (1 + 0 + 1 + 1)/4 = 0.75

# precision = np.sum((pred_arr==1) & (actual_arr==1)) / np.sum(pred_arr==1) if np.sum(pred_arr==1) else 0
# recall = np.sum((pred_arr==1) & (actual_arr==1)) / np.sum(actual_arr==1) if np.sum(actual_arr==1) else 0
# f1_score = 2*precision*recall / (precision + recall) if (precision + recall) else 0

tp = np.sum((pred_arr==1) & (actual_arr==1))
tn = np.sum((pred_arr==0) & (actual_arr==0))
fp = np.sum((pred_arr==1) & (actual_arr==0))
fn = np.sum((pred_arr==0) & (actual_arr==1))

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0


print(f"Predictions: {predictions}")
print(f"Actual: {actual_labels}")
print(f"Accuracy={accuracy:.2f}")
print(f"Precision={precision:.2f}")
print(f"Recall={recall:.2f}")
print(f"F1={f1_score:.2f}")
