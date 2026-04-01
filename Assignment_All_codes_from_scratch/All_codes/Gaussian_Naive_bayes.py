import numpy as np

n = int(input())
dataset = []

for _ in range(n):
    row = list(map(float, input().split()))
    dataset.append(row)
    
dataset = np.array(dataset)

X = dataset[:, :-2]
y = dataset[:, -2]

idx = int(0.7*len(X))
X_train, X_test = X[:idx], X[idx:]
y_train, y_test = y[:idx], y[idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}\n")

classes = np.unique(y_train) #eg. if y_train = [0, 1, 0, 1, 2], then classes = [0, 1, 2]
class_priors = {}

for cls in classes:
    class_priors[cls] = np.sum(y_train == cls) / len(y_train) #eg. if y_train = [0, 1, 0, 1, 2], then class_priors[0] = 2/5, class_priors[1] = 2/5, class_priors[2] = 1/5
    
for cls in classes:
    print(f"Class {int(cls)} Prior: {class_priors[cls]:.2f}")
    

mean_per_class = {}
std_per_class = {}

for cls in classes:
    class_data = X_train[y_train == cls] #eg. if X_train = [[1, 2], [3, 4], [5, 6]] and y_train = [0, 1, 0], then for cls = 0, class_data will be [[1, 2], [5, 6]]
    mean_per_class[cls] = np.mean(class_data, axis=0) # axis=0 means we want to calculate mean for each column, so if class_data = [[1, 2], [5, 6]], then mean_per_class[cls] will be [3, 4], axis=1 means row-wise mean
    std_per_class[cls] = np.std(class_data, axis=0)
    
    
def gaussian(x, mean, std):
    std = np.maximum(std, 1e-9)  #do not np.max because we want to keep the shape of std same as mean and x, if we do np.max then std will become a scalar and we cannot do element-wise division
    exp = np.exp(-((x - mean)**2) / (2*std**2))
    return exp / (np.sqrt(2*np.pi) * std)

# Formula for Gaussian Naive Bayes: P(xi|C) = (1/(sqrt(2*pi)*std)) * exp(-((xi - mean)**2)/(2*std**2)), we take log to avoid underflow and to convert multiplication to addition, so log(P(xi|C)) = log(1/(sqrt(2*pi)*std)) + log(exp(-((xi - mean)**2)/(2*std**2))) = -log(sqrt(2*pi)*std) - ((xi - mean)**2)/(2*std**2)   
predictions = []
for sample in X_test:
    class_scores = {}
    
    for cls in classes:
        log_prob = np.log(class_priors[cls])
        prob = gaussian(sample, mean_per_class[cls], std_per_class[cls])
        
        log_prob += np.sum(np.log(prob + 1e-9))  
        class_scores[cls] = log_prob
        
    # predicted_class = max(class_scores, key=class_scores.get) #eg. if class_scores = {0: -2.5, 1: -1.5, 2: -3.0}, then predicted_class will be 1 because it has the highest log probability
    
    predicted_class = np.argmax(list(class_scores.values())) #eg. if class_scores = {0: -2.5, 1: -1.5, 2: -3.0}, then list(class_scores.values()) will be [-2.5, -1.5, -3.0], so np.argmax(list(class_scores.values())) will return 1 because it has the highest log probability
    predictions.append(int(predicted_class))
    
print("\nPredictions:", predictions)
actual = list(map(int, y_test))
print("Actual:     ", actual)

tp = tn = fp = fn = 0
for i in range(len(y_test)):
    if y_test[i] == 1 and predictions[i] == 1:
        tp += 1
    elif y_test[i] == 1 and predictions[i] == 0:
        fn += 1
    elif y_test[i] == 0 and predictions[i] == 1:
        fp += 1
    elif y_test[i] == 0 and predictions[i] == 0:
        tn += 1
        
    
accuracy = (tp + tn)/ (tp + tn + fn + fp)
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
 
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print(f"\nAccuracy={accuracy:.2f}")
print(f"Precision={precision:.2f}")
print(f"Recall={recall:.2f}")
print(f"F1={f1_score:.2f}")
