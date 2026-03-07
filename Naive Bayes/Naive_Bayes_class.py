import numpy as np
import pandas as pd

class NaiveBayes:

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = {}
        self.mean = {}
        self.var = {}
        self.cat_prob = {}

        for c in self.classes:
            X_c = X[y == c]

            # Prior
            self.prior[c] = len(X_c) / len(X)

            # Numerical column (Temperature → column 2)
            nums = X_c[:, 2].astype(float)
            self.mean[c] = np.mean(nums)
            self.var[c] = np.var(nums) + 1e-6

            # Categorical columns (0 and 1)
            self.cat_prob[c] = {}
            for col in [0, 1]:
                self.cat_prob[c][col] = {}
                values = np.unique(X[:, col])
                for val in values:
                    count = np.sum(X_c[:, col] == val)
                    self.cat_prob[c][col][val] = (count + 1) / (len(X_c) + len(values))

    def gaussian(self, x, mean, var):
        return np.exp(-(x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)

    def predict(self, X):
        result = []

        for x in X:
            probs = []

            for c in self.classes:
                total = self.prior[c]

                # categorical
                for col in [0, 1]:
                    total *= self.cat_prob[c][col][x[col]]

                # numerical
                total *= self.gaussian(float(x[2]), self.mean[c], self.var[c])

                probs.append(total)

            result.append(self.classes[np.argmax(probs)])

        return np.array(result)


# ---------------- Dataset ----------------

data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny',
                'Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High',
                 'Normal','Normal','Normal','High','Normal','High'],
    'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
    'Play': ['No','No','Yes','Yes','Yes','No','Yes','No',
             'Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

X = df[['Outlook','Humidity','Temperature']].values
y = df['Play'].values

# Train
model = NaiveBayes()
model.fit(X, y)

# Predict
print(model.predict(np.array([['Rain','High',72]])))
print(model.predict(np.array([['Sunny','High',85]])))

# Accuracy
pred = model.predict(X)
print("Accuracy:", np.mean(pred == y))

# Confusion Matrix -> Manually compute confusion matrix for binary classification
tp = np.sum((pred == 'Yes') & (y == 'Yes'))
tn = np.sum((pred == 'No') & (y == 'No'))
fp = np.sum((pred == 'Yes') & (y == 'No'))
fn = np.sum((pred == 'No') & (y == 'Yes'))
print("Confusion Matrix:")
print("TP:", tp, "FP:", fp)
print("FN:", fn, "TN:", tn)

# Precision and Recall
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("Precision:", precision)
print("Recall:", recall)


