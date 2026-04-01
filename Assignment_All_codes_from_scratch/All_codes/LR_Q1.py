import math
import numpy as np 

# Take input
N = int(input())

features = []
targets = []

for _ in range(N):
    x, y = map(float, input().split())
    features.append(x)
    targets.append(y)

# Convert to numpy arrays
features = np.array(features)
targets = np.array(targets)

# EDA - first 5 rows 
# print(f"{features[:5]} {targets[:5]}") # This will print the first 5 values of features and targets as arrays, but we want to print them in the specified format.
for i in range(min(5, N)):
    print(f"{features[i]:.1f} {targets[i]:.1f}")

# Shape
print(f"({N},2)")

# Function to calculate stats using numpy
def stats(col):
    mean = np.mean(col)
    std = np.std(col)
    return mean, std, np.min(col), np.max(col)

# Feature stats
fm, fs, fmin, fmax = stats(features)

# Target stats
tm, ts, tmin, tmax = stats(targets)

print(f"{fm:.2f} {fs:.2f} {fmin:.2f} {fmax:.2f}")
print(f"{tm:.2f} {ts:.2f} {tmin:.2f} {tmax:.2f}")

# Normalize features
x_norm = (features - fm) / fs

# Gradient Descent
theta0 = 0.0
theta1 = 0.0
alpha = 0.01
epochs = 1000

for _ in range(epochs):
    predictions = theta0 + theta1 * x_norm
    
    errors = predictions - targets
    
    theta0 -= alpha * (np.sum(errors) / N)
    theta1 -= alpha * (np.dot(errors, x_norm) / N)

# Final predictions
predictions = theta0 + theta1 * x_norm
errors = predictions - targets

mse = np.sum(errors ** 2) / (2 * N)

print(f"Final theta0={theta0:.3f} | theta1={theta1:.3f} | Final MSE={mse:.2f}")

# Predict for new values
for val in [150, 200]:
    x_n = (val - fm) / fs
    result = theta0 + theta1 * x_n
    print(f"{result:.2f}")


"""
20
35 179
42 200
50 221
60 263
67 280
75 314
80 327
90 360
95 377
100 391
110 425
120 462
130 493
140 521
150 552
160 582
175 631
190 675
210 740
230 804 
"""