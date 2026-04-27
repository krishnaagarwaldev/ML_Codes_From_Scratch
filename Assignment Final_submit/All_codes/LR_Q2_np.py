import numpy as np

# Step 1: Take number of data points
num_rows = int(input())

# Step 2: Read dataset
data = []
for _ in range(num_rows):
    size, bedrooms, age, price = map(float, input().split())
    data.append([size, bedrooms, age, price])

data = np.array(data)  # Convert to NumPy array for easy operations

# Step 3: EDA (first 5 rows)
for row in data[:5]:
    # print(" ".join(f"{val:.1f}" for val in row))
    print(f"{int(row[0])} {int(row[1])} {int(row[2])} {int(row[3])}")

# Step 4: Print shape
print(f"({num_rows},4)")

# Step 5: Calculate statistics (mean, std, min, max)
means = np.mean(data, axis=0) # eg. if data is [[1,2,3,4], [5,6,7,8]], then means will be [3.0, 4.0, 5.0, 6.0]
stds = np.std(data, axis=0)
mins = np.min(data, axis=0)
maxs = np.max(data, axis=0)

# Print stats column-wise
for i in range(4):
    print(f"{means[i]:.2f} {stds[i]:.2f} {mins[i]:.2f} {maxs[i]:.2f}")

# Step 6: Normalize only input features (first 3 columns)
X = (data[:, :3] - means[:3]) / stds[:3] 

# Target variable
y = data[:, 3]

# Step 7: Add bias (column of 1s)
X_bias = np.c_[np.ones(num_rows), X]  # Shape becomes (N x 4)

# 🔹 Gradient Descent
theta = np.zeros(4)        # Initialize parameters [θ0, θ1, θ2, θ3]
learning_rate = 0.01
iterations = 300

for _ in range(iterations):
    predictions = X_bias @ theta           # Matrix multiplication
    errors = predictions - y               # Error vector
    
    gradient = (X_bias.T @ errors) / num_rows  # Compute gradient
    theta = theta - learning_rate * gradient   # Update parameters

# MSE calculation
mse_gd = np.mean((X_bias @ theta - y) ** 2) / 2
# mse_gd = np.sum((X_bias @ theta - y) ** 2) / (2 * num_rows)

# 🔹 Normal Equation 
# theta_normal = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y  # Closed-form solution
theta_normal = np.linalg.pinv(X_bias) @ y
mse_normal = np.mean((X_bias @ theta_normal - y) ** 2) / 2 # mse = (1/2N) * sum((X*theta - y)^2) 

# Step 8: Print results
print(f"Final theta=[{theta[0]:.3f}, {theta[1]:.3f}, {theta[2]:.3f}, {theta[3]:.3f}]")
print(f"Final MSE={mse_gd:.2f}")
print(f"MSE Difference={abs(mse_gd - mse_normal):.5f}")

# 🔹 Predictions
test_houses = np.array([[150, 3, 5], [200, 4, 2]])

# Normalize test data using same mean/std
test_norm = (test_houses - means[:3]) / stds[:3]

# Add bias
test_norm = np.c_[np.ones(len(test_norm)), test_norm]

# Predict prices
predictions = test_norm @ theta

for val in predictions:
    print(f"{val:.2f}")



"""
20
35 1 20 179
42 2 15 200
50 2 18 221
60 3 10 263
67 3 8 280
75 3 12 314
80 4 5 327
90 4 5 360
95 4 6 377
100 5 5 391
110 5 3 425
120 5 2 462
130 6 2 493
140 6 1 521
150 6 1 552
160 7 1 582
175 7 2 631
190 8 2 675
210 8 1 740
230 9 1 804 
"""