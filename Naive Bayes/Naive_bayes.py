import numpy as np

# Training data
salary = np.array([50, 60, 65, 70, 80])
education = np.array(["UG", "UG", "PG", "PG", "PhD"])
y = np.array([0, 0, 1, 1, 1])

# Test data
test_salary = 68
test_education = "PG"

# Separate classes
salary_0 = salary[y == 0]
salary_1 = salary[y == 1]

edu_0 = education[y == 0]
edu_1 = education[y == 1]

# Prior probabilities
prior_0 = len(salary_0) / len(salary)
prior_1 = len(salary_1) / len(salary)

# -------- Numerical (Gaussian) --------

# Mean and variance
mean_0 = np.mean(salary_0)
var_0 = np.var(salary_0)

mean_1 = np.mean(salary_1)
var_1 = np.var(salary_1)

# Gaussian probability
prob_salary_0 = (1 / np.sqrt(2 * np.pi * var_0)) * \
                np.exp(-((test_salary - mean_0)**2) / (2 * var_0))

prob_salary_1 = (1 / np.sqrt(2 * np.pi * var_1)) * \
                np.exp(-((test_salary - mean_1)**2) / (2 * var_1))

# -------- Categorical --------

# Frequency probability
prob_edu_0 = np.sum(edu_0 == test_education) / len(edu_0)
prob_edu_1 = np.sum(edu_1 == test_education) / len(edu_1)

# -------- Final Probability --------

final_0 = prior_0 * prob_salary_0 * prob_edu_0
final_1 = prior_1 * prob_salary_1 * prob_edu_1

# Prediction
if final_0 > final_1:
    print("Predicted Class: 0 (No Buy)")
else:
    print("Predicted Class: 1 (Buy)")