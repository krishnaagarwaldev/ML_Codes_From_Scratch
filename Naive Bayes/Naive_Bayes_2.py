import numpy as np

# ----------------------------
# Training Data
# ----------------------------

income = np.array([30, 40, 50, 60, 70, 80])
age = np.array([25, 35, 45, 20, 30, 50])
credit = np.array(["Bad", "Bad", "Good", "Good", "Good", "Bad"])

# Target: 0 = Not Approved, 1 = Approved
y = np.array([0, 0, 1, 1, 1, 1])

# ----------------------------
# Test Data
# ----------------------------

test_income = 55
test_age = 28
test_credit = "Good"

# ----------------------------
# Separate Classes
# ----------------------------

income_0 = income[y == 0]
income_1 = income[y == 1]

age_0 = age[y == 0]
age_1 = age[y == 1]

credit_0 = credit[y == 0]
credit_1 = credit[y == 1]

# ----------------------------
# Prior Probabilities
# ----------------------------

prior_0 = len(income_0) / len(income)
prior_1 = len(income_1) / len(income)

# =====================================================
# -------- NUMERICAL (Gaussian Naive Bayes) ----------
# =====================================================

# Mean and Variance for Income
mean_income_0 = np.mean(income_0)
var_income_0 = np.var(income_0)

mean_income_1 = np.mean(income_1)
var_income_1 = np.var(income_1)

# Gaussian formula for Income
prob_income_0 = (1 / np.sqrt(2 * np.pi * var_income_0)) * \
                np.exp(-((test_income - mean_income_0)**2) / (2 * var_income_0))

prob_income_1 = (1 / np.sqrt(2 * np.pi * var_income_1)) * \
                np.exp(-((test_income - mean_income_1)**2) / (2 * var_income_1))


# Mean and Variance for Age
mean_age_0 = np.mean(age_0)
var_age_0 = np.var(age_0)

mean_age_1 = np.mean(age_1)
var_age_1 = np.var(age_1)

# Gaussian formula for Age
prob_age_0 = (1 / np.sqrt(2 * np.pi * var_age_0)) * \
             np.exp(-((test_age - mean_age_0)**2) / (2 * var_age_0))

prob_age_1 = (1 / np.sqrt(2 * np.pi * var_age_1)) * \
             np.exp(-((test_age - mean_age_1)**2) / (2 * var_age_1))


# =====================================================
# -------- CATEGORICAL (Frequency Probability) -------
# =====================================================

prob_credit_0 = np.sum(credit_0 == test_credit) / len(credit_0)
prob_credit_1 = np.sum(credit_1 == test_credit) / len(credit_1)

# =====================================================
# -------- FINAL NAIVE BAYES FORMULA ------------------
# =====================================================

final_0 = prior_0 * prob_income_0 * prob_age_0 * prob_credit_0
final_1 = prior_1 * prob_income_1 * prob_age_1 * prob_credit_1

# Prediction
if final_0 > final_1:
    print("Predicted Class: 0 (Not Approved)")
else:
    print("Predicted Class: 1 (Approved)")