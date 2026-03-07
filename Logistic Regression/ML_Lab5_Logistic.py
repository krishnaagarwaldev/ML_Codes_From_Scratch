import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# %matplotlib inline


url = "https://raw.githubusercontent.com/susilvaalmeida/machine-learning-andrew-ng/refs/heads/master/data/ex2data1.txt"
df = pd.read_csv(url, header=None)
df.columns = ["exam_score1", "exam_score2", "labels"]

print(df.shape)
print(df.head())
print(df.describe())

x = df[["exam_score1", "exam_score2"]].values
y = df["labels"].values.reshape(-1, 1)

x_shape = x.shape[0]
x_bias = np.hstack((np.ones((x_shape, 1)), x))
print(x.shape, y.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(theta, x_bias, y_actual):
    m = len(y_actual)
    y_hat = sigmoid(x_bias @ theta)
    loss = -(1/m) * np.sum(y_actual * np.log(y_hat + 1e-5) + (1 - y_actual) * np.log(1 - y_hat + 1e-5))
    return loss

def gradient(x_bias, y_actual, theta, alpha, epochs):
    m = len(y_actual)
    for _ in range(epochs):
        y_hat = sigmoid(x_bias @ theta)
        slope = (1/m) * (x_bias.T @ (y_hat - y_actual))
        theta = theta - alpha * slope
    return theta


initial_theta = np.zeros((x_bias.shape[1], 1))

alpha = 0.01
epoch = 10000

final_theta = gradient(x_bias, y, initial_theta, alpha, epoch)

def predictions(x_bias, final_theta):
    prob = sigmoid(x_bias @ final_theta)
    return (prob >= 0.5).astype(int)

predicted = predictions(x_bias, final_theta)
train_score = np.mean(predicted == y) * 100

print("Learned Parameters:\n", final_theta)
print("Loss:", compute_loss(final_theta, x_bias, y))
print("Training Score:", train_score)

stu = np.array([1, 45, 85])
prob = sigmoid(stu @ final_theta)

print("Admission Chance for (45,85):", prob[0])



# # Step 1
# import numpy as np
# import pandas as pd
# import random
# import matplotlib.pyplot as plt
# # %matplotlib inline

# # x = np.random.randn(100,2) + np.array([0,100])
# # print(x)

# df = pd.read_csv("https://raw.githubusercontent.com/susilvaalmeida/machine-learning-andrew-ng/refs/heads/master/data/ex2data1.txt", header=None)
# #df = pd.DataFrame(data)
# df.columns = ['exam_score_1', 'exam_score_2', 'label']
# print(df.shape)
# print(df.head())
# print(df.describe())

# m = df.shape[0]
# x = np.hstack((np.ones((m,1)), df[['exam_score_1', 'exam_score_2']].values))
# y = np.array(df.label.values).reshape(-1,1)
# print(x.shape, y.shape)

# #Splitting data
# x_train = x[:80,:]
# x_test = x[80:,:]
# print(x_train.shape, x_test.shape)

# y_train = y[:80,:]
# y_test = y[80:,:]
# print(y_train.shape, y_test.shape)


# #Initializing model parameter
# # w =np.zeros(shape = (x.shape[1]))
# w = np.zeros((3,1))
# print(w.shape)
# alpha = 0.01
# epoch = 2000
# loss_hist = []

# def sigmoid(z):
#     return 1/(1+np.exp(-z))

# #Cost function
# def binary_cross_entropy(y_train, y_hat):
#     n = y_train.shape[0]  #80
#     loss = - (y_train*np.log1p(y_hat) + (1-y_train)*np.log1p(1-y_hat))
#     loss = loss.sum()/n #loss.mean()
#     return loss
    
# #Gradient descent
# for i in range(epoch):
#     y_hat = sigmoid (x_train @ w) #(80,3)*(3,1) = (80,1)
#     loss = binary_cross_entropy(y_train, y_hat)
#     loss_hist.append(loss)
    
#     train_size = x_train.shape[0]
#     gradient = (1/train_size)*(x_train.T @ (y_hat - y_train))   # (3,80)*(80,1) = (3,1)
#     w = w - alpha * gradient

# #print(loss_hist)

# #Plotting data
# plt.plot(loss_hist)
# plt.show()

# # Evalution metrics
# y_pred = x_test @ w  # (20,3)*(3,1) = (20,1)
# y_pred = sigmoid(y_pred)
# #print(y_pred)
# y_pred = y_pred > 0.5

# correct = (y_pred == y_test).sum()
# test_size = y_test.shape[0]
# accuracy = correct*100 / test_size
# print("Accuracy :- ", accuracy)


# def predictions(x, w):
#     prob = sigmoid(x @ w)
#     return (prob >= 0.5).astype(int)

# predicted = predictions(x, w)
# train_score = np.mean(predicted == y) * 100

# print("Learned Parameters:\n", w)
# y_hat = sigmoid(x @ w)
# print("Loss:", binary_cross_entropy(x, y_hat))
# print("Training Score:", train_score)

# stu = np.array([1, 45, 85])
# prob = sigmoid(stu @ w)

# print("Admission Chance for (45,85):", prob[0])