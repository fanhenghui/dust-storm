import numpy as np
import pylab as plt
import sys
# scale and center features


def feature_normalize(X):
    feature_num = X.shape[1]
    mu = np.zeros(shape=(1, feature_num))
    sigma = np.zeros(shape=(1, feature_num))
    X_norm = np.empty(X.shape)

    # sys.exit()
    for i in range(0, feature_num):
        mu[0, i] = np.mean(X[:, i], axis=0)
        sigma[0, i] = np.std(X[:, i], axis=0)
        X_norm[:, i] = (X[:, i] - mu[0, i]) / sigma[0, i]

    return X_norm, mu, sigma


def compute_cost(X, y, theta):
    # Comput cost for linear regression
    m = y.size
    h = X.dot(theta).flatten()
    err = ((h - y)**2).sum()
    J = 0.5 * (1.0 / m) * err
    return J


def gradient_descent(X, y, theta, alpha, iter):
    J_history = np.zeros(shape=(iter, 1))
    m_r = 1.0 / y.size
    for j in range(0, iter):
        h = (X.dot(theta).flatten() - y)  # calculate all samples
        for k in range(0, X.shape[1]):
            err = (h * X[:, k]).sum()
            theta[k][0] -= err * m_r * alpha
        J_history[j, 0] = compute_cost(X, y, theta)
    # print("J : " , J_history[iter-1,0])
    J = J_history[iter - 1, 0]
    return theta, J


def update_2d_line(X, y, theta):
    plt.scatter(X[:1], y)


file_house = "../../Data/housing/housing.data"
#file_house = "../../Data/housing/beijing/huilongguan1.txt"
train_data_rate = 0.7
iterations = 500
alpha = 0.01

#feature_rows = np.array([0, 1, 2, 3, 4, 5])
feature_rows = np.array([0, 4, 5])
#feature_rows = np.array([0])
print(feature_rows)
target_row = 13

deli = ','


print("numpy version : ", np.version.version)
#dataset = np.loadtxt(file_house, delimiter=deli)
dataset = np.loadtxt(file_house)
samples = dataset.shape[0]
print("dataset : ", dataset.shape)


train_size = int(samples * 0.7)
test_size = samples - train_size
train_data = np.empty(shape=(train_size, feature_rows.size))
for i in range(0, feature_rows.size):
    train_data[0:train_size, i] = dataset[0:train_size, feature_rows[i]]
print("training set : ", train_data.shape)


test_data = np.empty(shape=(test_size, feature_rows.size))
for i in range(0, feature_rows.size):
    test_data[0:test_size, i] = dataset[
        train_size:samples, feature_rows[i]]

print("testing set : ", test_data.shape)


m = train_data.shape[0]
y = dataset[0:train_size, target_row]
X_norm, mu, sigma = feature_normalize(train_data)
print("mean : ", mu)
print("variance : ", sigma)
print(X_norm[0, 0])
# sys.exit()

X = np.ones(shape=(m, 1 + feature_rows.size))
X[:, 1:feature_rows.size + 1] = X_norm
theta = np.zeros(shape=(1 + feature_rows.size, 1))


theta, J = gradient_descent(X, y, theta, alpha, iterations)
print("traing J ", J)
# print(theta[0][0])
# print(theta[1][0])
# print(theta)

print(theta)

for i in range(1, feature_rows.size + 1):
    theta[0][0] += (-theta[i][0] * mu[0][i - 1] / sigma[0][i - 1])
for i in range(1, feature_rows.size + 1):
    theta[i][0] = theta[i][0] / sigma[0][i - 1]

print(theta)

X2 = np.ones(shape=(test_data.shape[0], feature_rows.size + 1))
X2[:, 1:feature_rows.size + 1] = test_data
y2 = dataset[train_size:samples, target_row]
J2 = compute_cost(X2, y2, theta)
print("testing J ", J2)
