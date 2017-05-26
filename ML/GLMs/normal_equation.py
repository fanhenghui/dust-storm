import numpy as np
import pylab as plt
import sys

# scale and center features


def feature_normalize(X):
    X_norm = np.empty(X.shape)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    max_val = np.max(X)
    min_val = np.min(X)

    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma, max_val, min_val


def compute_cost(X, y, theta):
    # Comput cost for linear regression
    m = y.size
    h = X.dot(theta).flatten()
    err = ((h - y)**2).sum()
    J = 0.5 * (1.0 / m) * err
    return J


def update_line(X, theta, data):
    # Plot the data
    plt.cla()
    plt.scatter(data[:, 0], data[:, 1], marker='o', c='b')
    plt.title('House Price distribution')
    plt.xlabel('House rea')
    plt.ylabel('House Price')
    result = X.dot(theta).flatten()
    plt.plot(data[:, 0], result)
    plt.pause(100)


#file_house = "../../Data/housing/housing.data"
file_house = "../../Data/housing/beijing/huilongguan1.txt"
train_data_rate = 0.7
iterations = 500
alpha = 0.01

feature_row = 0
target_row = 1
deli = ','


dataset = np.loadtxt(file_house, delimiter=deli)
#dataset = np.loadtxt(file_house)
print("dataset : ", dataset.shape)

train_size = int(dataset.shape[0] * 0.7)
test_size = dataset.shape[0] - train_size
train_data = dataset[0:train_size, :]
test_data = dataset[train_size:dataset.shape[0], :]
print("training set : ", train_data.shape)
print("testing set : ", test_data.shape)


m = train_data.shape[0]
y = train_data[:, target_row]
X = np.ones(shape=(m, 2))
X[:, 1] = train_data[:, feature_row]

print(X[0][1])
print(X[3][1])
XT = X.T
print(XT.shape)
XXT = np.matmul(X.T, X)
print(X)
print(XT)
print(XXT)
XXTInv = np.linalg.inv(XXT)
print(XXTInv)

theta = np.matmul(XXTInv, np.matmul(XT, y))
print(theta)
# sys.exit()

X = np.ones(shape=(train_data.shape[0], 2))
X[:, 1] = train_data[:, feature_row]
y = train_data[:, target_row]
J = compute_cost(X, y, theta)
print("traing J ", J)

#update_line(X, theta, train_data)


X2 = np.ones(shape=(test_data.shape[0], 2))
X2[:, 1] = test_data[:, feature_row]
y2 = test_data[:, target_row]
J2 = compute_cost(X2, y2, theta)
print("testing J ", J2)

update_line(X2, theta, test_data)
