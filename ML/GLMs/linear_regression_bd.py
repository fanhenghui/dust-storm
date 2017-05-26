import numpy as np
import pylab as plt
import sys

# Evaluate the linear regression


def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    # Number of training samples
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

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
    plt.pause(1)
    # plt.close()


def gradient_descent(X, y, theta, alpha, num_iters, is_plot, data):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

        if is_plot == True and i % 10 == 0:
            update_line(X, theta, data)
        print("iteration ", i, ": cost: ", J_history[i, 0])
    return theta, J_history


# scale and center features
def featureNormalize(X):
    X_norm = np.empty(X.shape)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

############################
# 1. Load the dataset
############################

file1 = '../../Data/housing/beijing/beiyuan1.txt'
file2 = '../../Data/housing/beijing/huilongguan1.txt'
file3 = '../../Data/housing/beijing/xierqi1.txt'

dataset = np.loadtxt(file2, delimiter=',')
train_size = int(dataset.shape[0] * 0.7)

train_data = dataset[:train_size]
test_data = dataset[train_size:]

print("## data set: ", dataset.shape)
print("## random split data to train and test sets....")
print("## train set: ", train_data.shape)
print("## test set: ", test_data.shape)


############################
# 2. Initialize parameters
############################

X_raw = train_data[:, 0]
X, mu, sigma = featureNormalize(X_raw)
y = train_data[:, 1]

# Number of training samples
m = y.size

# Add a column of ones to X (interception data)
it = np.ones(shape=(m, 2))
it[:, 1] = X

# Initialize theta parameters
theta = np.zeros(shape=(2, 1))

# Some gradient descent settings
iterations = 500
alpha = 0.01


############################
# 3. Gradient descent
############################

is_plot = True
theta, J_history = gradient_descent(
    it, y, theta, alpha, iterations, is_plot, train_data)

############################
# 4. Result
############################

a = theta[1][0]
b = theta[0][0]

x0 = X_raw[0]
y0 = a * X[0] + b

x1 = X_raw[1]
y1 = a * X[1] + b

a = (y0 - y1) / (x0 - x1)
b = (y1 - a * x1)

print('a = ', a)
print('b = ', b)

f = open('result.txt', 'w')
f.write('a = ' + str(a) + '\n')
f.write('b = ' + str(b))
f.close()
