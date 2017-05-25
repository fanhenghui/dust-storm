import numpy as np
import sys

# scale and center features
def feature_normalize(X) :
    X_norm = np.empty(X.shape)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    max_val = np.max(X)
    min_val = np.min(X)

    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma , max_val , min_val

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


def gradient_descent(X, y ,theta , alpha , iter):
	#J_history = np.zeros(shape=(iter, 1))
	m_r = 1.0/y.size;
	for j in range(0,iter):
		h =( X.dot(theta).flatten() - y)
		err0 = h*X[:,0]
		err1 = h*X[:,1]
		#print(err0.sum()*m_r*alpha);
		#print(err1.sum()*m_r*alpha);
		theta[0][0] =theta[0][0]- err0.sum()*m_r*alpha;
		theta[1][0] =theta[1][0] - err1.sum()*m_r*alpha;

		_history[i, 0] = compute_cost(X, y, theta)
	return theta;


#--------------------------------------------------------------------------#


file_house = "../../Data/housing/beijing/huilongguan1.txt"
train_data_rate = 0.7
iterations = 500
alpha = 0.01


print("numpy version : " , np.version.version )
dataset = np.loadtxt(file_house , delimiter=',')
print("dataset : " , dataset.shape)

train_size = int(dataset.shape[0] * 0.7)
train_data = dataset[0:train_size , : ]
m = train_data.shape[0]
y = train_data[: ,1]
X_norm , mu , sigma , max_val , min_val = feature_normalize(train_data[:, 0])
X = np.ones(shape = (m , 2))
X[:, 1] = X_norm
print("X0 : " , X[0,0])
print("X1 : " , X[0,1])

#print(X[0])
#print(y[0])
#sys.exit()

#print(y.shape)
#print(X[0][0])
#print(X.shape)
#gradient_descent(X,y,theta,0.5,1)


theta = np.zeros(shape = (2,1))


#h = X.dot(theta).flatten()
#print("X sum is : " , X.sum())
#err = (h - y)*X[:, 0]
#print("Sum is : " , err.sum())

theta = gradient_descent(X , y , theta , alpha , iterations)
print(theta[0][0])
print(theta[1][0])

