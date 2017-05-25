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

def compute_cost(X , y , theta):
	err = 0;
	for i in range(0 , X.shape[0]):
		h = theta[0]*X[i][0] + theta[1]
		err +=  (y[i][0] - h)**2
	return err


def gradient_descent(X, y ,theta , alpha , iter):
	for j in range(0,iter):
		h =( X.dot(theta) - y)
		err0 = h*X[:,0]
		err1 = h*X[:,1]

		print(err0.sum())
		print(err1.sum())
		sys.exit()

		theta[0] = theta[0] - alpha*err0
		theta[1] -= alpha*err1


#--------------------------------------------------------------------------#


file_house = "../../Data/housing/beijing/huilongguan1.txt"

print("numpy version : " , np.version.version )
train_data = np.loadtxt(file_house , delimiter=',')
print(train_data.shape)

m = train_data.shape[0]
y = train_data[: ,1]
X_norm , mu , sigma , max_val , min_val = feature_normalize(train_data[0:m , 0])
X = np.ones(shape = (m , 2))
X[0:m , 0] = X_norm 

#print(X[0])
#print(y[0])
#sys.exit()

#print(y.shape)
#print(X[0][0])
#print(X.shape)
#gradient_descent(X,y,theta,0.5,1)


print("mean : ", mu)
print("Std : " , sigma)
print("Max : " , max_val)
print("Min : " , min_val)
print("Norm min : " , np.min(X_norm))
print("Norm max : " , np.max(X_norm))

iterations = 500
alpha = 0.01

theta = np.ones(shape = (2,1))

h = X.dot(theta).flatten()
print("h shape : " , y.shape)
err = (h - y)*X[:, 0]
print(err.sum())

