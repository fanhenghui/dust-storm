import numpy as np

# scale and center features
def feature_normalize(X) :
    X_norm = np.empty(X.shape)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def compute_cost(X , y , theta):
	err = 0;
	for i in range(0 , X.shape[0]):
		h = theta[0]*X[i][0] + theta[1]
		err +=  (y[i][0] - h)**2
	return err


def gradient_descent(X, y ,theta , alpha , iter):
	
	for j in range(0,iter):
		err0 = 0
		err1 = 0
		for i in range(0,X.shape[0]):
			h = 0
			h = theta[0]*X[i][0] + theta[1]
			err0 += (h - y[i][0])*X[i][0]
			err1 += (h - y[i][0])
		print(err0)
		print(err1)
		theta[0] = theta[0] - alpha*err0
		theta[1] -= alpha*err1


#--------------------------------------------------------------------------#


file_house = "../../Data/housing/housing.data"

print("numpy version : " , np.version.version )
data = np.loadtxt(file_house)
print(data.shape)
theta = np.array((2,1))
theta[0] = 1
theta[1] = 2
#print(theta)


X = data[0:data.shape[0] , 0:1]
y = data[0:data.shape[0] , 13:14]
#print(y.shape)
#print(X[0][0])
#print(X.shape)
gradient_descent(X,y,theta,0.5,1)