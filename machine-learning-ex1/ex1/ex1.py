import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, theta):
    return np.sum(
        (theta.T.dot(X) - y) ** 2
    )


def compute_gradient(X, y, theta, m):
    predictions = theta.T.dot(X)
    delta = predictions - y
    grad = (delta * X).sum(1)
    return grad / m


def gradient_decent(X, y, theta, alpha, iterations):
    J_history = np.ndarray([iterations, 1])
    m = len(y)
    for i in xrange(iterations):
        J_history[i] = compute_cost(X, y, theta)
        theta = theta + alpha * compute_gradient(X, y, theta, m)
        # print np.sum( np.sum(theta.dot(X) - y) * X ,1).shape,theta.shape
        # print (np.sum(theta.dot(X) - y) * X ).sum(1).shape,y.shape
        # print np.sum( (theta.dot(X) - y).dot(X.T) ,0).T.shape
    return theta, J_history


data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]

y = data[:, 1]
X = np.vstack((np.ones_like(X), X))
theta = np.zeros([2, ])
m = len(y)

real_x = data[:, 0]

theta, J_history = gradient_decent(X, y, theta, -0.001, 1500)
plt.figure(1)
plt.scatter(real_x ,y)
x_h = np.linspace(real_x.min() ,real_x.max())

plt.plot( x_h , theta[0] + theta[1]*x_h )
plt.figure(2)
plt.semilogx(J_history,)
print theta
r = 100
theta0_vals = np.linspace(-10, 10, r)
theta1_vals = np.linspace(-1, 4, r)
plt.figure(3)
J = np.ndarray((r,r))
for i in range( r ):
    for j in range(r):
        J[i,j] = compute_cost(X,y,np.hstack([theta0_vals[i],theta1_vals[j]]))
plt.contour(theta0_vals,theta1_vals , J,levels=np.logspace(-5,5,20))
plt.show()
