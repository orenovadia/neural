import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC ,SVC
a = scipy.io.loadmat('ex6data1.mat')
X , y = a['X'] , a['y']
C=1.0
h=.02
clf = LinearSVC(C=C)
clf = SVC(C=C,kernel='rbf',degree = 10)
clf.fit(X,y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=y,marker='o',s=30)

plt.show()