import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from matplotlib.mlab import PCA as mlabPCA

data = load_iris()
X = data.data
y = data.target
l1, l2 = [], []
for i in range(1, 10):
    break
    clf = KMeans(i).fit(data.data)
    print clf.inertia_
    l1.append(i)
    l2.append(clf.inertia_)


# plt.plot(np.asarray(l1),np.asarray(l2) )
def draw_clusters(X, ind1, ind2, clf):
    m, dim = X.shape
    l = []
    h = .2
    for i in range(dim):
        x_min, x_max = X[:, i].min() - 1, X[:, i].max() + 1
        l.append(
            np.arange(x_min, x_max, h)
        )

    xx, yy, zz, ww = np.meshgrid(
        *l
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel(), ww.ravel()])

    # Put the result into a color plot
    print Z.shape
    Z = Z.reshape(xx.shape)
    print Z.shape
    print xx.shape
    return xx, yy, zz, ww, Z


clf = KMeans(3).fit(data.data)
xx, yy, zz, ww, Z = draw_clusters(X, 0, 1, clf)

plt.subplot(3, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=y)

plt.subplot(3, 1, 2)
plt.scatter(X[:, 2], X[:, 3], c=y)

plt.subplot(3, 1, 3)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02  # point in the mesh [x_min, m_max]x[y_min, y_max].
mlab_pca = mlabPCA(X)
cutoff = mlab_pca.fracs[1]
users_2d = mlab_pca.project(X, minfrac=cutoff)
thang = mlab_pca.project(np.c_[xx.ravel(), yy.ravel(), zz.ravel(), ww.ravel()] , minfrac=cutoff)
Z = Z.ravel()

print Z.shape , users_2d.shape
xx , yy = users_2d[:,0] , users_2d[:,1]
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

#centroids_2d = mlab_pca.project(centroid_list, minfrac=cutoff)


plt.show()
