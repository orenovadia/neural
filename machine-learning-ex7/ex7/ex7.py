import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def compress_image_with_clusters():
    img = mpimg.imread('bird_small.png')
    w, h, pix = img.shape
    X = img.reshape((w * h, pix))

    for K in [10, ]:
        clf = KMeans(n_clusters=K).fit(X)
        idx = clf.predict(X)
        print idx.shape, clf.inertia_
        X_compressed = (clf.cluster_centers_[idx])
        compressed_img = X_compressed.reshape(img.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.imshow(compressed_img)
        plt.title('Compressed with %s clusters' % K)
        fig  = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        #ax=fig.add_subplot(projection='3d')
        ax.scatter(X[:,0],X[:,1],X[:,2],c=clf.labels_)
        plt.title('3d centeroid membership')
        pca = PCA(n_components=2)
        pca.fit(X)
        X_pca = pca.transform(X)
        plt.figure()
        plt.title('PCA 2d centeroid membership')
        plt.scatter(X_pca[:,0],X_pca[:,1] , c = clf.labels_)
        plt.show()


compress_image_with_clusters()
