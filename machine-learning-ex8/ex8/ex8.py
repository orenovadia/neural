import scipy.io
import matplotlib.pyplot as plt
mfile = 'ex8_movies.mat'
data = scipy.io.loadmat(mfile)
print data.keys()
Y = data['Y']
plt.subplot(1,2,1)
plt.imshow(Y)
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(data['R'])
plt.colorbar()
plt.show()

