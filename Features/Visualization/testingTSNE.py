from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import visualization_utils as vu

X, y = vu.getDataNorm()

tsne = TSNE(n_components=2)
X_TSNE2 = tsne.fit_transform(X)
plt.scatter(X_TSNE2[:, 0], X_TSNE2[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.savefig('Figures/T-SNE2d.png')
plt.show()


tsne = TSNE(n_components=3)
X_TSNE3 = tsne.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_TSNE3[:, 0], X_TSNE3[:, 1], X_TSNE3[:, 2], c=y, cmap="jet")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('Figures/T-SNE3d.png')
plt.show()
