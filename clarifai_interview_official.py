import numpy as np
import random
import math
from scipy.misc import imread, imresize


def euclidean_distance(a, b):
    if isinstance(a, list) and isinstance(b, list):
        a = np.array(a)
        b = np.array(b)

    return math.sqrt(sum((a - b) ** 2))

class KMeans(object):

    def __init__(self, K=5, max_iters=100, init='random', thres=0.01):
        self.K = K
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.init = init
        self.thres = thres

    def predict(self, X=None):
        """Perform clustering on the dataset."""

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.X = X

        self._initialize_centroids(self.init)
        centroids = self.centroids

        # Optimize clusters
        for i in range(self.max_iters):
            self._assign(centroids)
            centroids_old = centroids
            centroids = [self._get_centroid(cluster) for cluster in self.clusters]

            distance = 0
            for j in range(self.K):
                distance += euclidean_distance(centroids_old[j], centroids[j])
            print('distance is {:.4f}'.format(distance))

            if distance < self.thres:
                break
        # centroids are updated
        self.centroids = centroids

        # now do the clustering
        predictions = np.empty(self.n_samples)

        for i, cluster in enumerate(self.clusters):
            for index in cluster:
                predictions[index] = i
        return predictions

    def _initialize_centroids(self, init):
        """Set the initial centroids."""

        if init == 'random':
            self.centroids = [self.X[x] for x in
                              random.sample(range(self.n_samples), self.K)]
        else:
            raise ValueError('Unknown type of init parameter')

    def _assign(self, centroids):

        for row in range(self.n_samples):
            for i, cluster in enumerate(self.clusters):
                if row in cluster:
                    self.clusters[i].remove(row)
                    break

            closest = self._closest(row, centroids)
            self.clusters[closest].append(row)

    def _closest(self, fpoint, centroids):
        """Find the closest centroid for a point."""
        closest_index = None
        closest_distance = None
        for i, point in enumerate(centroids):
            dist = euclidean_distance(self.X[fpoint], point)
            if closest_index is None or dist < closest_distance:
                closest_index = i
                closest_distance = dist
        return closest_index

    def _get_centroid(self, cluster):
        """Get values by indices and take the mean."""
        return [np.mean(np.take(self.X[:, i], cluster)) for i in range(self.n_features)]


k = KMeans(K=5, max_iters=150, init='random', thres=10)

image = imread('mandrill.png')
image = imresize(image, (100, 100))

# on pixel
# X = image.reshape((-1, 3))  # each row is a sample with three dim (RGB)
# predictions = k.predict(X)

# cluster on the patch
Y = image.reshape((-1, 16))
predictions_Y = k.predict(Y)

for i in range(5):
    print('# of cluster {:d} is {:d}'.format(i, sum(predictions_Y==i)))

