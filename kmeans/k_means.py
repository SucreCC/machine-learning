import numpy as np


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        # 1 get random centroids
        centroids = KMeans.centroids_init(self.data, self.num_clusters)
        # start to train data
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            # get the distance form centroids to other sample points
            closest_centroids_ids = KMeans.centroids_find_closest(self.data, centroids)


    @staticmethod
    def centroids_init(data, num_clusters):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_clusters], :]
        return centroids
