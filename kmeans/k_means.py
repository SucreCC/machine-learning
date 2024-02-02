import numpy as np


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        centroids = KMeans.centroids_init(self.data, self.num_clusters)

        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            closest_centroids_ids = KMeans.centroids_find_closest(self.data, centroids)

    @staticmethod
    def centroids_init(self, data, num_clusters):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_examples], :]
        return centroids

    def centroids_find_closest(self, data, centroids):
        num_examples = self.data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        for example_index in range(num_examples):
            distance = np.zeros(num_centroids, 1)
            for centroid_index in range(num_centroids):
                distance_diff = data[example_index, :] - centroids[centroid_index]
                distance[centroid_index] = np.sum((distance_diff ** 2))
            closest_centroids_ids[example_index] = np.argmin(distance)

        return  closest_centroids_ids
