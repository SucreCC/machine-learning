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
            # update the index of centroids
            centroids = KMeans.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)
        return centroids, closest_centroids_ids

    @staticmethod
    def centroids_init(data, num_clusters):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_clusters], :]
        return centroids

    @staticmethod
    def centroids_find_closest(self, data, centroids):
        num_examples = self.data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        for example_index in range(num_examples):
            distance = np.zeros(num_examples, 1)
            for centroid_index in range(num_centroids):
                distance_diff = data[example_index, :] - centroids[centroid_index, :]
                distance[centroid_index] = np.sum(distance_diff ** 2)
            closest_centroids_ids[example_index] = np.argmin(distance)
        return closest_centroids_ids

    @staticmethod
    def centroids_compute(cls, data, closest_centroids_ids, num_clusters):
        num_features = data.shape[1]
        centroids = np.zeros((num_clusters, num_features))
        for centroids_id in range(num_clusters):
            closest_ids = closest_centroids_ids == centroids_id
            centroids[closest_ids] = np.mean(data[closest_ids.flatten(), :], axis=0)
        return centroids
