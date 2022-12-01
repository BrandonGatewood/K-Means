# Brandon Gatewood
# CS 445
# Program 3: k-means

import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = []
f = open('cluster_data.txt', 'r')
read = csv.reader(f)

for row in read:
    r = row[0].split()
    data.append(r)

data = np.array(list(np.float_(data)))


# KMEANS class contains the k-means algorithm, it will assign the number of clusters and randomly initialize k
# centroids.
class KMEANS:
    # Experiment with different k values
    # k = 3
    k = 5
    # k = 10
    max_iter = 300
    centroids = np.zeros(k)

    def __init__(self):
        # Randomly initialize k clusters
        self.centroids = data[np.random.choice(data.shape[0], size=self.k, replace=False), :]

    # Find the euclidean distance between a data point and clusters point
    def euclidean(self, x):
        return np.sqrt(np.sum((x - self.centroids) ** 2, axis=1))

    # Runs the k-means algorithm until converges or max iterations exceeded
    def fit(self):
        iteration = 0
        prev_centroids = None

        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each data point and assign it the nearest cluster
            sorted_points = [[] for _ in range(self.k)]

            for x in data:
                distances = self.euclidean(x)
                centroid_index = np.argmin(distances)
                sorted_points[centroid_index].append(x)

            # Save current centroid as previous and update new centroid
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]

            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def eval(self):
        centroids = []
        centroids_indexes = []
        for x in data:
            distances = self.euclidean(x)
            centroids_index = np.argmin(distances)
            centroids.append(self.centroids[centroids_index])
            centroids_indexes.append(centroids_index)

        return centroids, centroids_indexes


# Run the algorithm r times and select the solution that gives the lowest sum of squares error
r = 10
center = []
classification = []
sse = []

for i in range(r):
    kmeans = KMEANS()
    kmeans.fit()
    cntr, clss = kmeans.eval()
    center.append(cntr)
    classification.append(clss)
    sse.append(np.sum(len(clss) * np.var(clss)))

i = np.argmin(sse)

plt.scatter(data[:, 0], data[:, 1], c="red")

plt.plot([x for x, _ in center[i]], [y for _, y in center[i]], '+', markersize=10)
plt.show()
