# Brandon Gatewood
# CS 445
# Program 3: k-means

# This is a simple implementation for the k-means algorithm. The number of clusters and max iterations can be manually
# changed. The program will randomly initialize k centroids and run the k-means algorithm. It will start off with
# assigning each data point to the cluster whose mean has the least squared euclidean distance. Then it will calculate
# new means to be the centroids of the data points in the new clusters. This will repeat until the algorithm converges
# or max iterations exceeds.

import csv
import numpy as np
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
    # k = 5
    k = 10
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
        sorted_points = None

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

        # calculate sum square error
        sse = 0
        for i in range(self.k):
            sse += np.sum(sorted_points[i] - self.centroids[i])

        return self.centroids, sse


# Run the algorithm r times and select the solution that gives the lowest sum of squares error
r = 10
centroid_array = []
sse_array = []

for i in range(r):
    kmeans = KMEANS()
    i_centroid, i_sse = kmeans.fit()
    centroid_array.append(i_centroid)
    sse_array.append(i_sse)

# print data and centroids
i = np.argmin(sse_array)
print(sse_array[i])
plt.scatter(data[:, 0], data[:, 1], c="red")
plt.plot([x for x, _ in centroid_array[i]], [y for _, y in centroid_array[i]], '+', markersize=10)
plt.show()
