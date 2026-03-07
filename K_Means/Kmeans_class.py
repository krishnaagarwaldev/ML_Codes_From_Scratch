import numpy as np

class KMeans:

    def __init__(self, k=3, epochs=100):
        self.k = k
        self.epochs = epochs

    def fit(self, X):
        n, m = X.shape

        # Random centroids
        random_index = np.random.choice(n, self.k, replace=False)
        self.centroids = X[random_index]

        for i in range(self.epochs):
            clusters = []

            for x in X:
                distances = [np.sqrt(np.sum((x - c)**2)) for c in self.centroids]
                clusters.append(np.argmin(distances))

            clusters = np.array(clusters)

            new_centroids = []
            for j in range(self.k):
                points = X[clusters == j]
                new_centroids.append(np.mean(points, axis=0))

            self.centroids = np.array(new_centroids)

        self.labels = clusters

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [np.sqrt(np.sum((x - c)**2)) for c in self.centroids]
            predictions.append(np.argmin(distances))
        return np.array(predictions)

# --------------------------
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Train model
kmeans = KMeans(k=4)
kmeans.fit(X)

# Predict cluster labels
predictions = kmeans.predict(X)
print("Centroids:\n", kmeans.centroids)
print("Predicted cluster labels:\n", predictions)

# Visualize clusters
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=predictions, s=50, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.5)
plt.title("K-Means Clustering")
plt.show()