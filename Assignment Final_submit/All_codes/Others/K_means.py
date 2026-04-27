import numpy as np

def kmeans(X, k, max_iter=100):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        clusters = []
        for point in X:
            distances = np.sqrt(np.sum((centroids - point)**2, axis=1))
            cluster = np.argmin(distances)
            clusters.append(cluster)
        clusters = np.array(clusters)

        new_centroids = []
        for i in range(k):
            points = X[clusters == i]
            if len(points) == 0:
                new_centroids.append(centroids[i])  # keep old centroid
            else:
                new_centroids.append(points.mean(axis=0))
        new_centroids = np.array(new_centroids)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


def calculate_error(X, clusters, centroids):
    error = 0
    for i in range(len(X)):
        centroid = centroids[clusters[i]]
        error += np.sum((X[i] - centroid)**2)
    return error


# Dataset
X = np.array([
    [1.0, 1.5],
    [1.5, 2.0],
    [5.0, 8.0],
    [6.0, 9.0],
    [1.0, 0.6],
    [9.0, 11.0]
])

# Run KMeans
clusters, centroids = kmeans(X, k=2, max_iter=10)
print("Clusters:", clusters)
print("Centroids:", centroids)

# Elbow Method Error Calculation
errors = []
K = range(1, len(X) + 1)

for k in K:
    clusters, centroids = kmeans(X, k=k, max_iter=10)
    error = calculate_error(X, clusters, centroids)
    errors.append(error.tolist())

print("Errors:", errors)