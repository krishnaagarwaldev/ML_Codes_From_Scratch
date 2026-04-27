import numpy as np

# Load data
train = np.loadtxt("train.csv", delimiter=",")
val = np.loadtxt("val.csv", delimiter=",")
test = np.loadtxt("test.csv", delimiter=",")

# Input
K = int(input().split("=")[1])
max_iter = int(input().split("=")[1])


# ---------- KMeans++ Initialization ----------
def init_centroids(X, k, rng):
    centroids = [X[rng.integers(len(X))]]

    for _ in range(1, k):
        distances = []
        for point in X:
            d = []
            for c in centroids:
                d.append(np.sum((point - c) ** 2))
            distances.append(min(d))

        distances = np.array(distances)
        prob = distances / distances.sum()
        index = rng.choice(len(X), p=prob)

        centroids.append(X[index])

    return np.array(centroids)


# ---------- KMeans ----------
def kmeans(X, k, max_iter, rng):
    centroids = init_centroids(X, k, rng)

    for _ in range(max_iter):
        clusters = []

        # assign clusters
        for point in X:
            d = []
            for c in centroids:
                d.append(np.sum((point - c) ** 2))
            clusters.append(np.argmin(d))

        clusters = np.array(clusters)

        # update centroids
        new_centroids = []
        for i in range(k):
            points = X[clusters == i]

            if len(points) == 0:
                new_centroids.append(X[rng.integers(len(X))])
            else:
                new_centroids.append(points.mean(axis=0))

        new_centroids = np.array(new_centroids)

        # stop if no change
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


# ---------- Error (WCSS) ----------
def calculate_error(X, clusters, centroids):
    error = 0
    for i in range(len(X)):
        error += np.sum((X[i] - centroids[clusters[i]]) ** 2)
    return error


# ---------- Training ----------
rng = np.random.default_rng(0)

clusters, centroids = kmeans(train, K, max_iter, rng)

train_wcss = calculate_error(train, clusters, centroids)
train_j = train_wcss / len(train)

# validation
val_clusters = []
for point in val:
    d = []
    for c in centroids:
        d.append(np.sum((point - c) ** 2))
    val_clusters.append(np.argmin(d))

val_clusters = np.array(val_clusters)

val_wcss = calculate_error(val, val_clusters, centroids)
val_j = val_wcss / len(val)

# test
test_clusters = []
for point in test:
    d = []
    for c in centroids:
        d.append(np.sum((point - c) ** 2))
    test_clusters.append(np.argmin(d))

test_clusters = np.array(test_clusters)

test_wcss = calculate_error(test, test_clusters, centroids)
test_j = test_wcss / len(test)


# ---------- Output ----------
print(f"Train_J={train_j:.2f}")
print(f"Train_WCSS={train_wcss:.2f}")
print(f"Val_J={val_j:.2f}")
print(f"Val_WCSS={val_wcss:.2f}")
print(f"Test_J={test_j:.2f}")
print(f"Test_WCSS={test_wcss:.2f}")

centroid_str = ";".join([f"({c[0]:.2f},{c[1]:.2f})" for c in centroids])
print(f"Centroids=[{centroid_str}]")