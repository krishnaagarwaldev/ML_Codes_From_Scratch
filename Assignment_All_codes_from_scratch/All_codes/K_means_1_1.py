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
        # vectorized distance
        dist = np.min(((X[:, None] - centroids) ** 2).sum(axis=2), axis=1)
        prob = dist / dist.sum()
        centroids.append(X[rng.choice(len(X), p=prob)])

    return np.array(centroids)


# ---------- KMeans ----------
def kmeans(X, k, max_iter, rng):
    centroids = init_centroids(X, k, rng)

    for _ in range(max_iter):
        # assign clusters (NumPy)
        dist = ((X[:, None] - centroids) ** 2).sum(axis=2)
        clusters = np.argmin(dist, axis=1)

        # update centroids (simple loop)
        new_centroids = []
        for i in range(k):
            points = X[clusters == i]

            if len(points) == 0:
                new_centroids.append(X[rng.integers(len(X))])
            else:
                new_centroids.append(points.mean(axis=0))

        new_centroids = np.array(new_centroids)

        # stop if converged
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


# ---------- WCSS ----------
def calculate_error(X, clusters, centroids):
    # vectorized error
    return np.sum((X - centroids[clusters]) ** 2)


# ---------- Training ----------
rng = np.random.default_rng(0)

clusters, centroids = kmeans(train, K, max_iter, rng)

train_wcss = calculate_error(train, clusters, centroids)
train_j = train_wcss / len(train)


# ---------- Validation ----------
val_dist = ((val[:, None] - centroids) ** 2).sum(axis=2)
val_clusters = np.argmin(val_dist, axis=1)

val_wcss = calculate_error(val, val_clusters, centroids)
val_j = val_wcss / len(val)


# ---------- Test ----------
test_dist = ((test[:, None] - centroids) ** 2).sum(axis=2)
test_clusters = np.argmin(test_dist, axis=1)

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