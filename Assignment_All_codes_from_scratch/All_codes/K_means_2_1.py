import numpy as np

# Load data
train = np.loadtxt("train.csv", delimiter=",")
val = np.loadtxt("val.csv", delimiter=",")
test = np.loadtxt("test.csv", delimiter=",")

# Input
Kmax = int(input().split("=")[1])
max_iter = int(input().split("=")[1])


# ---------- KMeans++ Initialization ----------
def init_centroids(X, k, rng):
    centroids = [X[rng.integers(len(X))]]
    
    for _ in range(1, k):
        distances = []
        for point in X:
            d = [np.sum((point - c)**2) for c in centroids]
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

        # Assign clusters (like your style)
        for point in X:
            d = [np.sum((point - c)**2) for c in centroids]
            clusters.append(np.argmin(d))
        
        clusters = np.array(clusters)

        # Update centroids
        new_centroids = []
        for i in range(k):
            points = X[clusters == i]
            if len(points) == 0:
                new_centroids.append(X[rng.integers(len(X))])
            else:
                new_centroids.append(points.mean(axis=0))
        
        new_centroids = np.array(new_centroids)

        # Stop if converged
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


# ---------- WCSS ----------
def compute_wcss(X, clusters, centroids):
    error = 0
    for i in range(len(X)):
        d = np.sum((X[i] - centroids[clusters[i]])**2)
        error += max(d, 0)   # numerical safety
    return error


# ---------- Step 1: Validation WCSS ----------
wcss = []

for k in range(1, Kmax + 1):
    rng = np.random.default_rng(0)
    clusters, centroids = kmeans(train, k, max_iter, rng)

    # assign validation points
    val_clusters = []
    for point in val:
        d = [np.sum((point - c)**2) for c in centroids]
        val_clusters.append(np.argmin(d))
    
    val_clusters = np.array(val_clusters)

    wcss.append(compute_wcss(val, val_clusters, centroids))


# ---------- Step 2: Elbow ----------
x = np.arange(1, Kmax + 1)

a = wcss[-1] - wcss[0]
b = 1 - Kmax
c_line = Kmax * wcss[0] - wcss[-1]

distances = []
for i in range(Kmax):
    d = abs(a * x[i] + b * wcss[i] + c_line) / np.sqrt(a*a + b*b)
    distances.append(d)

elbow_k = np.argmax(distances) + 1


# ---------- Step 3: Test ----------
rng = np.random.default_rng(0)
clusters, centroids = kmeans(train, elbow_k, max_iter, rng)

test_clusters = []
for point in test:
    d = [np.sum((point - c)**2) for c in centroids]
    test_clusters.append(np.argmin(d))

test_clusters = np.array(test_clusters)

test_wcss = compute_wcss(test, test_clusters, centroids)
test_j = test_wcss / len(test)


# ---------- Output ----------
print(f"Elbow_K={elbow_k}")
print("WCSS_Val=[" + ",".join(f"{v:.2f}" for v in wcss) + "]")
print(f"Test_J_Elbow={test_j:.2f} Test_WCSS_Elbow={test_wcss:.2f}")