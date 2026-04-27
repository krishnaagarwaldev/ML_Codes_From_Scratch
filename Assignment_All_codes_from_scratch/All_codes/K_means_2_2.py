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
        dist = np.min(((X[:, None] - centroids) ** 2).sum(axis=2), axis=1)
        prob = dist / dist.sum()
        centroids.append(X[rng.choice(len(X), p=prob)])
    
    return np.array(centroids)


# ---------- KMeans ----------
def kmeans(X, k, max_iter, rng):
    centroids = init_centroids(X, k, rng)
    
    for _ in range(max_iter):
        # assign clusters (vectorized)
        dist = ((X[:, None] - centroids) ** 2).sum(axis=2)
        labels = np.argmin(dist, axis=1)

        # update centroids
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i)
            else X[rng.integers(len(X))]
            for i in range(k)
        ])

        # stop if converged
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids


# ---------- WCSS ----------
def compute_wcss(X, centroids):
    dist = ((X[:, None] - centroids) ** 2).sum(axis=2)
    dist = np.clip(dist, 0, None)   # numerical safety
    return np.sum(np.min(dist, axis=1))


# ---------- Step 1: Validation WCSS ----------
wcss = []
for k in range(1, Kmax + 1):
    rng = np.random.default_rng(0)
    c = kmeans(train, k, max_iter, rng)
    wcss.append(compute_wcss(val, c))


# ---------- Step 2: Elbow ----------
x = np.arange(1, Kmax + 1)

a = wcss[-1] - wcss[0]
b = 1 - Kmax
c_line = Kmax * wcss[0] - wcss[-1]

distances = np.abs(a * x + b * np.array(wcss) + c_line) / np.sqrt(a*a + b*b)
elbow_k = np.argmax(distances) + 1


# ---------- Step 3: Test ----------
rng = np.random.default_rng(0)
final_c = kmeans(train, elbow_k, max_iter, rng)

test_wcss = compute_wcss(test, final_c)
test_j = test_wcss / len(test)


# ---------- Output ----------
print(f"Elbow_K={elbow_k}")
print("WCSS_Val=[" + ",".join(f"{v:.2f}" for v in wcss) + "]")
print(f"Test_J_Elbow={test_j:.2f} Test_WCSS_Elbow={test_wcss:.2f}")