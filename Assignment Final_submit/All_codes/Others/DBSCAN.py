import numpy as np

## Part 1 :- DBSCAN

def euclidean_distance(x1, x2):
    return (np.sqrt(np.sum((x1-x2)**2)))
    
def region_query(X, point_idx, eps):
    neighbors = []
    for i in range(len(X)):
        # dist = np.linalg.norm(X[point_idx] - X[i])
        dist = euclidean_distance(X[point_idx], X[i])
        if dist <= eps:
            neighbors.append(i)
    return neighbors
    
def dbscan(X, eps, minpts):
    labels = [-1]*len(X)
    cluster_id = 0
    
    for i in range(len(X)):
        if labels[i] != -1:
            continue
        neighbors = region_query(X, i, eps)
        
        if len(neighbors) < minpts:
            labels[i] = -1
        else:
            # expand_cluster(X, labels, i, cluster_id, eps, minpts)
            
            labels[i] = cluster_id
            seed = set(neighbors)
            process = list(seed)
            j = 0
            while j<len(process):
                n = process[j]
                
                if labels[n] == -1:
                    labels[n] = cluster_id
                new_neighbors = region_query(X, n, eps)
                
                if len(new_neighbors) >= minpts:
                    neighbors += new_neighbors
                j += 1
            cluster_id += 1 
    return labels
    
X = np.array([[1,1], [1.2,1.1],[0.8, 0.9], [8,8], [8.2, 8.1], [25,25]])
labels = dbscan(X, eps=0.5, minpts=2)
print(labels)
