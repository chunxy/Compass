import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import argparse

def proclus(data, k, l, max_iter=10, random_state=None):
  """
    PROCLUS (Projected Clustering) algorithm implementation.

    Parameters:
        data (np.ndarray): The dataset (n_samples, n_features).
        k (int): Number of clusters.
        l (int): Average number of dimensions for subspaces.
        max_iter (int): Maximum number of iterations.
        random_state (int): Random seed for reproducibility.

    Returns:
        cluster_labels (np.ndarray): Cluster labels for each data point.
        medoids (np.ndarray): Medoids of the clusters.
        subspaces (list): List of subspaces for each cluster.
    """
  np.random.seed(random_state)
  n_samples, n_features = data.shape

  # Step 1: Select initial medoids
  medoid_indices = np.random.choice(n_samples, k, replace=False)
  medoids = data[medoid_indices]

  # Step 2: Iteratively refine medoids and subspaces
  for _ in range(max_iter):
    # Assign points to the nearest medoid
    distances = euclidean_distances(data, medoids)
    cluster_labels = np.argmin(distances, axis=1)

    # Update medoids
    for i in range(k):
      cluster_points = data[cluster_labels == i]
      if len(cluster_points) > 0:
        medoids[i] = cluster_points[np.argmin(np.sum(euclidean_distances(cluster_points, cluster_points), axis=1))]

    # Determine subspaces for each cluster
    subspaces = []
    for i in range(k):
      cluster_points = data[cluster_labels == i]
      if len(cluster_points) > 0:
        # Compute variances along each dimension
        variances = np.var(cluster_points, axis=0)
        # Select dimensions with the smallest variances
        subspace = np.argsort(variances)[:l]
        subspaces.append(subspace)
      else:
        subspaces.append(np.array([]))

    # Reassign points based on subspaces
    for i in range(k):
      subspace = subspaces[i]
      if len(subspace) > 0:
        distances[:, i] = np.linalg.norm(data[:, subspace] - medoids[i, subspace], axis=1)
      else:
        distances[:, i] = np.inf
    cluster_labels = np.argmin(distances, axis=1)

  return cluster_labels, medoids, subspaces


datasets = {
  "gist": 960,
  "crawl": 300,
  "glove100": 100,
  "audio": 128,
  "video": 1024,
  "sift": 128,
}

parser = argparse.ArgumentParser(description="Clustering script")
parser.add_argument("--name", type=str, required=True, help="Dataset name to process")
parser.add_argument("--nlist", type=int, required=True, help="Number of clusters")
parser.add_argument("--ndim", type=int, required=True, help="Intrinsic dimension")
args = parser.parse_args()

if args.name not in datasets:
  raise ValueError(f"Invalid dataset name: {args.name}. Must be one of {list(datasets.keys())}")

name = args.name
d = datasets[name]
nlist = args.nlist
ndim = args.ndim

print(f"Running {name}: {d} -> {ndim}")

# Example data (replace this with your actual data)
file = f"/home/chunxy/repos/Compass/data/{name}_base.float32"
data = np.fromfile(file, dtype=np.float32).reshape((-1, d))

labels, medoids, subspaces = proclus(data, nlist, ndim, random_state=42)
labels.tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.{ndim}.proclus.ranking")
medoids.tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.{ndim}.proclus.medoids")
np.array(subspaces).tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.{ndim}.proclus.subspaces")

# # Example usage
# if __name__ == "__main__":
#   # Generate synthetic data
#   np.random.seed(42)
#   data = np.random.rand(1000, 10)

#   # Run PROCLUS
#   nlist = 10  # Number of clusters
#   ndim = 3  # Average number of dimensions for subspaces
#   labels, medoids, subspaces = proclus(data, nlist, ndim, random_state=42)

#   print("Cluster labels:", labels)
#   print("Medoids:", medoids)
#   print("Subspaces:", subspaces)
#   labels.tofile(f"/home/chunxy/repos/Compass/test.{nlist}.{ndim}.proclus.ranking")
#   medoids.tofile(f"/home/chunxy/repos/Compass/test.{nlist}.{ndim}.proclus.medoids")
#   np.array(subspaces).tofile(f"/home/chunxy/repos/Compass/test.{nlist}.{ndim}.proclus.subspaces")
