import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class KMedoids:

  def __init__(self, n_clusters, max_iter=300, random_state=None):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.random_state = random_state
    self.medoids = None
    self.labels_ = None

  def fit(self, X, bs1=100000, bs2=100000):
    np.random.seed(self.random_state)
    n_samples = X.shape[0]

    # Initialize medoids randomly
    initial_medoids_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
    self.medoids = X[initial_medoids_idx]

    for _ in range(self.max_iter):
      # Assign each point to the nearest medoid
      distances = np.zeros((X.shape[0], self.medoids.shape[0]), dtype=np.float32)
      for start in range(0, X.shape[0], bs1):
        end = min(start + bs1, X.shape[0])
        distances[start:end] = euclidean_distances(X[start:end], self.medoids)
      self.labels_ = np.argmin(distances, axis=1)

      # Update medoids
      new_medoids = np.copy(self.medoids)
      for i in range(self.n_clusters):
        cluster_points = X[self.labels_ == i]
        if len(cluster_points) > 0:
          num_points = len(cluster_points)
          pairwise_distances = np.zeros((num_points, num_points), dtype=np.float32)
          for start in range(0, num_points, bs2):
            end = min(start + bs2, num_points)
            # pairwise_distances[start:end] = np.linalg.norm(cluster_points[start:end, np.newaxis] - cluster_points, axis=2)
            pairwise_distances[start:end] = euclidean_distances(cluster_points[start:end], cluster_points)
          medoid_idx = np.argmin(np.sum(pairwise_distances, axis=1))
          new_medoids[i] = cluster_points[medoid_idx]

      # Check for convergence
      if np.all(new_medoids == self.medoids):
        break
      self.medoids = new_medoids

  def predict(self, X):
    distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
    return np.argmin(distances, axis=1)


datasets = {
  "gist": 960,
  "crawl": 300,
  "glove100": 100,
  "audio": 128,
  "video": 1024,
  "sift": 128,
}

import argparse

parser = argparse.ArgumentParser(description="Clustering script")
parser.add_argument("--name", type=str, required=True, help="Dataset name to process")
parser.add_argument("--nlist", type=int, required=True, help="Number of clusters")
args = parser.parse_args()

if args.name not in datasets:
  raise ValueError(f"Invalid dataset name: {args.name}. Must be one of {list(datasets.keys())}")

name = args.name
d = datasets[name]
nlist = args.nlist

# Example data (replace this with your actual data)
file = f"/home/chunxy/repos/Compass/data/{name}_base.float32"
training_data = np.fromfile(file, dtype=np.float32).reshape((-1, d))
kmedoid = KMedoids(nlist, 100)
kmedoid.fit(training_data)
kmedoid.medoids.astype(np.float32).tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.kmedoids.medoids")
