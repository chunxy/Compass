from sklearn.manifold import LocallyLinearEmbedding as LLE
import numpy as np
import argparse

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
parser.add_argument("--n_neighbors", type=int, required=True, help="Number of neighbors")
parser.add_argument("--n_components", type=int, required=True, help="Number of components")
args = parser.parse_args()

if args.name not in datasets:
  raise ValueError(f"Invalid dataset name: {args.name}. Must be one of {list(datasets.keys())}")

name = args.name
d = datasets[name]
n_neighbors = args.n_neighbors
n_components = args.n_components

file = f"/home/chunxy/repos/Compass/data/{name}_base.float32"
training_data = np.fromfile(file, dtype=np.float32).reshape((-1, d))

# Perform BisectingKMeans clustering
lle = LLE(n_neighbors=n_neighbors, n_components=n_components)
lle.fit(training_data)
centroids_lle = lle.embedding_.astype(np.float32)
centroids_lle.tofile(f"/home/chunxy/repos/Compass/data/{name}.{n_neighbors}.{n_components}.lle.centroids")



