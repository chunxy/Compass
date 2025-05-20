from sklearn.cluster import HDBSCAN
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
args = parser.parse_args()

if args.name not in datasets:
  raise ValueError(f"Invalid dataset name: {args.name}. Must be one of {list(datasets.keys())}")

name = args.name
d = datasets[name]

file = f"/home/chunxy/repos/Compass/data/{name}_base.float32"
training_data = np.fromfile(file, dtype=np.float32).reshape((-1, d))

# Perform HDBSCAN clustering
hdbscan = HDBSCAN(min_cluster_size=100, store_centers="medoid")
hdbscan.fit(training_data)
centroids_hdbscan = hdbscan.centroids_.astype(np.float32)
nlist = len(centroids_hdbscan)
centroids_hdbscan.tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.hdbscan.medoids")
labels_hdbscan = hdbscan.labels_.astype(np.int64)
labels_hdbscan.tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.hdbscan.ranking")



