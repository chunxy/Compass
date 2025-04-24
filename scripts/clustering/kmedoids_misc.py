import argparse
import numpy as np

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
args = parser.parse_args()

if args.name not in datasets:
  raise ValueError(f"Invalid dataset name: {args.name}. Must be one of {list(datasets.keys())}")

name = args.name
d = datasets[name]
nlist = args.nlist

# Example data (replace this with your actual data)
file = f"/home/chunxy/repos/Compass/data/{name}_base.float32"
training_data = np.fromfile(file, dtype=np.float32).reshape((-1, d))


# from kmedoids import KMedoids
# subset = np.random.choice(training_data, 10000)
# kmedoid = KMedoids(nlist, metric="euclidean")
# kmedoid.fit(subset)
# centroids_kmedoids = kmedoid.cluster_centers_.astype(np.float32)
# centroids_kmedoids.tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.kmedoids.centroids")

from sklearn_extra.cluster import KMedoids
# subset = np.random.choice(training_data, 10000)
subset = training_data[:200000]
kmedoid = KMedoids(nlist, metric="euclidean", init="k-medoids++")
kmedoid.fit(subset)
centroids_kmedoids = kmedoid.cluster_centers_.astype(np.float32)
centroids_kmedoids.tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.kmedoids.medoids")