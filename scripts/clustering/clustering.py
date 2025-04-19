from sklearn.cluster import KMeans, BisectingKMeans
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
nlist = 10000

# Example data (replace this with your actual data)
file = f"/home/chunxy/repos/Compass/data/{name}_base.float32"
training_data = np.fromfile(file, dtype=np.float32).reshape((-1, d))

# Perform KMeans++ clustering
kmeans = KMeans(n_clusters=nlist, random_state=42, init="k-means++")
# kmeans_labels = kmeans.fit_predict(training_data)
kmeans.fit(training_data)
centroids_kmeans = kmeans.cluster_centers_.astype(np.float32)
centroids_kmeans.tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.kmeans.centroids")

# Perform BisectingKMeans clustering
bikmeans = BisectingKMeans(n_clusters=nlist)
bikmeans.fit(training_data)
centroids_bisect_kmeans = bikmeans.cluster_centers_.astype(np.float32)
centroids_bisect_kmeans.tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.bikmeans.centroids")

# # Perform MeanShift clustering
# mean_shift = MeanShift()
# mean_shift_labels = mean_shift.fit_predict(training_data)
# centroids_meanshift = mean_shift.cluster_centers_.astype(np.float32)
# centroids_meanshift.tofile(f"{name}.centroids.meanshift")
