from pathlib import Path

types = [{"shortcut": "bikmeans", "method": "BisectingKMeans"}]
# types = [{"shortcut": "kmedoids", "method": "KMedoids"}]
datasets = ["sift", "glove100", "gist", "crawl", "video", "audio"]
nlist_s = [1000, 2000, 5000, 10000, 20000]

for dataset in datasets:
  for nlist in nlist_s:
    for what_kmeans in types:
      template_opath = "/home/chunxy/repos/Compass/checkpoints/CompassR1d/{}/{}.ivf"
      old_ivf_path = template_opath.format(dataset, nlist)

      template_npath = "/home/chunxy/repos/Compass/checkpoints/{}/{}/{}.ivf"
      new_ivf_path = template_npath.format(what_kmeans["method"], dataset, nlist)

      template_centroid_path = "/home/chunxy/repos/Compass/data/{}.{}.{}.centroids"
      centroid_path = template_centroid_path.format(dataset, nlist, what_kmeans["shortcut"])

      if not Path(centroid_path).exists(): continue
      with open(new_ivf_path, "wb") as new_ivf:
        with open(old_ivf_path, "rb") as old_ivf:
          first_90 = old_ivf.read(90)
          new_ivf.write(first_90)
          num_floats = old_ivf.read(8)
          new_ivf.write(num_floats)
          stepsize = int.from_bytes(num_floats, "little", signed=False) * 4
          old_ivf.seek(stepsize, 1)
          with open(centroid_path, "rb") as centroid:
            centroids = centroid.read()
            new_ivf.write(centroids)
          remaining = old_ivf.read()
          new_ivf.write(remaining)
