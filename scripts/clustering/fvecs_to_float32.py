import struct
import numpy as np


def read_entries(file_path):
  entries = []
  with open(file_path, 'rb') as f:
    while True:
      # Read the int32 number (4 bytes)
      int_data = f.read(4)
      if not int_data:
        break  # End of file
      num_floats = struct.unpack('i', int_data)[0]

      # Read the corresponding float32 numbers
      float_data = f.read(4 * num_floats)
      if len(float_data) < 4 * num_floats:
        raise ValueError("File ended unexpectedly while reading floats.")
      floats = struct.unpack(f'{num_floats}f', float_data)

      # Append the entry to the list
      entries.append([*floats])
      # break
  return entries


# Example usage
if __name__ == "__main__":
  file_paths = {
    "gist": "/home/chunxy/datasets/gist/gist_base.fvecs",
    "crawl": "/home/chunxy/datasets/crawl/crawl_base.fvecs",
    "glove100": "/home/chunxy/datasets/glove100/glove100_base.fvecs",
    "video": "/home/chunxy/datasets/video/video_base.fvecs",
    "audio": "/home/chunxy/datasets/audio/audio_base.fvecs",
    "sift": "/home/chunxy/datasets/sift/sift_base.fvecs",
  }  # Replace with your file path
  for d, path in file_paths.items():
    entries = read_entries(path)

    # Write entries to a binary file
    output_file_path = f"{d}_base.float32"  # Replace with your desired output file path
    data = np.array(entries, dtype=np.float32)
    data.tofile(f"/home/chunxy/repos/Compass/data/{d}_base.float32")
