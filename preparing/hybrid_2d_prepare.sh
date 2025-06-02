# Generate hybrid attributes
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard sift_2_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard gist_2_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard crawl_2_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard glove100_2_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard video_2_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard audio_2_10000_float32

# Groundtruth preparation 1%, 9%, 25%, 64%, 81%
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 1100 1200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 8100 8200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 9100 9200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 1100 1200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 8100 8200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 9100 9200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 1100 1200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 8100 8200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 9100 9200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 1100 1200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 8100 8200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 9100 9200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_2_10000_float32 --l 100 200 --r 1100 1200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_2_10000_float32 --l 100 200 --r 8100 8200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_2_10000_float32 --l 100 200 --r 9100 9200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_2_10000_float32 --l 100 200 --r 1100 1200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_2_10000_float32 --l 100 200 --r 8100 8200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_2_10000_float32 --l 100 200 --r 9100 9200 --k 100


# Groundtruth preparation 2%, 5%, 10%, 20%
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 1100 2200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 1100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 2100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 4100 5200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 1100 2200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 1100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 2100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 4100 5200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 1100 2200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 1100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 2100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 4100 5200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 1100 2200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 1100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 2100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 4100 5200 --k 100
