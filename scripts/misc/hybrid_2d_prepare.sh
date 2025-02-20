# Generate hybrid attributes
/home/chunxy/repos/Compass/build/Release/src/apps/generate_attributes --datacard gist_2_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate_attributes --datacard crawl_2_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate_attributes --datacard glove100_2_10000_float32

# Groundtruth preparation
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard sift_2_10000_float32 --l 100 200 --r 9100 9200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 1100 1200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 8100 8200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard gist_2_10000_float32 --l 100 200 --r 9100 9200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 1100 1200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 8100 8200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard crawl_2_10000_float32 --l 100 200 --r 9100 9200 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 1100 1200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 3100 3200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 5100 5200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 8100 8200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth --datacard glove100_2_10000_float32 --l 100 200 --r 9100 9200 --k 100




