# Generate hybrid attributes
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard siftsmall_1_1000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard sift_1_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard gist_1_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard crawl_1_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard glove100_1_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard audio_1_10000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard video_1_10000_float32

# Groundtruth preparation
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 300 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 600 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 1100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 2100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 3100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 4100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 5100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 6100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 7100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 8100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard sift_1_10000_float32 --l 100 --r 9100 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 300 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 600 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 1100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 2100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 3100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 4100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 5100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 6100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 7100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 8100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard gist_1_10000_float32 --l 100 --r 9100 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 300 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 600 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 1100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 2100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 3100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 4100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 5100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 6100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 7100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 8100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard crawl_1_10000_float32 --l 100 --r 9100 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 300 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 600 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 1100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 2100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 3100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 4100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 5100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 6100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 7100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 8100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard glove100_1_10000_float32 --l 100 --r 9100 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 300 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 600 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 1100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 2100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 3100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 4100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 5100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 6100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 7100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 8100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 100 --r 9100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard audio_1_10000_float32 --l 0 --r 10000 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 200 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 300 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 600 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 1100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 2100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 3100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 4100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 5100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 6100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 7100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 8100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 100 --r 9100 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth --datacard video_1_10000_float32 --l 0 --r 10000 --k 100