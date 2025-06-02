# Generate hybrid attributes
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard siftsmall_1_1000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard siftsmall_2_1000_float32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard siftsmall_1_10_int32
/home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard siftsmall_1_1000_int32

# Groundtruth preparation
/home/chunxy/repos/Compass/build/src/Release/apps/compute-groundtruth --datacard siftsmall_1_1000_float32 --l 100 --r 110 --k 100
/home/chunxy/repos/Compass/build/src/Release/apps/compute-groundtruth --datacard siftsmall_1_1000_float32 --l 100 --r 120 --k 100
/home/chunxy/repos/Compass/build/src/Release/apps/compute-groundtruth --datacard siftsmall_1_1000_float32 --l 100 --r 150 --k 100
/home/chunxy/repos/Compass/build/src/Release/apps/compute-groundtruth --datacard siftsmall_1_1000_float32 --l 100 --r 200 --k 100
/home/chunxy/repos/Compass/build/src/Release/apps/compute-groundtruth --datacard siftsmall_1_1000_float32 --l 100 --r 300 --k 100
/home/chunxy/repos/Compass/build/src/Release/apps/compute-groundtruth --datacard siftsmall_1_1000_float32 --l 100 --r 600 --k 100
/home/chunxy/repos/Compass/build/src/Release/apps/compute-groundtruth --datacard siftsmall_1_1000_float32 --l 100 --r 1000 --k 100

/home/chunxy/repos/Compass/build/src/Release/apps/compute-groundtruth --datacard siftsmall_2_1000_float32 --l 100 200 --r 600 700 --k 100

/home/chunxy/repos/Compass/build/Release/src/apps/compute_filter_groundtruth --datacard siftsmall_1_10_int32 --k 100
/home/chunxy/repos/Compass/build/Release/src/apps/compute_filter_groundtruth --datacard siftsmall_1_1000_int32 --k 100

# Experiment
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_1d --datacard siftsmall_1_1000_float32 --l 100 --r 110 --k 100 --M 16 --efc 200 --nlist 100 --efs 100 --nrel 50
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_1d --datacard siftsmall_1_1000_float32 --l 100 --r 120 --k 100 --M 16 --efc 200 --nlist 100 --efs 100 --nrel 50
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_1d --datacard siftsmall_1_1000_float32 --l 100 --r 150 --k 100 --M 16 --efc 200 --nlist 100 --efs 100 --nrel 50
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_1d --datacard siftsmall_1_1000_float32 --l 100 --r 200 --k 100 --M 16 --efc 200 --nlist 100 --efs 100 --nrel 50
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_1d --datacard siftsmall_1_1000_float32 --l 100 --r 300 --k 100 --M 16 --efc 200 --nlist 100 --efs 100 --nrel 50
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_1d --datacard siftsmall_1_1000_float32 --l 100 --r 600 --k 100 --M 16 --efc 200 --nlist 100 --efs 100 --nrel 50
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_1d --datacard siftsmall_1_1000_float32 --l 100 --r 1000 --k 100 --M 16 --efc 200 --nlist 100 --efs 100 --nrel 50

/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_acorn --datacard siftsmall_1_10_int32 --k 100
