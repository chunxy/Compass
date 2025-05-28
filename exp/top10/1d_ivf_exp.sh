datasets=(sift gist crawl glove100 video audio)
nlist_s=(1000 2000 5000 10000)
r_s=(200 300 600 1100 2100)
nprobe_s=(10 15 20 25 30 35 40 45 50)

for dataset in ${datasets[@]}; do
  for nlist in ${nlist_s[@]}; do
    for r in ${r_s[@]}; do
      /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_ivf_1d --datacard ${dataset}_1_10000_float32 --l 100 --r ${r} --k 10 --nlist ${nlist} --nprobe ${nprobe_s[@]}
    done
  done
done

# /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_ivf_1d --datacard sift_1_10000_float32 --l 100 --r 200 --k 10 --nlist 1000 --nprobe 10
