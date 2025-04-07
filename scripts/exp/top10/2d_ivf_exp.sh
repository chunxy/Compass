datasets=(sift gist crawl glove100 video audio)
nlist_s=(1000 5000 10000)
r_s=("1100 1200" "3100 3200" "5100 5200" "8100 8200" "9100 9200")
nprobe_s=(10 20 30 40 50)

for dataset in ${datasets[@]}; do
  for nlist in ${nlist_s[@]}; do
    for r in "${r_s[@]}"; do
      /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_ivf --datacard ${dataset}_2_10000_float32 --l 100 200 --r ${r} --k 10 --nlist ${nlist} --nprobe ${nprobe_s[@]}
    done
  done
done