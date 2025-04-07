datasets=(sift gist crawl glove100 video audio)
r_s=("1100 1200" "3100 3200" "5100 5200" "8100 8200" "9100 9200")
efs_s=(10 20 60 100 200)
nrel_s=(500 1000)

for dataset in ${datasets[@]}; do
  for r in "${r_s[@]}"; do
    /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_graph --datacard ${dataset}_2_10000_float32 --l 100 200 --r ${r} --k 10 --M 16 --efc 200 --efs ${efs_s[@]} --nrel ${nrel_s[@]}
  done
done