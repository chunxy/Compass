datasets=(sift gist crawl glove100 video audio)
M_s=(32)
r_s=(200 300 600 1100 2100 3100 4100 5100 6100 7100 8100 9100)
efs_s=(100 200 300 400 500 1000)
nrel_s=(100 200 500)

for dataset in ${datasets[@]}; do
  for M in ${M_s[@]}; do
    for r in ${r_s[@]}; do
      /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_graph_1d --datacard ${dataset}_1_10000_float32 --l 100 --r ${r} --k 10 --M ${M} --efc 200 --efs ${efs_s[@]} --nrel ${nrel_s[@]}
    done
    /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_graph_1d --datacard ${dataset}_1_10000_float32 --l 0 --r 10000 --k 10 --M ${M} --efc 200 --efs ${efs_s[@]} --nrel ${nrel_s[@]}
  done
done

# /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_r_1d --datacard video_1_10000_float32 --l 100 --r 200 --k 10 --M 16 --efc 200 --nlist 1000 --efs 100 --nrel 100
