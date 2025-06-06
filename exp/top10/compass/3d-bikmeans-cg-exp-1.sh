datasets=(crawl)
M_s=(16 32)
efc_s=(200)
nlist_s=(10000 20000)
efs_s=(10 20 60 100 200 300)
nrel_s=(100 200)
for dataset in ${datasets[@]}; do
for M in ${M_s[@]}; do
for efc in ${efc_s[@]}; do
for nlist in ${nlist_s[@]}; do
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-bikmeans-cg --datacard ${dataset}_3_10000_float32 --l 100 200 300 --r 2100 2200 2300 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-bikmeans-cg --datacard ${dataset}_3_10000_float32 --l 100 200 300 --r 6100 6200 6300 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-bikmeans-cg --datacard ${dataset}_3_10000_float32 --l 100 200 300 --r 8100 8200 8300 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-bikmeans-cg --datacard ${dataset}_3_10000_float32 --l 100 200 300 --r 9100 9200 9300 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
done
done
done
done