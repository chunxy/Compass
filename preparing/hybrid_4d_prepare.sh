datasets=(sift gist crawl glove100 video audio)

# Generate hybrid attributes
for dataset in ${datasets[@]}; do
  /home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard ${dataset}_4_10000_float32
done

r1_s=(3100 5600 8100 9100)
r2_s=(3200 5700 8200 9200)
r3_s=(3300 5800 8300 9300)
r4_s=(3400 5900 8400 9400)
# Groundtruth preparation for 1%, 10%, 40%, 65%.
for dataset in ${datasets[@]}; do
  i=0
  while [ $i -lt ${#r1_s[@]} ]; do
    r1=${r1_s[$i]}
    r2=${r2_s[$i]}
    r3=${r3_s[$i]}
    r4=${r4_s[$i]}
    /home/chunxy/repos/Compass/build/Release/src/apps/compute-groundtruth \
    --datacard ${dataset}_4_10000_float32 --l 100 200 300 400 --r ${r1} ${r2} ${r3} ${r4} --k 10
    i=$((i + 1))
  done
done
