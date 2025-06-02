datasets=(sift gist crawl glove100 video audio)

# Generate hybrid attributes
for dataset in ${datasets[@]}; do
  /home/chunxy/repos/Compass/build/Release/src/apps/generate_attributes --datacard ${dataset}_3_10000_float32
done

r1_s=(2100 6100 8100 9100)
r2_s=(2200 6200 8200 9200)
r3_s=(2300 6300 8300 9300)
# Groundtruth preparation for 1%, 20%, 50%, 70%.
for dataset in ${datasets[@]}; do
  i=0
  while [ $i -lt ${#r1_s[@]} ]; do
    r1=${r1_s[$i]}
    r2=${r2_s[$i]}
    r3=${r3_s[$i]}
    /home/chunxy/repos/Compass/build/Release/src/apps/compute_groundtruth \
    --datacard ${dataset}_3_10000_float32 --l 100 200 300 --r ${r1} ${r2} ${r3} --k 10
    i=$((i + 1))
  done
done
