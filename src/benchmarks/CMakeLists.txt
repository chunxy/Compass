link_libraries(Boost::filesystem)

# Baseline
add_executable(bench-compass-ivf-1d bench-compass-ivf-1d.cpp)
add_executable(bench-compass-ivf-1d-imi bench-compass-ivf-1d-imi.cpp)
add_executable(bench-compass-ivf-1d-kmeanspp bench-compass-ivf-1d-kmeans++.cpp)
add_executable(bench-compass-ivf-1d-bikmeans bench-compass-ivf-1d-bikmeans.cpp)
add_executable(bench-compass-graph-1d bench-compass-graph-1d.cpp)
add_executable(bench-hop-hnsw-1d bench-hop-hnsw-1d.cpp)
# add_executable(bench-compass-nsg bench-compass-nsg.cpp)

# Legacy
# add_executable(bench-compass-r-trick-1d bench-compass-r-trick-1d.cpp)
add_executable(bench-compass-1d-k-gr bench-compass-1d-k-gr.cpp)
add_executable(bench-compass-1d-k-or bench-compass-1d-k-or.cpp)
add_executable(bench-compass-1d-k-or-cg bench-compass-1d-k-or-cg.cpp)

# KMeans and variants
add_executable(bench-compass-1d-k bench-compass-1d-k.cpp)
add_executable(bench-compass-1d-k-cg bench-compass-1d-k-cg.cpp)
add_executable(bench-compass-1d-kmedoids bench-compass-1d-kmedoids.cpp)
add_executable(bench-compass-1d-kmedoids-cg bench-compass-1d-kmedoids-cg.cpp)
add_executable(bench-compass-1d-bikmeans bench-compass-1d-bikmeans.cpp)
add_executable(bench-compass-1d-bikmeans-cg bench-compass-1d-bikmeans-cg.cpp)
add_executable(bench-compass-1d-bikmeans-or bench-compass-1d-bikmeans-or.cpp)

# Decent clustering
add_executable(bench-compass-1d-pca bench-compass-1d-pca.cpp)
add_executable(bench-compass-1d-pca-cg bench-compass-1d-pca-cg.cpp)

# Novel clustering
add_executable(bench-compass-1d-llem bench-compass-1d-llem.cpp)
add_executable(bench-compass-1d-idec bench-compass-1d-idec.cpp)

# Baseline
add_executable(bench-compass-ivf bench-compass-ivf.cpp)
add_executable(bench-compass-graph bench-compass-graph.cpp)

# Legacy
add_executable(bench-compass-k-or bench-compass-k-or.cpp)
add_executable(bench-compass-k-or-cg bench-compass-k-or-cg.cpp)

# KMeans and variants
add_executable(bench-compass-k bench-compass-k.cpp)
add_executable(bench-compass-k-cg bench-compass-k-cg.cpp)
add_executable(bench-compass-bikmeans bench-compass-bikmeans.cpp)
add_executable(bench-compass-bikmeans-cg bench-compass-bikmeans-cg.cpp)

# Decent clustering
add_executable(bench-compass-pca bench-compass-pca.cpp)
add_executable(bench-compass-pca-cg bench-compass-pca-cg.cpp)

