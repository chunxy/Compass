from pymilvus import MilvusClient
from utils import DA_S, CARDS
from utils import load_fvecs, load_ivecs
import time

# Token form
client = MilvusClient(
  uri="http://127.0.0.1:19530",
  token="root:Milvus",
)

for d in CARDS.keys():
  for da in DA_S:
    database_name = f"{d}_{da}".replace("-", "_")

    for card in CARDS[d][da]:
      query_vectors_np = load_fvecs(card.query_path, card.n_queries, card.dim)[:200]  # 200 queries only
      query_vectors = [query_vector.tolist() for query_vector in query_vectors_np]
      predicate = " and ".join([f"attr_{i} >= {card.interval[0][i]} and attr_{i} <= {card.interval[1][i]}" for i in range(card.attr_dim)])
      time_start = time.perf_counter_ns()
      res = client.search(
        collection_name=database_name,
        data=query_vectors,
        limit=10,  # Search for top-10.
        filter=predicate,
        output_fields=["id"],
        search_params={"hints": "iterative_filter"}
      )
      time_end = time.perf_counter_ns()
      time_taken = time_end - time_start

      groundtruth = load_ivecs(card.groundtruth_path, card.n_queries, card.n_groundtruth)[:200, :10]

      recall = 0
      for i, hits in enumerate(res):
        for hit in hits:
          if hit["id"] in groundtruth[i]:
            recall += 1
      print(f"Recall: {recall / 200 / 10}")
      print(f"QPS: {200 * 1e9 / time_taken}")
      exit()
