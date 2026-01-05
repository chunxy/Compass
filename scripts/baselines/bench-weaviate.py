import weaviate
import json
from utils import CARDS, DA_S
from utils import load_fvecs, load_ivecs
from weaviate.classes.query import Filter

# Step 2.1: Connect to your local Weaviate instance
with weaviate.connect_to_local() as client:
  for d in CARDS.keys():
    for da in DA_S:
      # Step 1.2: Create a collection
      database_name = f"{d}_{da}".replace("-", "_")
      # Step 2.2: Use this collection
      db = client.collections.use(database_name)

      # Step 2.3: Perform a vector search with NearVector
      for card in CARDS[d][da]:
        query_vectors = load_fvecs(card.query_path, card.n_queries, card.dim)[:200]  # 200 queries only
        predicate = Filter.all_of([Filter.by_property(f"attr_{i}").greater_or_equal(card.interval[0][i]) for i in range(card.attr_dim)] +
                                  [Filter.by_property(f"attr_{i}").less_or_equal(card.interval[1][i]) for i in range(card.attr_dim)])

        responses = [0 for i in range(200)]  # 200 queries only
        for i, query_vector in enumerate(query_vectors):
          # Search for top-10.
          response = db.query.near_vector(near_vector=query_vector, filters=predicate, limit=10)
          responses[i] = response

        groundtruth = load_ivecs(card.groundtruth_path, card.n_queries, card.n_groundtruth)[:200, :10]

        for obj in response.objects:
          print(json.dumps(obj.properties, indent=2))  # Inspect the results
        exit()
