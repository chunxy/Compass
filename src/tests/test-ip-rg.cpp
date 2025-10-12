
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include "utils/predicate.h"

void test(int nquery, int ntest) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dis(0, 1'000'000);

  std::vector<InplaceRangeQuery<float>> queries;
  queries.reserve(nquery);
  for (int i = 0; i < nquery; i++) {
    int l = dis(gen);
    int r = dis(gen);
    if (l > r) {
      std::swap(l, r);
    }
    queries.emplace_back(l, r, 1, 1);
  }

  int *labels = new int[ntest];
  for (int i = 0; i < ntest; i++) {
    labels[i] = dis(gen);
  }

  int cnt = 0;
  auto begin = std::chrono::high_resolution_clock::system_clock::now();
  for (int j = 0; j < nquery; j++) {
    for (int i = 0; i < ntest; i++) {
      if (queries[j](labels[i])) {
        cnt++;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::system_clock::now();
  std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << " ns"
            << std::endl;
  std::cout << "Averaged per query: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / nquery
            << " ns" << std::endl;
  std::cout << "Count: " << cnt << std::endl;
}

int main() {
  int nrange[] = {200};
  for (int ele : nrange) {
    test(ele, 5000);
  }
}