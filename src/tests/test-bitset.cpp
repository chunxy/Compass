
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <vector>
#include "roaring/roaring.hh"

void test_1m(int nset, int ntest) {
  std::cout << "Start testing with " << nset << " set out of 1000000 elements" << std::endl;
  roaring::Roaring bitset;
  std::set<int> set;
  // randomly sample nele numbers from 0 to 1000000 and add them to the bitset
  // fixing the random seed to 42
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dis(0, 1'000'000);
  for (int i = 0; i < nset; i++) {
    int rand_number = dis(gen);
    bitset.add(rand_number);
    set.insert(rand_number);
  }
  bitset.runOptimize();

  int *rand_numbers = new int[ntest];
  for (int i = 0; i < ntest; i++) {
    rand_numbers[i] = dis(gen);
  }

  // warmup
  int bitset_cnt = 0;
  auto start = std::chrono::high_resolution_clock::system_clock::now();
  for (int i = 0; i < ntest; i++) {
    if (bitset.contains(rand_numbers[i])) {
      bitset_cnt++;
    }
  }
  auto end = std::chrono::high_resolution_clock::system_clock::now();
  std::cout << "Bitset's time (no warmup) taken: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << " ns" << std::endl;

  bitset_cnt = 0;
  start = std::chrono::high_resolution_clock::system_clock::now();
  for (int i = 0; i < ntest; i++) {
    if (bitset.contains(rand_numbers[i])) {
      bitset_cnt++;
    }
  }
  end = std::chrono::high_resolution_clock::system_clock::now();
  std::cout << "Bitset's time taken: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
            << " ns" << std::endl;

  // warmup
  int set_cnt = 0;
  start = std::chrono::high_resolution_clock::system_clock::now();
  for (int i = 0; i < ntest; i++) {
    if (set.contains(rand_numbers[i])) {
      set_cnt++;
    }
  }
  end = std::chrono::high_resolution_clock::system_clock::now();
  std::cout << "Set's time (no warmup) taken: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << " ns" << std::endl;

  set_cnt = 0;
  start = std::chrono::high_resolution_clock::system_clock::now();
  for (int i = 0; i < ntest; i++) {
    if (set.contains(rand_numbers[i])) {
      set_cnt++;
    }
  }
  end = std::chrono::high_resolution_clock::system_clock::now();
  std::cout << "Set's time taken: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
            << " ns" << std::endl;
  std::cout << "Bitset's count: " << bitset_cnt << ", Set's count: " << set_cnt << std::endl;
}

int main() {
  int neles[] = {100000, 10000, 500000, 900000, 990000};
  for (int nele : neles) {
    test_1m(nele, 3000);
  }

  std::cout << "Start testing with binary search on 1000000 elements" << std::endl;
  std::vector<float> ints(1'000'000);
  for (int i = 0; i < 1'000'000; i++) {
    ints[i] = i;
  }

  const int n_binary_search = 1000;
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dis(0, 1'000'000);
  int ls[n_binary_search], rs[n_binary_search];
  for (int i = 0; i < n_binary_search; i++) {
    ls[i] = dis(gen);
    rs[i] = dis(gen);
  }

  auto start = std::chrono::high_resolution_clock::system_clock::now();

  for (int i = 0; i < 200; i++) {
    std::binary_search(ints.begin(), ints.end(), ls[i]);
    std::binary_search(ints.begin(), ints.end(), rs[i]);
  }
  auto end = std::chrono::high_resolution_clock::system_clock::now();
  std::cout << "Binary search time taken: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
            << " ns" << std::endl;
}