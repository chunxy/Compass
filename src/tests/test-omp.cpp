#include <omp.h>
#include <cstdio>

void f() {
#pragma omp parallel for num_threads(8) schedule(static)
  for (int i = 0; i < 8; i++) {
    // if (omp_get_thread_num() == 0) {
    //   printf("\tno. of threads: %d\n", omp_get_num_threads());
    // }
    printf("\tinner: %d\n", i);
  }
}

int main() {
  // omp_set_nested(1);
// #pragma omp parallel for num_threads(2) schedule(static)
#pragma omp parallel
#pragma omp single
#pragma omp taskloop
  for (int j = 0; j < 4; j++) {
    // if (omp_get_thread_num() == 0) {
    //   printf("no. of threads: %d\n", omp_get_num_threads());
    // }
    printf("outer: %d\n", j);
    f();
  }
  return 0;
}