#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#define debug 0
#define verbose_enabled 0
using namespace std;

float *sources;
float *queries;

int source_nb = 0;
int query_nb = 0;
int qrep_nb = 0;
int srep_nb = 0;
int dim = 0;
int K = 0;

typedef struct Point2Rep {
  int repIndex;
  float dist2rep;
} P2R;

typedef struct IndexAndDist {
  int index;
  float dist;
} IndexDist;

typedef struct repPoint_static {
  float maxdist;
  float mindist;
  uint npoints;
  uint noreps;
  float kuboundMax;
} R2all_static;

typedef struct repPoint_static_dev {
  float maxdist;
  float mindist;
  uint npoints;
  uint noreps;
  float kuboundMax;
} R2all_static_dev;

typedef struct repPoint_dynamic_v {
  vector<float> Vquerymembers;
  vector<IndexDist> Vsortedmembers;
  vector<int> Vreplist;
} R2all_dyn_v;

typedef struct repPoint_dynamic_p {
  int *memberID;
  IndexDist *sortedmembers;
  float *kubound;
  IndexDist *replist;
} R2all_dyn_p;

struct sort_dec {
  bool operator()(const IndexDist &left, const IndexDist &right) {
    return left.dist > right.dist;
  }
};

struct sort_inc {
  bool operator()(const IndexDist &left, const IndexDist &right) {
    return left.dist < right.dist;
  }
};

struct timespec t1, t2;

void timePoint(struct timespec &T1) { clock_gettime(CLOCK_REALTIME, &T1); }

float timeLen(struct timespec &T1, struct timespec &T2) {
  return T2.tv_sec - T1.tv_sec + (T2.tv_nsec - T1.tv_nsec) / 1.e9;
}

__device__ float Edistance_128(float *a, float *b, int dim = dim) {
  float distance = 0.0f;
  float4 *A = (float4 *)a;
  float4 *B = (float4 *)b;
  float tmp = 0.0f;
  for (int i = 0; i < int(dim / 4); i++) {
    float4 a_local = A[i];
    float4 b_local = __ldg(&B[i]);
    tmp = a_local.x - b_local.x;
    distance += tmp * tmp;
    tmp = a_local.y - b_local.y;
    distance += tmp * tmp;
    tmp = a_local.z - b_local.z;
    distance += tmp * tmp;
    tmp = a_local.w - b_local.w;
    distance += tmp * tmp;
  }
  for (int i = int(dim / 4) * 4; i < dim; i++) {
    tmp = (a[i]) - (b[i]);
    distance += tmp * tmp;
  }
  return distance;
}
__host__ __device__ float Edistance(float *A, float *B, int dim = dim) {
  float distance = 0.0f;
  for (int i = 0; i < dim; i++) {
    float tmp = A[i] - B[i];
    distance += tmp * tmp;
  }
  return distance;
}