/*i
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/cumlHandle.hpp"
#include <cuml/neighbors/knn.hpp>
#include "ml_mg_utils.h"
#include "selection/knn.h"
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <sstream>
#include <vector>

/* the following includes are for Sweet KNN */
#include <cuml/neighbors/common8.h>
#include <curand.h>
#include <curand_kernel.h>
#include <omp.h>
#include <pthread.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <cmath>
#include "cublas_v2.h"

namespace ML {

/**
   * @brief Flat C++ API function to perform a brute force knn on
   * a series of input arrays and combine the results into a single
   * output array for indexes and distances.
   *
   * @param handle the cuml handle to use
   * @param input an array of pointers to the input arrays
   * @param sizes an array of sizes of input arrays
   * @param n_params array size of input and sizes
   * @param D the dimensionality of the arrays
   * @param search_items array of items to search of dimensionality D
   * @param n number of rows in search_items
   * @param res_I the resulting index array of size n * k
   * @param res_D the resulting distance array of size n * k
   * @param k the number of nearest neighbors to return
   */
void brute_force_knn(cumlHandle &handle, float **input, int *sizes,
                     int n_params, int D, float *search_items, int n,
                     long *res_I, float *res_D, int k) {
  MLCommon::Selection::brute_force_knn(input, sizes, n_params, D, search_items,
                                       n, res_I, res_D, k,
                                       handle.getImpl().getStream());
}
__device__ __forceinline__ int F2I(float floatVal) {
  int intVal = __float_as_int(floatVal);
  return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float I2F(int intVal) {
  return __int_as_float((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}
__device__ float atomicMin_float(float *address, float val) {
  int val_int = F2I(val);
  int old = atomicMin((int *)address, val_int);
  return I2F(old);
}
__device__ float atomicMax_float(float *address, float val) {
  int val_int = F2I(val);
  int old = atomicMax((int *)address, val_int);
  return I2F(old);
}
__device__ float atomicAdd_float(float *address, float val) {
  int val_int = F2I(val);
  int old = atomicAdd((int *)address, val_int);
  return I2F(old);
}

void check(cudaError_t status, const char *message) {
  if (status != cudaSuccess) cout << message << endl;
}
void inline checkError(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("%s", msg);
    exit(EXIT_FAILURE);
  }
}

__global__ void Norm(float *point, float *norm, int size, int dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
      float tmp = point[tid * dim + i];
      dist += tmp * tmp;
    }
    norm[tid] = dist;
  }
}
__global__ void AddAll(float *queryNorm_dev, float *repNorm_dev,
                       float *query2reps_dev, int size, int rep_nb) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;
  if (tx < size && ty < rep_nb) {
    float temp = query2reps_dev[ty * size + tx];
    temp += (queryNorm_dev[tx] + repNorm_dev[ty]);
    query2reps_dev[ty * size + tx] = sqrt(temp);
  }
}
__global__ void findQCluster(float *query2reps_dev, P2R *q2rep_dev, int size,
                             int rep_nb, float *maxquery_dev,
                             R2all_static_dev *req2q_static_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < size) {
    float temp = FLT_MAX;
    int index = -1;
    for (int i = 0; i < rep_nb; i++) {
      float tmp = query2reps_dev[i * size + tid];
      if (temp > tmp) {
        index = i;
        temp = tmp;
      }
    }
    q2rep_dev[tid] = {index, temp};
    atomicAdd(&req2q_static_dev[index].npoints, 1);
    atomicMax_float(&maxquery_dev[index], temp);
  }
}
__global__ void findTCluster(float *source2reps_dev, P2R *s2rep_dev, int size,
                             int rep_nb, R2all_static_dev *req2s_static_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < size) {
    float temp = FLT_MAX;
    int index = -1;
    for (int i = 0; i < rep_nb; i++) {
      float tmp = source2reps_dev[i * size + tid];
      if (temp > tmp) {
        index = i;
        temp = tmp;
      }
    }
    s2rep_dev[tid] = {index, temp};
    atomicAdd(&req2s_static_dev[index].npoints, 1);
  }
}
__global__ void fillQMembers(P2R *q2rep_dev, int size, int *repsID,
                             R2all_dyn_p *req2q_dyn_p_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < size) {
    int repId = q2rep_dev[tid].repIndex;
    int memberId = atomicAdd(&repsID[repId], 1);
    req2q_dyn_p_dev[repId].memberID[memberId] = tid;
  }
}
__global__ void fillTMembers(P2R *s2rep_dev, int size, int *repsID,
                             R2all_dyn_p *req2s_dyn_p_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < size) {
    int repId = s2rep_dev[tid].repIndex;
    int memberId = atomicAdd(&repsID[repId], 1);
    req2s_dyn_p_dev[repId].sortedmembers[memberId] = {tid,
                                                      s2rep_dev[tid].dist2rep};
  }
}
__device__ int reorder = 0;
__global__ void reorderMembers(int rep_nb, int *repsID, int *reorder_members,
                               R2all_dyn_p *req2q_dyn_p_dev) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < rep_nb) {
    if (repsID[tid] != 0) {
      int reorderId = atomicAdd(&reorder, repsID[tid]);
      memcpy(reorder_members + reorderId, req2q_dyn_p_dev[tid].memberID,
             repsID[tid] * sizeof(int));
    }
  }
}

__global__ void selectReps_cuda(float *queries_dev, int qrep_nb,
                                int *qIndex_dev, int *totalSum_dev,
                                int totalTest, int dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < totalTest * qrep_nb * qrep_nb) {
    int test = tid / (qrep_nb * qrep_nb);
    int repId = int(tid % (qrep_nb * qrep_nb)) / qrep_nb;
    float distance = Edistance_128(
      queries_dev + qIndex_dev[test * qrep_nb + repId] * dim,
      queries_dev +
        qIndex_dev[test * qrep_nb + int(tid % (qrep_nb * qrep_nb)) % qrep_nb] *
          dim,
      dim);
    atomicAdd(&totalSum_dev[test], int(distance));
  }
}
__device__ int repTest = 0;

__global__ void selectReps_max(int *totalSum_dev, int totalTest) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    float distance = 0.0f;
    for (int i = 0; i < totalTest; i++) {
      if (distance < totalSum_dev[i]) {
        distance = totalSum_dev[i];
        repTest = i;
      }
    }
    #if verbose_enabled
    printf("repTest %d\n", repTest);
    #endif
   
  }
}

__global__ void selectReps_copy(float *queries_dev, float *qreps_dev,
                                int qrep_nb, int *qIndex_dev, int dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < qrep_nb) {
    memcpy(qreps_dev + tid * dim,
           queries_dev + qIndex_dev[repTest * qrep_nb + tid] * dim,
           dim * sizeof(float));
  }
}
// get last error and print it
void print_last_error() {
  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    printf("cudaError: cudaGetLastError() returned %d: %s - %s\n",
           cudaError, cudaGetErrorName(cudaError),
           cudaGetErrorString(cudaError));  
  }
}

void clusterReps(float *&queries_dev, float *&sources_dev, float *&qreps_dev,
                 float *&sreps_dev, float *&maxquery_dev, P2R *&q2rep_dev,
                 P2R *&s2rep_dev, R2all_static_dev *&rep2q_static_dev,
                 R2all_static_dev *&rep2s_static_dev,
                 R2all_dyn_p *&rep2q_dyn_p_dev, R2all_dyn_p *&rep2s_dyn_p_dev,
                 float *&query2reps_dev, R2all_static *&rep2q_static,
                 R2all_static *&rep2s_static, R2all_dyn_p *&rep2q_dyn_p,
                 R2all_dyn_p *&rep2s_dyn_p, int *&reorder_members) {
  cublasHandle_t cublas_handle;
  checkError(cublasCreate(&cublas_handle), "cublasCreate() error!\n");

  cudaError_t status;
  status =
    cudaMalloc((void **)&query2reps_dev, qrep_nb * query_nb * sizeof(float));
  check(status, "cMalloc 1 failed \n");

  status = cudaMalloc((void **)&queries_dev, query_nb * dim * sizeof(float));
  check(status, "Malloc queries failed\n");
  status = cudaMemcpy(queries_dev, queries, query_nb * dim * sizeof(float),
                      cudaMemcpyHostToDevice);
  check(status, "Memcpy queries failed\n");

  status = cudaMalloc((void **)&sources_dev, source_nb * dim * sizeof(float));
  check(status, "Malloc sources failed\n");
  status = cudaMemcpy(sources_dev, sources, source_nb * dim * sizeof(float),
                      cudaMemcpyHostToDevice);
  check(status, "Mem sources failed\n");

  status = cudaMalloc((void **)&qreps_dev, qrep_nb * dim * sizeof(float));
  check(status, "Malloc reps failed\n");

  status = cudaMalloc((void **)&sreps_dev, srep_nb * dim * sizeof(float));
  check(status, "Malloc reps failed\n");

  int totalTest = 10;
  int *qIndex_dev, *qIndex;
  qIndex = (int *)malloc(totalTest * qrep_nb * sizeof(int));

  status = cudaMalloc((void **)&qIndex_dev, qrep_nb * totalTest * sizeof(int));
  check(status, "cMalloc2 failed\n");

  srand(2015);
  for (int i = 0; i < totalTest; i++)
    for (int j = 0; j < qrep_nb; j++)
      qIndex[i * qrep_nb + j] = rand() % query_nb;
  cudaMemcpy(qIndex_dev, qIndex, totalTest * qrep_nb * sizeof(int),
             cudaMemcpyHostToDevice);
  int *totalSum_dev;

  status = cudaMalloc((void **)&totalSum_dev, totalTest * sizeof(float));
  check(status, "cMalloc 3 failed\n");
  cudaMemset(totalSum_dev, 0, totalTest * sizeof(float));

  selectReps_cuda<<<(totalTest * qrep_nb * qrep_nb + 255) / 256, 256>>>(
    queries_dev, qrep_nb, qIndex_dev, totalSum_dev, totalTest, dim);
  cudaDeviceSynchronize();
  print_last_error();

  selectReps_max<<<1, 1>>>(totalSum_dev, totalTest);
  selectReps_copy<<<(qrep_nb + 255) / 256, 256>>>(queries_dev, qreps_dev,
                                                  qrep_nb, qIndex_dev, dim);

  qIndex = (int *)malloc(totalTest * srep_nb * sizeof(int));

  status = cudaMalloc((void **)&qIndex_dev, srep_nb * totalTest * sizeof(int));
  check(status, "cMalloc 4 failed\n");

  srand(2015);
  for (int i = 0; i < totalTest; i++)
    for (int j = 0; j < srep_nb; j++)
      qIndex[i * srep_nb + j] = rand() % source_nb;
  cudaMemcpy(qIndex_dev, qIndex, totalTest * srep_nb * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMemset(totalSum_dev, 0, totalTest * sizeof(float));

  selectReps_cuda<<<(totalTest * srep_nb * srep_nb + 255) / 256, 256>>>(
    sources_dev, srep_nb, qIndex_dev, totalSum_dev, totalTest, dim);

  selectReps_max<<<1, 1>>>(totalSum_dev, totalTest);

  selectReps_copy<<<(srep_nb + 255) / 256, 256>>>(sources_dev, sreps_dev,
                                                  srep_nb, qIndex_dev, dim);
  cudaDeviceSynchronize();

  status =
    cudaMalloc((void **)&rep2q_static_dev, qrep_nb * sizeof(R2all_static_dev));
  check(status, "Malloc rep2qs_static failed\n");
  cudaMemcpy(rep2q_static_dev, rep2q_static, qrep_nb * sizeof(R2all_static_dev),
             cudaMemcpyHostToDevice);

  status =
    cudaMalloc((void **)&rep2s_static_dev, srep_nb * sizeof(R2all_static_dev));
  check(status, "Malloc rep2qs_static failed\n");
  cudaMemcpy(rep2s_static_dev, rep2s_static, srep_nb * sizeof(R2all_static_dev),
             cudaMemcpyHostToDevice);

  float *queryNorm_dev, *qrepNorm_dev, *sourceNorm_dev, *srepNorm_dev;

  status = cudaMalloc((void **)&queryNorm_dev, query_nb * sizeof(float));
  check(status, "cMalloc 5 failed\n");
  status = cudaMalloc((void **)&sourceNorm_dev, source_nb * sizeof(float));
  check(status, "cMalloc 6 failed\n");

  status = cudaMalloc((void **)&qrepNorm_dev, qrep_nb * sizeof(float));
  check(status, "cMalloc 7 failed\n");

  status = cudaMalloc((void **)&srepNorm_dev, srep_nb * sizeof(float));
  check(status, "cMalloc 8 failed\n");

  struct timespec t3, t4, t35;
  timePoint(t3);

  const float alpha = -2.0f;
  const float beta = 0.0f;
  cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, query_nb, qrep_nb, dim,
              &alpha, queries_dev, dim, qreps_dev, dim, &beta, query2reps_dev,
              query_nb);
  cudaDeviceSynchronize();
  timePoint(t35);
  #if verbose_enabled
  printf("cublasSgemm warm up time %f\n", timeLen(t3, t35));
  #endif
 
  timePoint(t1);
  Norm<<<(query_nb + 255) / 256, 256>>>(queries_dev, queryNorm_dev, query_nb,
                                        dim);

  cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, query_nb, qrep_nb, dim,
              &alpha, queries_dev, dim, qreps_dev, dim, &beta, query2reps_dev,
              query_nb);

  cudaDeviceSynchronize();
  timePoint(t3);
  Norm<<<(qrep_nb + 255) / 256, 256>>>(qreps_dev, qrepNorm_dev, qrep_nb, dim);
  dim3 block2D(16, 16, 1);
  dim3 grid2D_q((query_nb + 15) / 16, (qrep_nb + 15) / 16, 1);
  AddAll<<<grid2D_q, block2D>>>(queryNorm_dev, qrepNorm_dev, query2reps_dev,
                                query_nb, qrep_nb);

  status = cudaMalloc((void **)&maxquery_dev, qrep_nb * sizeof(float));
  check(status, "cMalloc 9 failed\n");

  cudaMemset(maxquery_dev, 0, qrep_nb * sizeof(float));

  status = cudaMalloc((void **)&q2rep_dev, query_nb * sizeof(P2R));
  check(status, "Malloc q2rep failed\n");
  findQCluster<<<(query_nb + 255) / 256, 256>>>(query2reps_dev, q2rep_dev,
                                                query_nb, qrep_nb, maxquery_dev,
                                                rep2q_static_dev);

  timePoint(t35);
  #if verbose_enabled
  printf("query rep first part time %f\n", timeLen(t3, t35));
  #endif
 

  int *qrepsID;
  status = cudaMalloc((void **)&qrepsID, qrep_nb * sizeof(int));
  check(status, "cMalloc 10 failed\n");

  cudaMemset(qrepsID, 0, qrep_nb * sizeof(int));

  cudaMemcpy(rep2q_static, rep2q_static_dev, qrep_nb * sizeof(R2all_static_dev),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < qrep_nb; i++) {
    status =
      cudaMalloc((void **)&rep2q_dyn_p[i].replist, srep_nb * sizeof(IndexDist));
    check(status, "cMalloc 11 failed\n");

    status = cudaMalloc((void **)&rep2q_dyn_p[i].kubound, K * sizeof(float));
    check(status, "cMalloc 12 failed\n");

    status = cudaMalloc((void **)&rep2q_dyn_p[i].memberID,
                        rep2q_static[i].npoints * sizeof(int));
    check(status, "cMalloc 13 failed\n");
  }

  status = cudaMalloc((void **)&rep2q_dyn_p_dev, qrep_nb * sizeof(R2all_dyn_p));
  check(status, "cMalloc 14 failed\n");

  cudaMemcpy(rep2q_dyn_p_dev, rep2q_dyn_p, qrep_nb * sizeof(R2all_dyn_p),
             cudaMemcpyHostToDevice);
  fillQMembers<<<(query_nb + 255) / 256, 256>>>(q2rep_dev, query_nb, qrepsID,
                                                rep2q_dyn_p_dev);

  status = cudaMalloc((void **)&reorder_members, query_nb * sizeof(int));
  check(status, "cMalloc 15 failed\n");

  reorderMembers<<<(qrep_nb + 255) / 256, 256>>>(
    qrep_nb, qrepsID, reorder_members, rep2q_dyn_p_dev);

  cudaDeviceSynchronize();
  timePoint(t4);
  #if verbose_enabled
  printf("query rep time  %f\n", timeLen(t3, t4));
  #endif
   

  float *source2reps_dev;
  status =
    cudaMalloc((void **)&source2reps_dev, source_nb * srep_nb * sizeof(float));
  check(status, "cMalloc 16 failed\n");

  cudaDeviceSynchronize();
  timePoint(t3);
  Norm<<<(source_nb + 255) / 256, 256>>>(sources_dev, sourceNorm_dev, source_nb,
                                         dim);

  cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, source_nb, srep_nb, dim,
              &alpha, sources_dev, dim, sreps_dev, dim, &beta, source2reps_dev,
              source_nb);

  cudaDeviceSynchronize();
  timePoint(t35);
  #if verbose_enabled
  printf("source rep first part time %f\n", timeLen(t3, t35));
  #endif
 
  Norm<<<(srep_nb + 255) / 256, 256>>>(sreps_dev, srepNorm_dev, srep_nb, dim);
  dim3 grid2D_s((source_nb + 15) / 16, (srep_nb + 15) / 16, 1);
  AddAll<<<grid2D_s, block2D>>>(sourceNorm_dev, srepNorm_dev, source2reps_dev,
                                source_nb, srep_nb);

  status = cudaMalloc((void **)&s2rep_dev, source_nb * sizeof(P2R));
  check(status, "Malloc s2rep failed\n");
  findTCluster<<<(source_nb + 255) / 256, 256>>>(
    source2reps_dev, s2rep_dev, source_nb, srep_nb, rep2s_static_dev);
  int *srepsID;
  status = cudaMalloc((void **)&srepsID, srep_nb * sizeof(int));
  check(status, "cMalloc 17 failed\n");

  cudaMemset(srepsID, 0, srep_nb * sizeof(int));
  cudaMemcpy(rep2s_static, rep2s_static_dev, srep_nb * sizeof(R2all_static_dev),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < srep_nb; i++) {
    status = cudaMalloc((void **)&rep2s_dyn_p[i].sortedmembers,
                        rep2s_static[i].npoints * sizeof(R2all_dyn_p));
    check(status, "cMalloc 18 failed\n");
  }
  status = cudaMalloc((void **)&rep2s_dyn_p_dev, srep_nb * sizeof(R2all_dyn_p));
  check(status, "cMalloc 19 failed\n");

  cudaMemcpy(rep2s_dyn_p_dev, rep2s_dyn_p, srep_nb * sizeof(R2all_dyn_p),
             cudaMemcpyHostToDevice);
  fillTMembers<<<(source_nb + 255) / 256, 256>>>(s2rep_dev, source_nb, srepsID,
                                                 rep2s_dyn_p_dev);
  timePoint(t3);
#pragma omp parallel for
  for (int i = 0; i < srep_nb; i++) {
    if (rep2s_static[i].npoints > 0) {
      vector<IndexDist> temp;
      temp.resize(rep2s_static[i].npoints);
      cudaMemcpy(&temp[0], rep2s_dyn_p[i].sortedmembers,
                 rep2s_static[i].npoints * sizeof(IndexDist),
                 cudaMemcpyDeviceToHost);
      sort(temp.begin(), temp.end(), sort_inc());
      cudaMemcpy(rep2s_dyn_p[i].sortedmembers, &temp[0],
                 rep2s_static[i].npoints * sizeof(IndexDist),
                 cudaMemcpyHostToDevice);
    }
  }

  timePoint(t4);
  cudaFree(query2reps_dev);
  status =
    cudaMalloc((void **)&query2reps_dev, query_nb * srep_nb * sizeof(float));
  check(status, "cMalloc 20 failed\n");

  dim3 grid2D_qsrep((query_nb + 15) / 16, (srep_nb + 15) / 16, 1);

  cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, query_nb, srep_nb, dim,
              &alpha, queries_dev, dim, sreps_dev, dim, &beta, query2reps_dev,
              query_nb);
  AddAll<<<grid2D_qsrep, block2D>>>(queryNorm_dev, srepNorm_dev, query2reps_dev,
                                    query_nb, srep_nb);
  #if verbose_enabled
  printf("source rep time %f\n", timeLen(t3, t4));
  #endif
                                   
}

void AllocateAndCopyH2D(R2all_static_dev *&rep2q_static_dev,
                        R2all_static_dev *&rep2s_static_dev,
                        R2all_static *&rep2q_static,
                        R2all_static *&rep2s_static) {
  cudaError_t status;
  status =
    cudaMemcpy(rep2q_static_dev, rep2q_static,
               qrep_nb * sizeof(R2all_static_dev), cudaMemcpyHostToDevice);
  check(status, "Memcpy rep2qs_static failed\n");

  status =
    cudaMemcpy(rep2s_static_dev, rep2s_static,
               srep_nb * sizeof(R2all_static_dev), cudaMemcpyHostToDevice);
  check(status, "Memcpy rep2qs_static failed\n");
  #if verbose_enabled
  printf("sizeof static static_dev %zu %zu\n", sizeof(R2all_static),
         sizeof(R2all_static_dev));
  #endif

}

__global__ void RepsUpperBound(float *qreps_dev, float *sreps_dev,
                               float *maxquery_dev,
                               R2all_static_dev *rep2q_static_dev,
                               R2all_dyn_p *rep2q_dyn_p_dev,
                               R2all_static_dev *rep2s_static_dev,
                               R2all_dyn_p *rep2s_dyn_p_dev, int qrep_nb,
                               int srep_nb, int dim, int K) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < qrep_nb) {
    int UBoundCount = 0;
    for (int i = 0; i < srep_nb; i++) {
      float rep2rep =
        Edistance_128(qreps_dev + tid * dim, sreps_dev + i * dim, dim);
      int count = 0;
      while (count < K && count < rep2s_static_dev[i].npoints) {
        float g2pUBound = maxquery_dev[tid] + rep2rep +
                          rep2s_dyn_p_dev[i].sortedmembers[count].dist;

        if (UBoundCount < K) {
          rep2q_dyn_p_dev[tid].kubound[UBoundCount] = g2pUBound;
          if (rep2q_static_dev[tid].kuboundMax < g2pUBound)
            rep2q_static_dev[tid].kuboundMax = g2pUBound;

          UBoundCount++;
        } else {
          if (rep2q_static_dev[tid].kuboundMax > g2pUBound) {
            float max_local = 0.0f;
            for (int j = 0; j < K; j++) {
              if (rep2q_dyn_p_dev[tid].kubound[j] ==
                  rep2q_static_dev[tid].kuboundMax) {
                rep2q_dyn_p_dev[tid].kubound[j] = g2pUBound;
              }
              if (max_local < rep2q_dyn_p_dev[tid].kubound[j]) {
                max_local = rep2q_dyn_p_dev[tid].kubound[j];
              }
            }
            rep2q_static_dev[tid].kuboundMax = max_local;
          }
        }
        count++;
      }
    }
  }
}

__global__ void FilterReps(float *qreps_dev, float *sreps_dev,
                           float *maxquery_dev,
                           R2all_static_dev *rep2q_static_dev,
                           R2all_dyn_p *rep2q_dyn_p_dev,
                           R2all_static_dev *rep2s_static_dev, int qrep_nb,
                           int srep_nb, int dim, int K) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy =
    threadIdx.y + blockIdx.y * blockDim.y;  //calculate reps[tidy].replist;
  if (tidx < srep_nb && tidy < qrep_nb) {
    float distance =
      Edistance(qreps_dev + tidy * dim, sreps_dev + tidx * dim, dim);
    if (distance - maxquery_dev[tidy] - rep2s_static_dev[tidx].maxdist <
        rep2q_static_dev[tidy].kuboundMax) {
      int rep_id = atomicAdd(&rep2q_static_dev[tidy].noreps, 1);
      rep2q_dyn_p_dev[tidy].replist[rep_id].index = tidx;
      rep2q_dyn_p_dev[tidy].replist[rep_id].dist = distance;
#if debug
      printf("tidy = %d tidx = %d distance = %.10f\n", tidy, tidx, distance);
#endif
    }
  }
}

__device__ int Total = 0;
__global__ void printTotal() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) printf("Total %d\n", Total);
}

__global__ void KNNQuery_base(
  float *queries_dev, float *sources_dev, float *query2reps_dev, P2R *q2rep_dev,
  R2all_static_dev *rep2q_static_dev, R2all_dyn_p *rep2q_dyn_p_dev,
  R2all_static_dev *rep2s_static_dev, R2all_dyn_p *rep2s_dyn_p_dev,
  int query_nb, int dim, int K, IndexDist *knearest1, int *reorder_members) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < query_nb) {
    tid = reorder_members[tid];
    int repIndex = q2rep_dev[tid].repIndex;
    float theta = rep2q_static_dev[repIndex].kuboundMax;
    int Kcount = 0;
    int count = 0;

    IndexDist knearest[1000];
    for (int i = 0; i < rep2q_static_dev[repIndex].noreps; i++) {
      int minlb_rid = rep2q_dyn_p_dev[repIndex].replist[i].index;
      float query2rep = 0.0f;
      query2rep = query2reps_dev[tid + minlb_rid * query_nb];

      for (int j = rep2s_static_dev[minlb_rid].npoints - 1; j >= 0; j--) {
        IndexDist sourcej = rep2s_dyn_p_dev[minlb_rid].sortedmembers[j];
#if debug
        if (tid == 0) printf("j %d %.10f\n", sourcej.index, sourcej.dist);
#endif

        float p2plbound = query2rep - sourcej.dist;
        if (p2plbound > theta)
          break;
        else if (p2plbound < theta * (-1.0f))
          continue;
        else if (p2plbound <= theta && p2plbound >= theta * (-1.0f)) {
          float query2source = Edistance_128(
            queries_dev + tid * dim, sources_dev + sourcej.index * dim, dim);
          count++;
          atomicAdd(&Total, 1);

#if debug
          if (tid == 0) {
            printf("query2source %.10f %.10f %.10f\n", query2source, p2plbound,
                   theta);
          }
#endif

          int insert = -1;
          for (int kk = 0; kk < Kcount; kk++) {
            if (query2source < knearest[kk].dist) {
              insert = kk;
              break;
            }
          }
          if (Kcount < K) {
            if (insert == -1) {
              knearest[Kcount] = {sourcej.index, query2source};
            } else {
              for (int move = Kcount - 1; move >= insert; move--) {
                knearest[move + 1] = knearest[move];
              }
              knearest[insert] = {sourcej.index, query2source};
            }
            Kcount++;
          } else {  //Kcount = K
            if (insert == -1)
              continue;
            else {
              for (int move = K - 2; move >= insert; move--) {
                knearest[move + 1] = knearest[move];
              }

              knearest[insert] = {sourcej.index, query2source};
              theta = knearest[K - 1].dist;
            }
          }
        }
      }
    }
    memcpy(&knearest1[tid * K], knearest, K * sizeof(IndexDist));
  }
}
__global__ void KNNQuery_theta(P2R *q2rep_dev,
                               R2all_static_dev *rep2q_static_dev, int query_nb,
                               float *thetas) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < query_nb) {
    int repIndex = q2rep_dev[tid].repIndex;
    thetas[tid] = rep2q_static_dev[repIndex].kuboundMax;
  }
}

__global__ void KNNQuery(float *queries_dev, float *sources_dev,
                         float *sreps_dev, P2R *q2rep_dev,
                         R2all_static_dev *rep2q_static_dev,
                         R2all_dyn_p *rep2q_dyn_p_dev,
                         R2all_static_dev *rep2s_static_dev,
                         R2all_dyn_p *rep2s_dyn_p_dev, int query_nb, int dim,
                         int K, IndexDist *knearest, float *thetas, int tpq,
                         int *reorder_members) {
  int ttid = threadIdx.x + blockIdx.x * blockDim.x;
  int tp = ttid % tpq;
  int tid = ttid / tpq;
  if (tid < query_nb) {
    tid = reorder_members[tid];
    ttid = tid * tpq + tp;
    int repIndex = q2rep_dev[tid].repIndex;
    int Kcount = 0;
    int count = 0;

    for (int i = 0; i < rep2q_static_dev[repIndex].noreps; i++) {
      int minlb_rid = rep2q_dyn_p_dev[repIndex].replist[i].index;
      float query2rep = 0.0f;
      query2rep =  
        Edistance_128(queries_dev + tid * dim, sreps_dev + minlb_rid * dim,
                      dim);

      for (int j = rep2s_static_dev[minlb_rid].npoints - 1 - tp; j >= 0;
           j -= tpq) {
        IndexDist sourcej = rep2s_dyn_p_dev[minlb_rid].sortedmembers[j];
#if debug
        if (tid == 0) printf("j %d %.10f\n", sourcej.index, sourcej.dist);
#endif
        float p2plbound = query2rep - sourcej.dist;
        if (p2plbound > *(volatile float *)&thetas[tid])
          break;
        else if (p2plbound < *(volatile float *)&thetas[tid] * (-1.0f))
          continue;
        else if (p2plbound <= *(volatile float *)&thetas[tid] &&
                 p2plbound >= *(volatile float *)&thetas[tid] * (-1.0f)) {
          float query2source = Edistance_128(
            queries_dev + tid * dim, sources_dev + sourcej.index * dim, dim);
          count++;
          atomicAdd(&Total, 1);

          int insert = -1;
          for (int kk = 0; kk < Kcount; kk++) {
            if (query2source < knearest[ttid * K + kk].dist) {
              insert = kk;
              break;
            }
          }
          if (Kcount < K) {
            if (insert == -1) {
              knearest[ttid * K + Kcount] = {sourcej.index, query2source};
            } else {
              for (int move = Kcount - 1; move >= insert; move--) {
                knearest[ttid * K + move + 1] = knearest[ttid * K + move];
              }
              knearest[ttid * K + insert] = {sourcej.index, query2source};
            }
            Kcount++;
          } else {  //Kcount = K
            if (insert == -1)
              continue;
            else {
              for (int move = K - 2; move >= insert; move--) {
                knearest[ttid * K + move + 1] = knearest[ttid * K + move];
              }

              knearest[ttid * K + insert] = {sourcej.index, query2source};
              atomicMin_float(&thetas[tid], knearest[ttid * K + K - 1].dist);
            }
          }
        }
      }
    }
  }
}

__global__ void final(int k, IndexDist *knearest, int tpq, int query_nb,
                      IndexDist *final_knearest, int *tag_base) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int *tag = tid * tpq + tag_base;
  if (tid < query_nb) {
    for (int i = 0; i < k; i++) {
      float min = knearest[tid * tpq * k + tag[0]].dist;
      int index = 0;
      for (int j = 1; j < tpq; j++) {
        float value = knearest[(tid * tpq + j) * k + tag[j]].dist;
        if (min > value) {
          min = value;
          index = j;
        }
      }
      final_knearest[tid * k + i] =
        knearest[(tid * tpq + index) * k + tag[index]];
      tag[index]++;
    }
  }
}

void *work(void *para) { cudaFree(0); }

/**
   * @brief Flat C++ API function to perform a sweet knn on
   * a series of input arrays and combine the results into a single
   * output array for indexes and distances.
   *
   * @param D the dimensionality of the arrays
   * @param search_items array of items to search of dimensionality D
   * @param n number of rows in search_items
   * @param res_I the resulting index array of size n * k
   * @param res_D the resulting distance array of size n * k
   * @param k the number of nearest neighbors to return
   */
void sweet_knn(int D, float *search_items, int n, long *res_I, float *res_D,
               int k) {

  pthread_t thread2;
  timePoint(t1);
  int rc = pthread_create(&thread2, NULL, work, NULL);

  query_nb = n;   
  source_nb = n;  
  dim = D;
  qrep_nb = (int)(3 * std::sqrt(query_nb));   //empirically found
  srep_nb = (int)(3 * std::sqrt(source_nb));  //empirically found
  K = k;

  sources = (float *)malloc(source_nb * dim * sizeof(float));
  queries = (float *)malloc(query_nb * dim * sizeof(float));

  // setup source and query points
  cudaMemcpy(sources, search_items, source_nb * dim * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(queries, search_items, query_nb * dim * sizeof(float),
             cudaMemcpyDeviceToHost);

  R2all_static *rep2q_static =
    (R2all_static *)malloc(qrep_nb * sizeof(R2all_static));

  // initializing memory pointed by rep2q_static
  for (int i = 0; i < qrep_nb; i++) {
    rep2q_static[i].maxdist = 0.0f;
    rep2q_static[i].mindist = FLT_MAX;
    rep2q_static[i].npoints = 0;
    rep2q_static[i].noreps = 0;
    rep2q_static[i].kuboundMax = 0.0f;
  }

  R2all_static *rep2s_static =
    (R2all_static *)malloc(srep_nb * sizeof(R2all_static));

  // initializing memory pointed by rep2s_static
  for (int i = 0; i < srep_nb; i++) {
    rep2s_static[i].maxdist = 0.0f;
    rep2s_static[i].mindist = FLT_MAX;
    rep2s_static[i].npoints = 0;
    rep2s_static[i].noreps = 0;
    rep2s_static[i].kuboundMax = 0.0f;
  }

  float *queries_dev, *sources_dev, *qreps_dev, *sreps_dev;
  P2R *q2rep_dev, *s2rep_dev;
  R2all_static_dev *rep2q_static_dev;
  R2all_dyn_p *rep2q_dyn_p_dev;
  R2all_static_dev *rep2s_static_dev;
  R2all_dyn_p *rep2s_dyn_p_dev;
  float *query2reps_dev;
  float *maxquery_dev;

  int *reorder_members;

  R2all_dyn_p *rep2q_dyn_p =
    (R2all_dyn_p *)malloc(qrep_nb * sizeof(R2all_dyn_p));
  R2all_dyn_p *rep2s_dyn_p =
    (R2all_dyn_p *)malloc(srep_nb * sizeof(R2all_dyn_p));

  timePoint(t1);
  cudaError_t status;
  status =
    cudaMalloc((void **)&query2reps_dev, qrep_nb * query_nb * sizeof(float));
  check(status, "cMalloc 21 failed\n");
  timePoint(t2);
  #if verbose_enabled
  printf("cudaFree time %f\n", timeLen(t1, t2));
  #endif

  //cluster queries and sources to reps
  clusterReps(queries_dev, sources_dev, qreps_dev, sreps_dev, maxquery_dev,
              q2rep_dev, s2rep_dev, rep2q_static_dev, rep2s_static_dev,
              rep2q_dyn_p_dev, rep2s_dyn_p_dev, query2reps_dev, rep2q_static,
              rep2s_static, rep2q_dyn_p, rep2s_dyn_p, reorder_members);

  //tranfer data structures to GPU.
  AllocateAndCopyH2D(rep2q_static_dev, rep2s_static_dev, rep2q_static,
                     rep2s_static);
  timePoint(t2);

  #if verbose_enabled
  printf("prepo time %f\n", timeLen(t1, t2));
  #endif
 

  if (cudaGetLastError() != cudaSuccess) cout << "error 16" << endl;

  //Kernel 1: upperbound for each rep
  RepsUpperBound<<<(qrep_nb + 255) / 256, 256>>>(
    qreps_dev, sreps_dev, maxquery_dev, rep2q_static_dev, rep2q_dyn_p_dev,
    rep2s_static_dev, rep2s_dyn_p_dev, qrep_nb, srep_nb, dim, K);

  if (cudaGetLastError() != cudaSuccess)
    cout << "Kernel RepsUpperBound failed" << endl;

  //Kernel 2: filter reps based on upperbound and lowerbound;
  dim3 block(16, 16, 1);
  dim3 grid((srep_nb + block.x - 1) / block.x,
            (qrep_nb + block.y - 1) / block.y, 1);
  FilterReps<<<grid, block>>>(qreps_dev, sreps_dev, maxquery_dev,
                              rep2q_static_dev, rep2q_dyn_p_dev,
                              rep2s_static_dev, qrep_nb, srep_nb, dim, K);
  struct timespec sort_start, sort_end;
  timePoint(sort_start);
  cudaMemcpy(rep2q_static, rep2q_static_dev, qrep_nb * sizeof(R2all_static_dev),
             cudaMemcpyDeviceToHost);

#pragma omp parallel for
  for (int i = 0; i < qrep_nb; i++) {
    vector<IndexDist> temp;
    temp.resize(rep2q_static[i].noreps);
    cudaMemcpy(&temp[0], rep2q_dyn_p[i].replist,
               rep2q_static[i].noreps * sizeof(IndexDist),
               cudaMemcpyDeviceToHost);
    sort(temp.begin(), temp.end(), sort_inc());

    cudaMemcpy(rep2q_dyn_p[i].replist, &temp[0],
               rep2q_static[i].noreps * sizeof(IndexDist),
               cudaMemcpyHostToDevice);
  }

  timePoint(sort_end);
  #if verbose_enabled
  printf("sort query replist time %f\n", timeLen(sort_start, sort_end));
  #endif
 

  //Kernel 3: knn for each point
  IndexDist *knearest, *final_knearest;
  int tpq = (2048 * 13) / query_nb;
  IndexDist *knearest_h = (IndexDist *)malloc(query_nb * K * sizeof(IndexDist));

  status = cudaMalloc((void **)&knearest,
                      query_nb * (tpq + 1) * K * sizeof(IndexDist));
  check(status, "cMalloc 22 failed\n");

  if (tpq > 1) {
    float *theta;

    status = cudaMalloc((void **)&theta, query_nb * sizeof(float));
    check(status, "cMalloc 23 failed\n");

    KNNQuery_theta<<<(query_nb + 255) / 256, 256>>>(q2rep_dev, rep2q_static_dev,
                                                    query_nb, theta);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    KNNQuery<<<(tpq * query_nb + 255) / 256, 256>>>(
      queries_dev, sources_dev, sreps_dev, q2rep_dev, rep2q_static_dev,
      rep2q_dyn_p_dev, rep2s_static_dev, rep2s_dyn_p_dev, query_nb, dim, K,
      knearest, theta, tpq, reorder_members);
    final_knearest = knearest + query_nb * tpq * K;

    int *tag_base;

    status = cudaMalloc((void **)&tag_base, tpq * query_nb * sizeof(int));
    check(status, "cMalloc 24 failed\n");

    cudaMemset(tag_base, 0, tpq * query_nb * sizeof(int));
    final<<<(query_nb + 255) / 256, 256>>>(K, knearest, tpq, query_nb,
                                           final_knearest, tag_base);
  } else {
    KNNQuery_base<<<(query_nb + 255) / 256, 256>>>(
      queries_dev, sources_dev, query2reps_dev, q2rep_dev, rep2q_static_dev,
      rep2q_dyn_p_dev, rep2s_static_dev, rep2s_dyn_p_dev, query_nb, dim, K,
      knearest, reorder_members);
  }
  cudaDeviceSynchronize();
  timePoint(t2);
  #if verbose_enabled
  printf("total time %f\n", timeLen(t1, t2));
  #endif
 
  #if verbose_enabled
  printTotal<<<1, 1>>>();
  #endif
 
  if (tpq > 1) {
    cudaMemcpy(knearest_h, final_knearest, query_nb * K * sizeof(IndexDist),
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(knearest_h, knearest, query_nb * K * sizeof(IndexDist),
               cudaMemcpyDeviceToHost);
  }
  // print top k neighbors for i = 100
  #if verbose_enabled
  int i = 100;
  for (int j = 0; j < K; j++)
    printf("i,k %d %d  %d %f\n", i, j, knearest_h[i * K + j].index,
           knearest_h[i * K + j].dist);
  #endif

  // store resulting indices and distances into result arrays
  // this step is highly inefficient and must be fixed - FIXME
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < K; j++) {
      cudaMemcpy(res_I + (i * K + j), knearest_h + (i * K + j), sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(res_D + (i * K + j),
                 (char *)(knearest_h + (i * K + j)) + sizeof(int),
                 sizeof(float), cudaMemcpyHostToDevice);
    }
  }
  cudaDeviceSynchronize();

  free(queries);
  free(sources);
  free(rep2q_static);
  free(rep2s_static);
  return;
}

/**
   * @brief A flat C++ API function that chunks a host array up into
   * some number of different devices
   *
   * @param ptr an array on host to chunk
   * @param n number of rows in host array
   * @param D number of cols in host array
   * @param devices array of devices to use
   * @param output an array of output pointers to allocate and use
   * @param sizes output array sizes
   * @param n_chunks number of chunks to spread across device arrays
   */
void chunk_host_array(cumlHandle &handle, const float *ptr, int n, int D,
                      int *devices, float **output, int *sizes, int n_chunks) {
  chunk_to_device<float, int>(ptr, n, D, devices, output, sizes, n_chunks,
                              handle.getImpl().getStream());
}

/**
	 * Build a kNN object for training and querying a k-nearest neighbors model.
	 * @param D 	number of features in each vector
	 */
kNN::kNN(const cumlHandle &handle, int D, bool verbose)
  : D(D), total_n(0), indices(0), verbose(verbose), owner(false) {
  this->handle = const_cast<cumlHandle *>(&handle);
  sizes = nullptr;
  ptrs = nullptr;
}

kNN::~kNN() {
  try {
    if (this->owner) {
      if (this->verbose) std::cout << "Freeing kNN memory" << std::endl;
      for (int i = 0; i < this->indices; i++) {
        CUDA_CHECK(cudaFree(this->ptrs[i]));
      }
    }

  } catch (const std::exception &e) {
    std::cout << "An exception occurred releasing kNN memory: " << e.what()
              << std::endl;
  }

  delete ptrs;
  delete sizes;
}

void kNN::reset() {
  if (this->indices > 0) {
    this->indices = 0;
    this->total_n = 0;
  }
}

/**
	 * Fit a kNN model by creating separate indices for multiple given
	 * instances of kNNParams.
	 * @param input  an array of pointers to data on (possibly different) devices
	 * @param N 	 number of items in input array.
	 */
void kNN::fit(float **input, int *sizes, int N) {
  if (this->owner)
    for (int i = 0; i < this->indices; i++) {
      CUDA_CHECK(cudaFree(this->ptrs[i]));
    }

  if (this->verbose) std::cout << "N=" << N << std::endl;

  reset();

  this->indices = N;
  this->ptrs = (float **)malloc(N * sizeof(float *));
  this->sizes = (int *)malloc(N * sizeof(int));

  for (int i = 0; i < N; i++) {
    this->ptrs[i] = input[i];
    this->sizes[i] = sizes[i];
  }
}

/**
	 * Search the kNN for the k-nearest neighbors of a set of query vectors
	 * @param search_items set of vectors to query for neighbors
	 * @param n 		   number of items in search_items
	 * @param res_I 	   pointer to device memory for returning k nearest indices
	 * @param res_D		   pointer to device memory for returning k nearest distances
	 * @param k			   number of neighbors to query
	 */
void kNN::search(float *search_items, int n, long *res_I, float *res_D, int k) {
  MLCommon::Selection::brute_force_knn(ptrs, sizes, indices, D, search_items, n,
                                       res_I, res_D, k,
                                       handle->getImpl().getStream());
}

/**
     * Chunk a host array up into one or many GPUs (determined by the provided
     * list of gpu ids) and fit a knn model.
     *
     * @param ptr       an array in host memory to chunk over devices
     * @param n         number of elements in ptr
     * @param devices   array of device ids for chunking the ptr
     * @param n_chunks  number of elements in gpus
     * @param out       host pointer (size n) to store output
     */
void kNN::fit_from_host(float *ptr, int n, int *devices, int n_chunks) {
  if (this->owner)
    for (int i = 0; i < this->indices; i++) {
      CUDA_CHECK(cudaFree(this->ptrs[i]));
    }

  reset();

  this->owner = true;

  float **params = new float *[n_chunks];
  int *sizes = new int[n_chunks];

  chunk_to_device<float>(ptr, n, D, devices, params, sizes, n_chunks,
                         handle->getImpl().getStream());

  fit(params, sizes, n_chunks);
}
};  // namespace ML

/**
 * @brief Flat C API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param handle the cuml handle to use
 * @param input an array of pointers to the input arrays
 * @param sizes an array of sizes of input arrays
 * @param n_params array size of input and sizes
 * @param D the dimensionality of the arrays
 * @param search_items array of items to search of dimensionality D
 * @param n number of rows in search_items
 * @param res_I the resulting index array of size n * k
 * @param res_D the resulting distance array of size n * k
 * @param k the number of nearest neighbors to return
 */
extern "C" cumlError_t knn_search(const cumlHandle_t handle, float **input,
                                  int *sizes, int n_params, int D,
                                  float *search_items, int n, long *res_I,
                                  float *res_D, int k) {
  cumlError_t status;

  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      MLCommon::Selection::brute_force_knn(input, sizes, n_params, D,
                                           search_items, n, res_I, res_D, k,
                                           handle_ptr->getImpl().getStream());
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}

/**
 * @brief A flat C api function that chunks a host array up into
 * some number of different devices
 *
 * @param ptr an array on host to chunk
 * @param n number of rows in host array
 * @param D number of cols in host array
 * @param devices array of devices to use
 * @param output an array of output pointers to allocate and use
 * @param sizes output array sizes
 * @param n_chunks number of chunks to spread across device arrays
 */
extern "C" cumlError_t chunk_host_array(const cumlHandle_t handle,
                                        const float *ptr, int n, int D,
                                        int *devices, float **output,
                                        int *sizes, int n_chunks) {
  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if (status == CUML_SUCCESS) {
    try {
      ML::chunk_to_device<float, int>(ptr, n, D, devices, output, sizes,
                                      n_chunks,
                                      handle_ptr->getImpl().getStream());
    } catch (...) {
      status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}