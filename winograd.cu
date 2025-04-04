#include <cublas_v2.h>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <iostream>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "utils.h"

void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ swapped_V,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_h][ti.tile_in_w];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*swapped_V_tensor_t)[ti.tile_in_h][ti.tile_in_w];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;
  swapped_V_tensor_t swapped_V_tensor = (swapped_V_tensor_t)swapped_V;

  float z0, z1, z2, z3, z4, z5, z6;
  #pragma omp parallel for
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    //std::cout<<omp_get_num_threads()<<std::endl;
    // #pragma omp parallel for
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      z6 = packed_image_tensor[idx][0][w];

      z0 = 4.0f * z6;

      z6 = packed_image_tensor[idx][1][w];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = packed_image_tensor[idx][2][w];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = packed_image_tensor[idx][3][w];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = packed_image_tensor[idx][4][w];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = packed_image_tensor[idx][5][w];

      z5 += z6;

      swapped_V_tensor[idx][0][w] = z0;
      swapped_V_tensor[idx][1][w] = z1;
      swapped_V_tensor[idx][2][w] = z2;
      swapped_V_tensor[idx][3][w] = z3;
      swapped_V_tensor[idx][4][w] = z4;
      swapped_V_tensor[idx][5][w] = z5;
    }
    // #pragma omp parallel for
    for (int64_t h = 0; h < ti.tile_in_h; ++h) {
      z6 = swapped_V_tensor[idx][h][0];

      z0 = 4.0f * z6;

      z6 = swapped_V_tensor[idx][h][1];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = swapped_V_tensor[idx][h][2];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = swapped_V_tensor[idx][h][3];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = swapped_V_tensor[idx][h][4];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = swapped_V_tensor[idx][h][5];

      z5 += z6;

      V_tensor[h][0][idx] = z0;
      V_tensor[h][1][idx] = z1;
      V_tensor[h][2][idx] = z2;
      V_tensor[h][3][idx] = z3;
      V_tensor[h][4][idx] = z4;
      V_tensor[h][5][idx] = z5;
    }
  }
}

void filter_transform(float *__restrict__ filter,
                      float *__restrict__ swapped_U,
                      float *__restrict__ U,
                      const filter_shape_t fs,
                      const U_shape_t us,
                      const int64_t collapsed_dim_size) {
  typedef float(*filter_tensor_t)[fs.h][fs.w];
  typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
  typedef float(*swapped_U_tensor_t)[us.h][us.w];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  U_tensor_t U_tensor = (U_tensor_t)U;
  swapped_U_tensor_t swapped_U_tensor = (swapped_U_tensor_t)swapped_U;

  float z0, z1, z2, z3, z4, z5, z6;
  #pragma omp parallel for
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    // #pragma omp parallel for
    for (int64_t w = 0; w < fs.w; ++w) {
      z6 = filter_tensor[idx][0][w];


      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = filter_tensor[idx][1][w];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = filter_tensor[idx][2][w];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      swapped_U_tensor[idx][0][w] = z0;
      swapped_U_tensor[idx][1][w] = z1;
      swapped_U_tensor[idx][2][w] = z2;
      swapped_U_tensor[idx][3][w] = z3;
      swapped_U_tensor[idx][4][w] = z4;
      swapped_U_tensor[idx][5][w] = z5;
    }
    // #pragma omp parallel for
    for (int64_t h = 0; h < us.h; ++h) {
      z6 = swapped_U_tensor[idx][h][0];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = swapped_U_tensor[idx][h][1];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = swapped_U_tensor[idx][h][2];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[h][0][idx] = z0;
      U_tensor[h][1][idx] = z1;
      U_tensor[h][2][idx] = z2;
      U_tensor[h][3][idx] = z3;
      U_tensor[h][4][idx] = z4;
      U_tensor[h][5][idx] = z5;
    }
  }
}

void output_transform(float *__restrict__ swapped_M,
                      float *__restrict__ M,
                      float *__restrict__ swapped_Y,
                      float *__restrict__ Y,
                      const tiling_info_t ti,
                      const int64_t collapsed_dim_size) {

  typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*swapped_M_tensor_t)[ti.tile_in_h][ti.tile_in_w];
  typedef float(*swapped_Y_tensor_t)[ti.tile_out_h][ti.tile_in_w];
  M_tensor_t M_tensor = (M_tensor_t)M;
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  swapped_M_tensor_t swapped_M_tensor = (swapped_M_tensor_t)swapped_M;
  swapped_Y_tensor_t swapped_Y_tensor = (swapped_Y_tensor_t)swapped_Y;
  
  #pragma omp parallel for
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    for (int64_t w = 0; w < 6; ++w) {
      swapped_M_tensor[idx][0][w] = M_tensor[0][w][idx];
      swapped_M_tensor[idx][1][w] = M_tensor[1][w][idx];
      swapped_M_tensor[idx][2][w] = M_tensor[2][w][idx];
      swapped_M_tensor[idx][3][w] = M_tensor[3][w][idx];
      swapped_M_tensor[idx][4][w] = M_tensor[4][w][idx];
      swapped_M_tensor[idx][5][w] = M_tensor[5][w][idx];
    }
  }

  float z0, z1, z2, z3, z4;
  #pragma omp parallel for
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    // #pragma omp parallel for
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      z4 = swapped_M_tensor[idx][0][w];
      z0 = z4;

      z4 = swapped_M_tensor[idx][1][w];
      z0 = z0 + z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = swapped_M_tensor[idx][2][w];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = swapped_M_tensor[idx][3][w];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = swapped_M_tensor[idx][4][w];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = swapped_M_tensor[idx][5][w];
      z3 += z4;

      swapped_Y_tensor[idx][0][w] = z0;
      swapped_Y_tensor[idx][1][w] = z1;
      swapped_Y_tensor[idx][2][w] = z2;
      swapped_Y_tensor[idx][3][w] = z3;
    }
    // #pragma omp parallel for
    for (int64_t h = 0; h < ti.tile_out_h; ++h) {
      z4 = swapped_Y_tensor[idx][h][0];

      z0 = z4;

      z4 = swapped_Y_tensor[idx][h][1];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = swapped_Y_tensor[idx][h][2];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = swapped_Y_tensor[idx][h][3];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = swapped_Y_tensor[idx][h][4];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = swapped_Y_tensor[idx][h][5];

      z3 += z4;

      swapped_Y_tensor[idx][h][0] = z0;
      swapped_Y_tensor[idx][h][1] = z1;
      swapped_Y_tensor[idx][h][2] = z2;
      swapped_Y_tensor[idx][h][3] = z3;
    }
  }
}

void image_packing(float *__restrict__ image,
                   float *__restrict__ packed_image,
                   const image_shape_t is,
                   const tiling_info_t ti) {
  typedef float(*packedImage_tensor_t)[is.ic][ti.tile_in_h][ti.tile_in_w];
  typedef float(*image_tensor_t)[is.ic][is.h][is.w];
  packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
  image_tensor_t image_tensor = (image_tensor_t)image;
  #pragma omp parallel for
  for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
    tile_index_t tidx = get_tile_index(tile, ti);
    int64_t batch = tidx.b, ww = tidx.tw<<2, hh = tidx.th<<2;
    int64_t hed = MIN(ti.tile_in_h, is.h - hh), wed = MIN(ti.tile_in_w, is.w - ww);
    // #pragma omp parallel for collapse(3)
    for (int64_t ic = 0; ic < is.ic; ic++) {
      for (int64_t h = 0; h < hed; ++h) {
        for (int64_t w = 0; w < wed; ++w) {
            packed_image_tensor[tile][ic][h][w] = image_tensor[batch][ic][h+hh][w+ww];
        }
      }
    }
  }
}

void output_unpacking_store(float *__restrict__ swapped_Y,
                            float *__restrict__ out,
                            const out_shape_t os,
                            const tiling_info_t ti) {
  typedef float(*swapped_Y_tensor_t)[ti.num_tiles][ti.tile_out_h][ti.tile_in_w];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  swapped_Y_tensor_t swapped_Y_tensor = (swapped_Y_tensor_t)swapped_Y;
  out_tensor_t out_tensor = (out_tensor_t)out;
  #pragma omp parallel for
  for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
    tile_index_t tidx = get_tile_index(tile, ti);
    int64_t batch = tidx.b, ww = tidx.tw<<2, hh = tidx.th<<2;
    int64_t hed = MIN(ti.tile_out_h, os.h - hh), wed = MIN(ti.tile_in_w, os.w - ww);
    // #pragma omp parallel for collapse(3)
    for (int64_t oc = 0; oc < os.oc; oc++) {
      for (int64_t h = 0; h < hed; ++h) {
        for (int64_t w = 0; w < wed; ++w) {
            out_tensor[batch][oc][h+hh][w+ww] = swapped_Y_tensor[oc][tile][h][w];
        }
      }
    }
  }
}

void sgemm(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C, const int64_t BatchCount) {

  // 创建cuBLAS句柄?
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS初始化失败\n");
    return;
  }

  // 分配GPU内存
  float *d_A, *d_B, *d_C;
  cudaError_t cudaStatus;
  cudaStatus = cudaMalloc((void **)&d_A, sizeof(float) * BatchCount * M * K);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc失败: %s\n", cudaGetErrorString(cudaStatus));
    cublasDestroy(handle);
    return;
  }
  cudaStatus = cudaMalloc((void **)&d_B, sizeof(float) * BatchCount * N * K);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc失败: %s\n", cudaGetErrorString(cudaStatus));
    cudaFree(d_A);
    cublasDestroy(handle);
    return;
  }
  cudaStatus = cudaMalloc((void **)&d_C, sizeof(float) * BatchCount * N * M);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc失败: %s\n", cudaGetErrorString(cudaStatus));
    cudaFree(d_A);
    cudaFree(d_B);
    cublasDestroy(handle);
    return;
  }

  // 复制数据到GPU
  cudaStatus = cudaMemcpy(d_A, A, sizeof(float) * BatchCount * M * K, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy失败: %s\n", cudaGetErrorString(cudaStatus));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return;
  }
  cudaStatus = cudaMemcpy(d_B, B, sizeof(float) * BatchCount * N * K, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy失败: %s\n", cudaGetErrorString(cudaStatus));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return;
  }
  //C n*m = A m*k * B n*k
  //C = B * A^T
  //column-major
  //C m*n = A m*k * B k*n   
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // 计算每个矩阵的步长(stride)
  long long strideA = M * K;
  long long strideB = K * N;
  long long strideC = M * N;
  
  status = cublasSgemmStridedBatched(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             d_A, K, strideA,
                             d_B, K, strideB,
                             &beta,
                             d_C, M, strideC,
                             BatchCount);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasSgemmStridedBatched失败\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return;
  }

  cudaStatus = cudaMemcpy(C, d_C, sizeof(float) * BatchCount * N * M, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy失败: %s\n", cudaGetErrorString(cudaStatus));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return;
  }
  // 清理资源
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);
}

void winograd_convolution(
    float *__restrict__ image,
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter,
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out) {
  
  // // 创建CUDA事件用于GPU操作计时
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // float milliseconds = 0;
  
  // // CPU计时变量
  // auto total_start = std::chrono::high_resolution_clock::now();
  // auto step_start = total_start;
  // auto step_end = total_start;
  
  /* 初始化形状和内存分配 */
  const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
  const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
  const out_shape_t os = get_output_shape(is, fs);
  const tiling_info_t ti = get_tiling_info(is, os);
  const U_shape_t us = get_U_shape(fs, ti);
  const V_shape_t vs = get_V_shape(is, ti);

  float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
  float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
  float *swapped_V = (float *)malloc(sizeof(float) * vs.num_tiles * vs.ic * ti.tile_in_h * ti.tile_in_w);
  float *swapped_U = (float *)malloc(sizeof(float) * us.oc * us.ic * ti.tile_in_h * ti.tile_in_w);
  float *swapped_M = (float *)malloc(sizeof(float) * us.oc * vs.num_tiles * ti.tile_in_h * ti.tile_in_w);
  float *swapped_Y = (float *)malloc(sizeof(float) * os.oc * ti.num_tiles * ti.tile_out_h * ti.tile_in_w);
  float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
  float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
  float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
  float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

  
  // // 1. 单独计时filter_transform
  // step_start = std::chrono::high_resolution_clock::now();
  // filter_transform(filter, swapped_U, U, fs, us, us.oc * us.ic);
  // step_end = std::chrono::high_resolution_clock::now();
  // std::cout << "filter_transform: " 
  //           << std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count() / 1000.0 
  //           << " ms" << std::endl;
  
  // // 2. 单独计时image_packing
  // step_start = std::chrono::high_resolution_clock::now();
  // image_packing(image, packed_image, is, ti);
  // step_end = std::chrono::high_resolution_clock::now();
  // std::cout << "image_packing: " 
  //           << std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count() / 1000.0 
  //           << " ms" << std::endl;
  
  // // 3. 单独计时image_transform
  // step_start = std::chrono::high_resolution_clock::now();
  // image_transform(packed_image, swapped_V, V, vs, ti, vs.ic * vs.num_tiles);
  // step_end = std::chrono::high_resolution_clock::now();
  // std::cout << "image_transform: " 
  //           << std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count() / 1000.0 
  //           << " ms" << std::endl;

  #pragma omp sections
  {
    #pragma omp section
    {
      filter_transform(filter, swapped_U, U, fs, us, us.oc * us.ic);
    }
    #pragma omp section
    {
      image_packing(image, packed_image, is, ti);
      image_transform(packed_image, swapped_V, V, vs, ti, vs.ic * vs.num_tiles);
    }
  }
  
  // // 4. 计时sgemm(含GPU操作) - 保持不变
  // step_start = std::chrono::high_resolution_clock::now();
  // cudaEventRecord(start);
  sgemm(vs.num_tiles, us.oc, us.ic, (float *)(V), (float *)(U), (float *)(M), ti.tile_in_h*ti.tile_in_w);
  // cudaEventRecord(stop);
  // step_end = std::chrono::high_resolution_clock::now();
  
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // std::cout << "sgemm total: " 
  //           << std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count() / 1000.0 
  //           << " ms (GPU kernel: " << milliseconds << " ms)" << std::endl;

  // // 5. 拆分output_transform的统计
  // step_start = std::chrono::high_resolution_clock::now();
  output_transform(swapped_M, M, swapped_Y, Y, ti, us.oc * vs.num_tiles);
  // step_end = std::chrono::high_resolution_clock::now();
  // std::cout << "output_transform: " 
  //           << std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count() / 1000.0 
  //           << " ms" << std::endl;
  
  // // 计时output_unpacking_store
  // step_start = std::chrono::high_resolution_clock::now();
  output_unpacking_store(swapped_Y, out, os, ti);
  // step_end = std::chrono::high_resolution_clock::now();
  // std::cout << "output_unpacking_store: " 
  //           << std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count() / 1000.0 
  //           << " ms" << std::endl;

  // // 释放内存
  // step_start = std::chrono::high_resolution_clock::now();
  free(packed_filter);
  free(packed_image);
  free(swapped_U);
  free(swapped_V);
  free(swapped_M);
  free(swapped_Y);
  free(U);
  free(V);
  free(M);
  free(Y);
  // step_end = std::chrono::high_resolution_clock::now();
  
  // auto total_end = std::chrono::high_resolution_clock::now();
  // double total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() / 1000.0;
  // std::cout << "Memory free: " 
  //           << std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count() / 1000.0 
  //           << " ms" << std::endl;
  // std::cout << "--------------------------------------" << std::endl;
  // std::cout << "Total execution time: " << total_time << " ms" << std::endl;
  
  // // 销毁CUDA事件
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
}