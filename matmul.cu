#include <cstdlib>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void matmul(){}

void init_matrix_squre(int *a, int n){
    for (int i =0; i < n; i++){
        for (int j =0; j < n; j++){
            a[i * n + j] = rand() % 100;
        }
    }
}

void verify_matmul_squre(int *a, int *b, int *c, int N){
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        tmp += a[i * N + k] * b[k * N + j];
      }
      assert(tmp == c[i * N + j]);
    }
  }
}

void vec_add_err_check(int *a, int *b, int *c, int n){
    for (int i =0; i<n; i++){
        assert( c[i] == a[i] + b[i]);
    }
}

void init_vec(int *a, int N){
    for (int i =0; i<N; i++){
        a[i] = rand() % 100;
    }
}


__global__ void matmul_suare_kernel(int *a, int *b, int *c, int N){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}

__global__ void vectorAdd(int *a, int *b, int *c, int n){
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid<n){
        c[tid] = a[tid] + b[tid];
    }
}

void matadd (){
    int N = 1 << 10;
    size_t bytes = N * sizeof(int);
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    // allocat host mem
    h_a = (int *) malloc(bytes);
    h_b = (int *) malloc(bytes);
    h_c = (int *) malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    init_vec(h_a, N);
    init_vec(h_b, N);
    init_vec(h_c, N);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (int) ceil(N / NUM_THREADS);

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    printf("vector addition computed succesfully \n");
}

void matmul_square  (){
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    h_a = (int *) malloc(bytes);
    h_b = (int *) malloc(bytes);
    h_c = (int *) malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    init_matrix_squre(h_a, N);
    init_matrix_squre(h_b, N);
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 16;
    int GRID_SIZE = (int) ceil(N/BLOCK_SIZE);

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    matmul_suare_kernel<<< grid, threads >>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes,cudaMemcpyDeviceToHost);

    verify_matmul_squre(h_a, h_b, h_c, N);

    printf("MATMUL COMPLETED \n");

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
     
}

void verify_mat_cublas(float *a, float *b, float *c, int N){
  float tmp;
  float eps = 0.001;
  for(int i =0; i<N; i++){
    for(int j =0; j<N; j++){
      tmp = 0.0;
      for(int k =0; k<N; k++){
        tmp += a[k * N + i] * b[j * N + k];
      }
      assert(fabs(tmp - c[j * N + i]) < eps);
    }
  }
}


void matmul_cublas(){
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    curandGenerateUniform(prng, d_a, N*N);
    curandGenerateUniform(prng, d_b, N*N);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // scaling
    float alpha = 1.0f;
    float beta = 0.0f;

    // c = alpha*a @ b + c
    // (m,n) = (m,k) * (k,n)
    // handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_a, N, d_b, N, &beta, d_c, N);

    cudaMemcpy(h_a, d_a, bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes,cudaMemcpyDeviceToHost);

    verify_mat_cublas(h_a, h_b, h_c, N);

    printf("CUBLAS MATMUL COMPLETED \n");
}


int main(){
    // int N = 1 << 10;

    // size_t bytes = N * N * sizeof(int);
    // int *a, int *b, int *c;

    // cudaMallocManaged(&a, bytes);
    // cudaMallocManaged(&b, bytes);
    // cudaMallocManaged(&c, bytes);

    // init_matrix(a, N);
    // init_matrix(b, N);

    // int threads = 16;
    // int blocks = (n + threads - 1) / threads;
    // matadd();
    // matmul_square();
    matmul_cublas();
}