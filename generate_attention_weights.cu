#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "linear.h"
#include "attention.h"
#include "attention_cuda.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <omp.h>

template <typename Func>
double benchmark(Func&& fn, int runs = 10) {
    using namespace std::chrono;
    double total_ms = 0;
    for (int i = 0; i < runs; ++i) {
        auto start = high_resolution_clock::now();
        fn();
        cudaDeviceSynchronize(); // Ensure GPU work is done before timing ends
        auto end = high_resolution_clock::now();
        total_ms += duration<double, std::milli>(end - start).count();
    }
    return total_ms / runs;
}

void write_binary(const char* filename, float* data, int size) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        exit(1);
    }
    fwrite(data, sizeof(float), size, f);
    fclose(f);
}


float* read_binary(const char* filename, size_t elements) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open %s\n", filename);
        exit(1);
    }
    float* data = (float*)malloc(elements * sizeof(float));
    fread(data, sizeof(float), elements, file);
    fclose(file);
    return data;
}



int test_attention(){
    int B = 64, T = 256, C = 384, H = 64;
    float scale = 1.0f / sqrtf(H);

    float *q = (float*)malloc(B*T*H*sizeof(float));
    float *k = (float*)malloc(B*T*H*sizeof(float));
    float *v = (float*)malloc(B*T*H*sizeof(float));

    float *attn = (float*)malloc(B*T*T*sizeof(float));
    float *out = (float*)malloc(B*T*H*sizeof(float));

    float *x = read_binary("./data/x.bin", B*T*C);
    float *W_k = read_binary("./data/W_k.bin", H*C);
    float *W_q = read_binary("./data/W_q.bin", H*C);
    float *W_v = read_binary("./data/W_v.bin", H*C);

    float *d_q, *d_k, *d_attn, *d_v, *d_out;
    float *d_x, *d_W_k, *d_W_q, *d_W_v;
    
    cudaMalloc(&d_x, B*T*C*sizeof(float));
    cudaMalloc(&d_W_k, H*C*sizeof(float));
    cudaMalloc(&d_W_q, H*C*sizeof(float));
    cudaMalloc(&d_W_v, H*C*sizeof(float));

    cudaMalloc(&d_q, B*T*H*sizeof(float));
    cudaMalloc(&d_k, B*T*H*sizeof(float));
    cudaMalloc(&d_v, B*T*H*sizeof(float));
    cudaMalloc(&d_attn, B*T*T*sizeof(float));
    cudaMalloc(&d_out, B*T*H*sizeof(float));

    cudaMemcpy(d_x, x, B*T*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_k, W_k, H*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_q, W_q, H*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_v, W_v, H*C*sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);




    linear_layer_cuda(d_x, d_W_q, d_q, B, T, C, H);
    linear_layer_cuda( d_x, d_W_k, d_k, B, T, C, H);
    linear_layer_cuda( d_x, d_W_v, d_v, B, T, C, H);

    compute_attention_weigths_cublas(handle, d_q, d_k, d_attn, B, T, H, scale);
    apply_causal_mask_cuda(d_attn, B, T);
    softmax_reduction_cuda(d_attn, B, T);
    compute_output_cublas(handle, d_attn, d_v, d_out, B, T, H);


    cudaMemcpy(out, d_out, B*T*H*sizeof(float), cudaMemcpyDeviceToHost);


    // write_binary("./data/output.bin", out, B*T*H);
    free(x);
     free(W_q); free(W_k); free(W_v);
    free(q); free(k); free(v); free(attn); free(out);
    return 0;
    
}

int main (){
    test_attention();
    return 0;
}


    // printf("Output (B=%d, T=%d, H=%d):\n", B, T, H);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // printf("Device: %s\n", prop.name);
    // printf("Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
    // printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    
    // double bandwidth = (double)prop.memoryClockRate * prop.memoryBusWidth * 2 / 8 / 1000000;
    // printf("Theoretical Memory Bandwidth: %.2f GB/s\n", bandwidth);


        // double gpu_ms = benchmark([&]() {
    // compute_attention_weights_omp(q, k, attn, B, T, H, scale);

    // });
    // printf("compute_attention_weights_omp: %f \n", gpu_ms);