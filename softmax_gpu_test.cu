#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include "attention_cuda.h"


float* allocate_and_init(int size) {
    float* data;
    
    cudaMallocManaged(&data, size * sizeof(float));
    for (int i = 0; i < size; ++i)
        data[i] = (float)(rand() % 10);
    return data;
}

double benchmark_softmax(void (*softmax_fn)(float*, int, int), float* A, int B, long long int T, int runs) {
    printf("Warming up for B=%d, T=%lld...\n", B, T);
    for (int i = 0; i < 3; ++i) {
        softmax_fn(A, B, T);
        cudaError_t warmup_err = cudaGetLastError();
        if (warmup_err != cudaSuccess) {
            printf("Warmup run %d failed: %s\n", i, cudaGetErrorString(warmup_err));
        }
        cudaDeviceSynchronize();
    }
    printf("Warmup complete, starting timed runs...\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed = 0.0f;
    int successful_runs = 0;

    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start);
        softmax_fn(A, B, T);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error in run %d: %s\n", i, cudaGetErrorString(err));
            continue; 
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  Run %d: %.6f ms\n", i, ms);
        elapsed += ms;
        successful_runs++;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (successful_runs == 0) {
        printf("ERROR: No successful runs!\n");
        return 0.0;
    }
    
    double avg_time = elapsed / successful_runs / 1000.0;
    printf("  Average time: %.9f seconds (%d successful runs)\n", avg_time, successful_runs);
    
    if (avg_time <= 0) {
        printf("WARNING: Zero or negative execution time!\n");
    }
    
    return avg_time;
}
void save_results(const char* filename, const char* method,
                  long long int* sizes, long  double* gflops, size_t* bytes, int N) {
    FILE* f = fopen(filename, "w");
    fprintf(f, "method,input_bytes,GFLOPS\n");
    for (int i = 0; i < N; ++i) {
        fprintf(f, "%s,%zu,%Lf\n", method, bytes[i], gflops[i]);
    }
    fclose(f);
}

int main() {
    const int B = 64;
    const long long int T_vals[] = {64, 91, 128, 181, 256, 362, 512, 724, 1024, 4300};
    const long long int num_sizes = sizeof(T_vals) / sizeof(T_vals[0]);

    long long int input_sizes[num_sizes];
    size_t input_bytes[num_sizes];
    long double gflops_crude[num_sizes], gflops_reduction[num_sizes];

    for (int i = 0; i < num_sizes; ++i) {
        long long int T = T_vals[i];
        long long int size = B * T * T;
        float* A = allocate_and_init(size);

        long double time_crude = benchmark_softmax(softmax_crude_cuda, A, B, T, 10);
        long double time_red   = benchmark_softmax(softmax_reduction_cuda, A, B, T, 10);

        long double flops = 4.0L * B;
        flops *= T;
        flops *= T;

        input_sizes[i] = T;
        input_bytes[i] = size * sizeof(float);
        gflops_crude[i] = (flops / time_crude) / 1e9;
        gflops_reduction[i] = (flops / time_red) / 1e9;

        cudaFree(A);
    }

    save_results("./results/softmax_crude_cuda_perf.csv", "crude", input_sizes, gflops_crude, input_bytes, num_sizes);
    save_results("./results/softmax_reduction_cuda_perf.csv", "reduction", input_sizes, gflops_reduction, input_bytes, num_sizes);

    return 0;
}