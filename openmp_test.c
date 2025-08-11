#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "attention.h"
#include <omp.h>
#include "linear.h"
#include <string.h>

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

void save_results_to_csv(const char* filename, const char* function_name, 
                        int* threads, double* times, double* flops_per_sec, 
                        long long* mem_bytes, int num_measurements) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Failed to create %s\n", filename);
        return;
    }
    
    fprintf(file, "Function,Threads,Time(s),FLOP/s,Memory(Bytes)\n");
    for (int i = 0; i < num_measurements; i++) {
        fprintf(file, "%s,%d,%.6f,%.2f,%lld\n", 
                function_name, threads[i], times[i], flops_per_sec[i], mem_bytes[i]);
    }
    fclose(file);
    printf("Results saved to %s\n", filename);
}
void test_linear_layer(float* input, float* weights, float* output, 
                           int B, int T, int C, int H, int max_threads) {
    printf("\n=== Testing linear_layer_omp ===\n");

    // FLOP count: B*T*H dot products, each dot product does 2*C operations (multiply + add)
    long long flop_count = (long long)B * T * H * 2 * C;
    // Memory: read input (B*T*C), weights (H*C), write output (B*T*H)
    long long mem_bytes = ((long long)B * T * C + (long long)H * C + (long long)B * T * H) * sizeof(float);

    int* threads_arr = (int*)malloc(max_threads * sizeof(int));
    double* times_arr = (double*)malloc(max_threads * sizeof(double));
    double* flops_arr = (double*)malloc(max_threads * sizeof(double));
    long long* mem_arr = (long long*)malloc(max_threads * sizeof(long long));

    printf("Threads,Time(s),FLOP/s,Memory(Bytes)\n");

    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);

        double start = omp_get_wtime();
        linear_layer_omp(input, weights, output, B, T, C, H);
        double end = omp_get_wtime();

        double time = end - start;
        double flops_per_sec = flop_count / time;

        threads_arr[t-1] = t;
        times_arr[t-1] = time;
        flops_arr[t-1] = flops_per_sec;
        mem_arr[t-1] = mem_bytes;

        printf("%d,%.6f,%.2f,%lld\n", t, time, flops_per_sec, mem_bytes);
    }

    save_results_to_csv("./results/linear_layer_performance.csv", "linear_layer_omp",
                        threads_arr, times_arr, flops_arr, mem_arr, max_threads);

    free(threads_arr);
    free(times_arr);
    free(flops_arr);
    free(mem_arr);
}

void test_compute_attention_weights(float* q, float* k, float* attn, 
                                   int B, int T, int H, float scale, int max_threads) {
    printf("\n=== Testing compute_attention_weights_omp ===\n");
    
    // FLOP count: B * T * T * (2 * H) for dot products + B * T * T for scaling
    long long flop_count = (long long)B * T * T * (2 * H + 1);
    // Memory: reading q,k (2*B*T*H) + writing attn (B*T*T)
    long long mem_bytes = (long long)B * T * (2 * H * T + T) * sizeof(float);
    
    int* threads_arr = (int*)malloc(max_threads * sizeof(int));
    double* times_arr = (double*)malloc(max_threads * sizeof(double));
    double* flops_arr = (double*)malloc(max_threads * sizeof(double));
    long long* mem_arr = (long long*)malloc(max_threads * sizeof(long long));
    
    printf("Threads,Time(s),FLOP/s,Memory(Bytes)\n");
    
    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);
        
        double start = omp_get_wtime();
        compute_attention_weights_omp(q, k, attn, B, T, H, scale);
        double end = omp_get_wtime();
        
        double time = end - start;
        double flops_per_sec = flop_count / time;
        
        threads_arr[t-1] = t;
        times_arr[t-1] = time;
        flops_arr[t-1] = flops_per_sec;
        mem_arr[t-1] = mem_bytes;
        
        printf("%d,%.6f,%.2f,%lld\n", t, time, flops_per_sec, mem_bytes);
    }
    
    save_results_to_csv("./results/attention_weights_omp_performance.csv", "compute_attention_weights_omp",
                        threads_arr, times_arr, flops_arr, mem_arr, max_threads);
    
    free(threads_arr);
    free(times_arr);
    free(flops_arr);
    free(mem_arr);
}

void test_apply_causal_mask(float* attn, int B, int T, int max_threads) {
    printf("\n=== Testing apply_causal_mask_omp ===\n");
    
    // FLOP count: approximately B * T * T/2 comparisons and assignments
    long long flop_count = (long long)B * T * T / 2;
    // Memory: reading and writing attn matrix
    long long mem_bytes = (long long)B * T * T * 2 * sizeof(float);
    
    int* threads_arr = (int*)malloc(max_threads * sizeof(int));
    double* times_arr = (double*)malloc(max_threads * sizeof(double));
    double* flops_arr = (double*)malloc(max_threads * sizeof(double));
    long long* mem_arr = (long long*)malloc(max_threads * sizeof(long long));
    
    printf("Threads,Time(s),FLOP/s,Memory(Bytes)\n");
    
    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);
        
        double start = omp_get_wtime();
        apply_causal_mask_omp(attn, B, T);
        double end = omp_get_wtime();
        
        double time = end - start;
        double flops_per_sec = flop_count / time;
        
        threads_arr[t-1] = t;
        times_arr[t-1] = time;
        flops_arr[t-1] = flops_per_sec;
        mem_arr[t-1] = mem_bytes;
        
        printf("%d,%.6f,%.2f,%lld\n", t, time, flops_per_sec, mem_bytes);
    }
    
    save_results_to_csv("./results/causal_mask_omp_performance.csv", "apply_causal_mask_omp",
                        threads_arr, times_arr, flops_arr, mem_arr, max_threads);
    
    free(threads_arr);
    free(times_arr);
    free(flops_arr);
    free(mem_arr);
}

void test_softmax(float* attn, int B, int T, int max_threads) {
    printf("\n=== Testing softmax_omp ===\n");
    
    // FLOP count: B * T * (max finding + sum + exp + division) ≈ B * T * 5T
    long long flop_count = (long long)B * T * 5 * T;
    // Memory: reading and writing attention matrix
    long long mem_bytes = (long long)B * T * T * 2 * sizeof(float);
    
    int* threads_arr = (int*)malloc(max_threads * sizeof(int));
    double* times_arr = (double*)malloc(max_threads * sizeof(double));
    double* flops_arr = (double*)malloc(max_threads * sizeof(double));
    long long* mem_arr = (long long*)malloc(max_threads * sizeof(long long));
    
    printf("Threads,Time(s),FLOP/s,Memory(Bytes)\n");
    
    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);
        
        double start = omp_get_wtime();
        softmax_omp(attn, B, T);
        double end = omp_get_wtime();
        
        double time = end - start;
        double flops_per_sec = flop_count / time;
        
        threads_arr[t-1] = t;
        times_arr[t-1] = time;
        flops_arr[t-1] = flops_per_sec;
        mem_arr[t-1] = mem_bytes;
        
        printf("%d,%.6f,%.2f,%lld\n", t, time, flops_per_sec, mem_bytes);
    }
    
    save_results_to_csv("./results/softmax_omp_performance.csv", "softmax_omp",
                        threads_arr, times_arr, flops_arr, mem_arr, max_threads);
    
    free(threads_arr);
    free(times_arr);
    free(flops_arr);
    free(mem_arr);
}

void test_compute_output(float* attn, float* v, float* out, int B, int T, int H, int max_threads) {
    printf("\n=== Testing compute_output_omp ===\n");
    
    // FLOP count: B * T * T * H (matrix multiplication)
    long long flop_count = (long long)B * T * T * H;
    // Memory: reading attn (B*T*T) + reading v (B*T*H) + writing out (B*T*H)
    long long mem_bytes = (long long)B * T * (T + 2 * H) * sizeof(float);
    
    int* threads_arr = (int*)malloc(max_threads * sizeof(int));
    double* times_arr = (double*)malloc(max_threads * sizeof(double));
    double* flops_arr = (double*)malloc(max_threads * sizeof(double));
    long long* mem_arr = (long long*)malloc(max_threads * sizeof(long long));
    
    printf("Threads,Time(s),FLOP/s,Memory(Bytes)\n");
    
    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);
        
        double start = omp_get_wtime();
        compute_output_omp(attn, v, out, B, T, H);
        double end = omp_get_wtime();
        
        double time = end - start;
        double flops_per_sec = flop_count / time;
        
        threads_arr[t-1] = t;
        times_arr[t-1] = time;
        flops_arr[t-1] = flops_per_sec;
        mem_arr[t-1] = mem_bytes;
        
        printf("%d,%.6f,%.2f,%lld\n", t, time, flops_per_sec, mem_bytes);
    }
    
    save_results_to_csv("./results/compute_output_omp_performance.csv", "compute_output_omp",
                        threads_arr, times_arr, flops_arr, mem_arr, max_threads);
    
    free(threads_arr);
    free(times_arr);
    free(flops_arr);
    free(mem_arr);
}

int main() {
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
    
    int max_threads = omp_get_max_threads();
    printf("Maximum threads available: %d\n", max_threads);
    printf("Problem size: B=%d, T=%d, C=%d, H=%d\n", B, T, C, H);
    
    linear_layer_omp(x, W_q, q, B, T, C, H);
    linear_layer_omp(x, W_k, k, B, T, C, H);
    linear_layer_omp(x, W_v, v, B, T, C, H);
    
    
    test_linear_layer(x, W_q, q, B, T, C, H, max_threads);
    test_compute_attention_weights(q, k, attn, B, T, H, scale, max_threads);
    test_apply_causal_mask(attn, B, T, max_threads);
    test_softmax(attn, B, T, max_threads);
    test_compute_output(attn, v, out, B, T, H, max_threads);
    
    printf("\n=== All tests completed ===\n");
    printf("Results saved to individual CSV files:\n");
    printf("- attention_weights_performance.csv\n");
    printf("- causal_mask_performance.csv\n");
    printf("- softmax_performance.csv\n");
    printf("- compute_output_performance.csv\n");
    
    // Cleanup
    free(q);
    free(k);
    free(v);
    free(attn);
    free(out);
    free(x);
    free(W_k);
    free(W_q);
    free(W_v);
    
    return 0;
}