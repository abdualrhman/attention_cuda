#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "linear.h"
#include "attention.h"
#include <chrono>
#include <omp.h>

template <typename Func>
double benchmark_cpu(Func&& fn, int runs = 10) {
    using namespace std::chrono;

    double total_ms = 0.0;

    for (int i = 0; i < runs; ++i) {
        auto start = high_resolution_clock::now();
        fn();  
        auto end = high_resolution_clock::now();

        double elapsed_ms = duration<double, std::milli>(end - start).count();
        total_ms += elapsed_ms;
    }

    return total_ms / runs;  // Return average time in milliseconds
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

    linear_layer_omp(x, W_q, q, B, T, C, H);
    linear_layer_omp(x, W_k, k, B, T, C, H);
    linear_layer_omp(x, W_v, v, B, T, C, H);

    omp_set_num_threads(62); 
    compute_attention_weights_omp(q, k, attn, B, T, H, scale);
    
    
    
    apply_causal_mask_omp(attn, B, T);

    omp_set_num_threads(50); 
    // printf("num_threads at runtime: %d\n", omp_get_max_threads());
    // double gpu_ms = benchmark_cpu([&]() {
    // softmax_omp(attn, B, T);


    // });
    // printf("softmax_omp: %f \n", gpu_ms);
    softmax_omp(attn, B, T);
    compute_output_omp(attn, v, out, B, T, H);

    
    write_binary("./data/output.bin", out, B*T*H);
    
    free(x);
     free(W_q); free(W_k); free(W_v);
    free(q); free(k); free(v); free(attn); free(out);
    return 0;
    
}

int main (){
    test_attention();
    return 0;
}


