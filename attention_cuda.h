#include <cublas_v2.h>
#include <cuda_runtime.h>

void compute_attention_weigths_cublas(cublasHandle_t handle, float* q, float* k, float* attn,
    int B, int T, int head_size, float scale);

void apply_causal_mask_cuda(float *attn, int B, int T);
void softmax_crude_cuda(float *A, int B, int T);
void softmax_reduction_cuda(float * A, int B, int T);
void compute_output_cublas(
    cublasHandle_t handle,
    float *attn,    // (B * T * T)
    float *v,       // (B * T * H)
    float *out,     // (B * T * H)
    int B, 
    int T, 
    int H
);


void linear_layer_cuda(
    float* input,       // (B, T, C)
    float* weights,     // (head_size, C)
    float* output,      // (B, T, head_size)
    int B,
    int T,
    int C,
    int head_size
);