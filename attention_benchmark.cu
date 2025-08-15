#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "linear.h"
#include "attention.h"
#include "attention_cuda.h"


#define CHECK_CUDA(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

#define CHECK_CUBLAS(x) do { \
  cublasStatus_t st = (x); \
  if (st != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
    exit(1); \
  } \
} while(0)

static void write_binary(const char* filename, const float* data, size_t elements) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Error opening %s\n", filename); exit(1); }
    fwrite(data, sizeof(float), elements, f);
    fclose(f);
}

static float* read_binary(const char* filename, size_t elements) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", filename); exit(1); }
    float* data = (float*)malloc(elements * sizeof(float));
    if (!data) { fprintf(stderr, "malloc failed for %zu floats\n", elements); exit(1); }
    size_t got = fread(data, sizeof(float), elements, f);
    if (got != elements) { fprintf(stderr, "Short read in %s (got %zu, want %zu)\n", filename, got, elements); exit(1); }
    fclose(f);
    return data;
}


struct Shapes {
    int B, T, C, H;
    float scale;  
};


static void attention_forward(
    cublasHandle_t handle,
    Shapes& s,
    float* d_x,
    float* d_Wq,
    float* d_Wk,
    float* d_Wv,
    float* d_q,
    float* d_k,
    float* d_v,
    float* d_attn,
    float* d_out
) {
    linear_layer_cuda(d_x, d_Wq, d_q, s.B, s.T, s.C, s.H);
    linear_layer_cuda(d_x, d_Wk, d_k, s.B, s.T, s.C, s.H);
    linear_layer_cuda(d_x, d_Wv, d_v, s.B, s.T, s.C, s.H);

    compute_attention_weigths_cublas(handle, d_q, d_k, d_attn, s.B, s.T, s.H, s.scale);

    apply_causal_mask_cuda(d_attn, s.B, s.T);

    softmax_reduction_cuda(d_attn, s.B, s.T);

    compute_output_cublas(handle, d_attn, d_v, d_out, s.B, s.T, s.H);
}

template <typename Work>
static float time_with_events(Work&& work, int warmup = 5, int runs = 30) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) { work(); }
    CHECK_CUDA(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < runs; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        work();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_ms / runs;
}

// Time each stage separately (optional detailed breakdown).
struct StageTimes {
    float linear_q_ms, linear_k_ms, linear_v_ms;
    float qkt_ms, mask_ms, softmax_ms, av_ms;
    float total_ms;
};

static StageTimes attention_forward_timed_breakdown(
    cublasHandle_t handle,
 Shapes& s,
 float* d_x,
 float* d_Wq,
 float* d_Wk,
 float* d_Wv,
    float* d_q,
    float* d_k,
    float* d_v,
    float* d_attn,
    float* d_out,
    int warmup = 5,
    int runs = 30
) {
    cudaEvent_t st, sp;
    CHECK_CUDA(cudaEventCreate(&st));
    CHECK_CUDA(cudaEventCreate(&sp));

    auto time_stage = [&](auto&& stage_callable) {
        for (int i = 0; i < warmup; ++i) stage_callable();
        CHECK_CUDA(cudaDeviceSynchronize());

        float acc = 0.0f;
        for (int i = 0; i < runs; ++i) {
            CHECK_CUDA(cudaEventRecord(st));
            stage_callable();
            CHECK_CUDA(cudaEventRecord(sp));
            CHECK_CUDA(cudaEventSynchronize(sp));
            float ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, st, sp));
            acc += ms;
        }
        return acc / runs;
    };

    StageTimes t{};

    t.linear_q_ms = time_stage([&]{ linear_layer_cuda(d_x, d_Wq, d_q, s.B, s.T, s.C, s.H); });
    t.linear_k_ms = time_stage([&]{ linear_layer_cuda(d_x, d_Wk, d_k, s.B, s.T, s.C, s.H); });
    t.linear_v_ms = time_stage([&]{ linear_layer_cuda(d_x, d_Wv, d_v, s.B, s.T, s.C, s.H); });

    t.qkt_ms = time_stage([&]{ compute_attention_weigths_cublas(handle, d_q, d_k, d_attn, s.B, s.T, s.H, s.scale); });

    t.mask_ms = time_stage([&]{ apply_causal_mask_cuda(d_attn, s.B, s.T); });

    t.softmax_ms = time_stage([&]{ softmax_reduction_cuda(d_attn, s.B, s.T); });

    t.av_ms = time_stage([&]{ compute_output_cublas(handle, d_attn, d_v, d_out, s.B, s.T, s.H); });

    t.total_ms = t.linear_q_ms + t.linear_k_ms + t.linear_v_ms + t.qkt_ms + t.mask_ms + t.softmax_ms + t.av_ms;

    CHECK_CUDA(cudaEventDestroy(st));
    CHECK_CUDA(cudaEventDestroy(sp));

    return t;
}

static float attention_forward_timed_e2e(
    cublasHandle_t handle,
     Shapes& s,
     float* d_x,
     float* d_Wq,
     float* d_Wk,
     float* d_Wv,
    float* d_q,
    float* d_k,
    float* d_v,
    float* d_attn,
    float* d_out,
    int warmup = 5,
    int runs = 30
) {
    return time_with_events([&](){
        attention_forward(handle, s, d_x, d_Wq, d_Wk, d_Wv, d_q, d_k, d_v, d_attn, d_out);
    }, warmup, runs);
}


int main() {
    Shapes s;
    s.B = 64; s.T = 256; s.C = 384; s.H = 64;
    s.scale = 1.0f / sqrtf((float)s.H);

    float* h_x  = read_binary("./data/x.bin",   (size_t)s.B*s.T*s.C);
    float* h_Wq = read_binary("./data/W_q.bin", (size_t)s.H*s.C);
    float* h_Wk = read_binary("./data/W_k.bin", (size_t)s.H*s.C);
    float* h_Wv = read_binary("./data/W_v.bin", (size_t)s.H*s.C);

    float *d_x, *d_Wq, *d_Wk, *d_Wv;
    float *d_q, *d_k, *d_v, *d_attn, *d_out;

    CHECK_CUDA(cudaMalloc(&d_x,    (size_t)s.B*s.T*s.C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq,   (size_t)s.H*s.C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk,   (size_t)s.H*s.C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv,   (size_t)s.H*s.C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q,    (size_t)s.B*s.T*s.H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k,    (size_t)s.B*s.T*s.H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v,    (size_t)s.B*s.T*s.H*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn, (size_t)s.B*s.T*s.T*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out,  (size_t)s.B*s.T*s.H*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x,  h_x,  (size_t)s.B*s.T*s.C*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq, (size_t)s.H*s.C*sizeof(float),     cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk, (size_t)s.H*s.C*sizeof(float),     cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, (size_t)s.H*s.C*sizeof(float),     cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float avg_ms = attention_forward_timed_e2e(
        handle, s, d_x, d_Wq, d_Wk, d_Wv, d_q, d_k, d_v, d_attn, d_out,
        5, 50
    );
    printf("Attention pipeline (E2E): %.3f ms (avg over runs)\n", avg_ms);

    StageTimes st = attention_forward_timed_breakdown(
        handle, s, d_x, d_Wq, d_Wk, d_Wv, d_q, d_k, d_v, d_attn, d_out,
       5, 50
    );
    printf("Breakdown (avg ms): Q %.3f | K %.3f | V %.3f | QK^T %.3f | mask %.3f | softmax %.3f | AV %.3f | sum %.3f\n",
           st.linear_q_ms, st.linear_k_ms, st.linear_v_ms, st.qkt_ms, st.mask_ms, st.softmax_ms, st.av_ms, st.total_ms);

    float* h_out = (float*)malloc((size_t)s.B*s.T*s.H*sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_out, d_out, (size_t)s.B*s.T*s.H*sizeof(float), cudaMemcpyDeviceToHost));
    write_binary("./data/output.bin", h_out, (size_t)s.B*s.T*s.H);
    printf("Saved ./data/output.bin\n");

    free(h_x); free(h_Wq); free(h_Wk); free(h_Wv); free(h_out);
    CHECK_CUDA(cudaFree(d_x)); CHECK_CUDA(cudaFree(d_Wq)); CHECK_CUDA(cudaFree(d_Wk)); CHECK_CUDA(cudaFree(d_Wv));
    CHECK_CUDA(cudaFree(d_q)); CHECK_CUDA(cudaFree(d_k)); CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_attn)); CHECK_CUDA(cudaFree(d_out));
    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
