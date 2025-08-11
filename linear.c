#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// output = x @ W.T + b
void linear_layer(
    float* input,        // (B, T, C)
    float* weights,      // (head_size, C)
    float* output,       // (B, T, head_size)
    int B,
    int T,
    int C,
    int head_size
) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < head_size; ++h) {
                float sum = 0.0f;
                for (int c = 0; c < C; ++c) {
                    sum += input[(b * T + t) * C + c] * weights[h * C + c];
                }
                output[(b * T + t) * head_size + h] = sum;
            }
        }
    }
}


void linear_layer_omp(
    float* input,        // (B, T, C)
    float* weights,      // (head_size, C)
    float* output,       // (B, T, head_size)
    int B,
    int T,
    int C,
    int head_size
) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < head_size; ++h) {
                float sum = 0.0f;
                for (int c = 0; c < C; ++c) {
                    sum += input[(b * T + t) * C + c] * weights[h * C + c];
                }
                output[(b * T + t) * head_size + h] = sum;
            }
        }
    }
}