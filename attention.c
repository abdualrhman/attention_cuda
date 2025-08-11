#include <math.h>
#include <stdio.h>
#include <omp.h>

void compute_attention_weights(float* q, float* k, float* attn,
    int B, int T, int head_size, float scale) {
        // q @ k.T scaled
    for (int b = 0; b < B; ++b) {
        for (int t1 = 0; t1 < T; ++t1) {
            for (int t2 = 0; t2 < T; ++t2) {
                float score = 0.0f;
                for (int h = 0; h < head_size; ++h) {
                    score += q[(b*T + t1)*head_size + h] * k[(b*T + t2)*head_size + h];
                }
                attn[(b*T + t1)*T + t2] = score * scale;
            }
        }
    }
}

// set upper triangle to -INF
void apply_causal_mask(float* attn, int B, int T) {
    for (int b = 0; b < B; ++b) {
        for (int t1 = 0; t1 < T; ++t1) {
            for (int t2 = t1 + 1; t2 < T; ++t2) {
                attn[(b*T + t1)*T + t2] =  -INFINITY;  // or -1e9f 
            }
        }
    }
}

void compute_output(
    float* attn, // (B*T*T)
    float* v, // (B*T*H)
    float* out, // (B*T*H)
    int B,
    int T,
    int head_size
) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < head_size; ++h) {
                float sum = 0.0f;
                for (int t2 = 0; t2 < T; ++t2) {
                    sum += attn[(b*T + t)*T + t2] * v[(b*T + t2)*head_size + h];
                }
                out[(b*T + t)*head_size + h] = sum;
            }
        }
    }
}

void softmax(
            float* mat, // (B*T*T)
            int B,
            int T       
) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            float max_val = -INFINITY;
            for (int i = 0; i < T; ++i) {
                float val = mat[(b*T + t)*T + i];
                if (val > max_val) max_val = val;
            }

            float sum = 0.0f;
            for (int i = 0; i < T; ++i) {
                float val = expf(mat[(b*T + t)*T + i] - max_val);  // stability
                mat[(b*T + t)*T + i] = val;
                sum += val;
            }

            for (int i = 0; i < T; ++i) {
                mat[(b*T + t)*T + i] /= sum;
            }
        }
    }
}



void compute_attention_weights_omp(float* q, float* k, float* attn,
    int B, int T, int head_size, float scale) {
        // q @ k.T scaled
    // omp_set_num_threads(62);
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; ++b) {
        for (int t1 = 0; t1 < T; ++t1) {
            for (int t2 = 0; t2 < T; ++t2) {
                float score = 0.0f;
                for (int h = 0; h < head_size; ++h) {
                    score += q[(b*T + t1)*head_size + h] * k[(b*T + t2)*head_size + h];
                }
                attn[(b*T + t1)*T + t2] = score * scale;
            }
        }
    }
}

void apply_causal_mask_omp(float* attn, int B, int T) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t1 = 0; t1 < T; ++t1) {
            for (int t2 = t1 + 1; t2 < T; ++t2) {
                attn[(b*T + t1)*T + t2] = -INFINITY;  
            }
        }
    }
}

void softmax_omp(
            float* mat, // (B*T*T)
            int B,
            int T       
) {
    #pragma omp parallel for
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            float max_val = -INFINITY;
            for (int i = 0; i < T; ++i) {
                float val = mat[(b*T + t)*T + i];
                if (val > max_val) max_val = val;
            }

            float sum = 0.0f;
            for (int i = 0; i < T; ++i) {
                float val = expf(mat[(b*T + t)*T + i] - max_val);  // stability
                mat[(b*T + t)*T + i] = val;
                sum += val;
            }

            for (int i = 0; i < T; ++i) {
                mat[(b*T + t)*T + i] /= sum;
            }
        }
    }
}

void compute_output_omp(
    float* attn, // (B*T*T)
    float* v, // (B*T*H)
    float* out, // (B*T*H)
    int B,
    int T,
    int head_size
) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < head_size; ++h) {
                float sum = 0.0f;
                for (int t2 = 0; t2 < T; ++t2) {
                    sum += attn[(b*T + t)*T + t2] * v[(b*T + t2)*head_size + h];
                }
                out[(b*T + t)*head_size + h] = sum;
            }
        }
    }
}