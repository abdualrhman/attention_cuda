void linear_layer(
    float* input,        // (B, T, C)
    float* weights,      // (head_size, C)
    float* output,       // (B, T, head_size)
    int B,
    int T,
    int C,
    int head_size
);
void linear_layer_omp(
    float* input,        // (B, T, C)
    float* weights,      // (head_size, C)
    float* output,       // (B, T, head_size)
    int B,
    int T,
    int C,
    int head_size
);