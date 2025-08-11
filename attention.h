void compute_attention_weights(
    float* q,
    float* k,
    float* attn,
    int B, 
    int T, 
    int head_size, 
    float scale
);

void apply_causal_mask(float* attn, int B, int T);

void compute_output(
    float* attn,
    float* v,
    float* out,
    int B,
    int T,
    int head_size
);

void softmax(
    float* mat,
    int B,
    int T       
);


void compute_attention_weights_omp(
    float* q,
    float* k,
    float* attn,
    int B, 
    int T, 
    int head_size, 
    float scale
);

void apply_causal_mask_omp(float* attn, int B, int T);

void softmax_omp(
    float* mat,
    int B,
    int T       
);


void compute_output_omp(
    float* attn,
    float* v,
    float* out,
    int B,
    int T,
    int head_size
);