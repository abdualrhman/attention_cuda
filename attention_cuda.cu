#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>


#define BLOCK_DIM_Y 128


void compute_attention_weigths_cublas(
    cublasHandle_t handle,
    float* q,       // (B * T * H)
    float* k,       // (B * T * H)
    float* attn,    // (B * T * T)
    int B, int T, int H, 
    float scale
){
        // compute  q * k.T scaled
        // cublasSgemmStridedBatched computes C = alpha * op(A) * op(B) + beta * C 
        
        // in row-major, lda is the number of columns
        // in column-major, lda is the number of rows
        // the matrices are stored in row-major, hence:
        int lda = H; // q
        int ldb = H; // k
        int ldc = T; // attn
 
        int m_dim = T; // rows of op(A) and rows of C
        int n_dim = T; // cols of op(B) and cols of C
        int k_dim = H; // shared dim of op(A) and op(B)

        long long strideA = T * H;
        long long strideB = T * H;
        long long strideC = T * T;

        cublasOperation_t opA = CUBLAS_OP_N;
        cublasOperation_t opB = CUBLAS_OP_T;

        float alpha = scale;
        float beta = 0.0f;

        cublasStatus_t stat = cublasSgemmStridedBatched(
            handle, 
            opB, opA,
            m_dim, n_dim, k_dim, 
            &alpha,
            k, ldb, strideB, // k
            q, lda, strideA, // q
            &beta, 
            attn, ldc, strideC,
            B
        );

        if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error in sgemmStridedBatched: %d\n", stat);
    }
}


__global__ void apply_causal_mask_kernel(float *attn, int B, int T){
    int b = blockIdx.z;
    int t1 = (blockIdx.x * blockDim.x) + threadIdx.x; // col
    int t2 = (blockIdx.y * blockDim.y) + threadIdx.y; // row
    
    // if (b >= B || t1 >= T || t2 >= T) return;
    
    if (b < B && t1< T && t2 < T && t1 < t2){
        attn[(b * T + t1) * T + t2] = -INFINITY; 
    }
}

// set upper triangle to -INF
void apply_causal_mask_cuda(float *attn, int B, int T){
    // input matrix of size (B * T * T)
    // parallelize over batches 
    int BLOCK_SIZE = 16;
    int GRID_SIZE = (int) ceil(T/BLOCK_SIZE);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(GRID_SIZE, GRID_SIZE, B);

    apply_causal_mask_kernel<<<grid, block>>>(attn, B, T);
}




__global__
void softmax_crude_kernel(
    float *A, 
    int B, 
    int T
){
    int b = blockIdx.z;
    int row = (blockIdx.y * blockDim.y) + threadIdx.y; 

    if (row < T && b < B){
        float max_val = A[(b * T + row) * T];
        for (int i = 0; i < T; ++i){
            max_val = max(A[(b * T + row) * T + i], max_val);
        }
        float sum = 0.0f;
        for (int i = 0; i < T; ++i){
            float val = expf(A[(b * T + row) * T + i] - max_val); // overflow stability 
            A[(b * T + row) * T + i] = val;
            sum += val;
        }
        for (int i = 0; i < T; ++i){
            A[(b * T + row) * T + i] /= sum;
        }
    }
}

void softmax_crude_cuda(
    float *A, 
    int B, 
    int T
){
    // input matrix of size (B * T * T)
    // parallelize over batches 
    int BLOCK_SIZE = 256;
    int GRID_SIZE = (int) ceil(T/BLOCK_SIZE);

    dim3 block(1, BLOCK_SIZE);
    dim3 grid(1, GRID_SIZE, B);

    softmax_crude_kernel<<<grid, block>>>(A, B, T);
}

__global__ void softmax_reduction_kernel(
    float* A,  
    int B,     
    int T      
) {
    // Each thread block processes one row across all batches
    int batch_id = blockIdx.z;  // Which batch (0 to B-1)
    int row = blockIdx.x;       // Which row within T x T matrix (0 to T-1)
    int ty = threadIdx.y;       // Thread within block (0 to BLOCK_DIM_Y-1)
    int warp_id = ty / 32;
    
    __shared__ float reduction[BLOCK_DIM_Y/32];
    
    // Only process valid batches and rows
    if (batch_id >= B || row >= T) {
        return;  // Early exit for invalid threads
    }
    // Calculate base offset for this batch and row
    int base_offset = batch_id * T * T + row * T;
    
    // Step 1: Find maximum value in this row (for numerical stability)
    float maxval = -INFINITY;
    for (int i = ty; i < T; i += BLOCK_DIM_Y) {
        float val = A[base_offset + i];
        if (val != -INFINITY) {  // Only consider non-infinite values for max
            maxval = fmaxf(maxval, val);
        }
    }
    
    // Warp-level reduction for maximum
    for (int mask = 16; mask > 0; mask /= 2) {
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
    }
    
    // Store warp results in shared memory
    if (ty % 32 == 0) {
        reduction[warp_id] = maxval;
    }
    __syncthreads();
    
    // Block-level reduction for maximum
    if (warp_id == 0) {
        maxval = ty < BLOCK_DIM_Y/32 ? reduction[ty] : -INFINITY;
        for (int mask = 16; mask > 0; mask /= 2) {
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
        }
    }
    
    // Broadcast final maximum to all threads
    if (ty == 0) {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    
    // Step 2: Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int i = ty; i < T; i += BLOCK_DIM_Y) {
        float val = A[base_offset + i];
        if (val != -INFINITY) {
            sum_exp += expf(val - maxval);
        }
        // If val == -INFINITY, exp(-INFINITY - maxval) = 0, so we add nothing
    }
    
    // Warp-level reduction for sum
    for (int mask = 16; mask > 0; mask /= 2) {
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, mask, 32);
    }
    
    // Store warp results in shared memory
    if (ty % 32 == 0) {
        reduction[warp_id] = sum_exp;
    }
    __syncthreads();
    
    // Block-level reduction for sum
    if (warp_id == 0) {
        sum_exp = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0.0f;
        for (int mask = 16; mask > 0; mask /= 2) {
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, mask, 32);
        }
    }
    
    // Broadcast final sum to all threads
    if (ty == 0) {
        reduction[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = reduction[0];
    
    // Step 3: Compute final softmax values (in-place)
    for (int i = ty; i < T; i += BLOCK_DIM_Y) {
        float val = A[base_offset + i];
        if (val == -INFINITY || sum_exp == 0.0f) {
            A[base_offset + i] = 0.0f;  // -INFINITY or all-INFINITY row maps to 0
        } else {
            A[base_offset + i] = expf(val - maxval) / sum_exp;
        }
    }
}



void softmax_reduction_cuda(
    float *A, 
    int B, 
    int T
){
    // input matrix of size (B * T * T)
    // parallelize over batches 
    
    // Grid dimensions:
    // - blockIdx.x: handles rows (0 to T-1)
    // - blockIdx.z: handles batches (0 to B-1)
    dim3 blockSize(1, BLOCK_DIM_Y, 1);
    dim3 gridSize(T, 1, B);
    
    softmax_reduction_kernel<<<gridSize, blockSize>>>(A, B, T);
}


void compute_output_cublas(
    cublasHandle_t cublas_handle,
    float* attn, // (B*T*T) 
    float* v,    // (B*T*H) 
    float* out,  // (B*T*H) 
    int B,
    int T,
    int head_size
) {
        // compute  attn * v
        // cublasSgemmStridedBatched computes C = alpha * op(A) * op(B) + beta * C 
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    

    cublasSgemmStridedBatched(
        cublas_handle,
        CUBLAS_OP_N,    
        CUBLAS_OP_N,   
        head_size,     
        T,             
        T,             
        &alpha,        
        v,             
        head_size,     
        T * head_size, 
        attn,          
        T,              
        T * T,          
        &beta,          
        out,            
        head_size,      
        T * head_size,  
        B               
    );
}


__global__ void linear_layer_kernel(
    const float* __restrict__ input,    // (B, T, C)
    const float* __restrict__ weights,  // (head_size, C)
    float* output,                      // (B, T, head_size)
    int B,
    int T,
    int C,
    int head_size
) {
    int b = blockIdx.z;
    int t = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < B && t < T && h < head_size) {
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            float inp = input[(b * T + t) * C + c];
            float w = weights[h * C + c];
            sum += inp * w;
        }
        output[(b * T + t) * head_size + h] = sum;
    }
}

void linear_layer_cuda(
        float* input,       // (B, T, C)
    float* weights,     // (head_size, C)
    float* output,      // (B, T, head_size)
    int B,
    int T,
    int C,
    int head_size
){
    dim3 blockDim(256);  
    dim3 gridDim((head_size + blockDim.x - 1) / blockDim.x, T, B);

linear_layer_kernel<<<gridDim, blockDim>>>(
    input, weights, output, B, T, C, head_size);
}