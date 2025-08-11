
```
conda install --yes --file requirements.txt
```
## linear.c
```
output = x @ W.T + b

```

```
g++ test.c linear.c attention.c -o test -lm

```


module load nvhpc/24.11

nvc++ -cuda cudaDeviceQuery.cpp -o cudaDeviceQuery


nvc++ -cuda  test.cu linear.c attention_cuda.cu attention.c -o test -lm -lcublas -lcurand


nvc++ -cuda  openmp_test.c linear.c attention_cuda.cu attention.c -o openmp_test -lm -lcublas -lcurand -mp



g++ cpu_test.c linear.c  attention.c -o cpu_test -lm -fopenmp
OMP_NUM_THREADS=4

``` c

__global__ void softmax_kernel4(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  __shared__ float reduction[BLOCK_DIM_Y/32]; 
  if (row < h)
  {
    float maxval = 0;
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      // get the maximum of the block 
      maxval = fmaxf(maxval, a[row*w + i]);
    }

    for (int mask = 16; mask>0; mask/=2)
    {
      maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = maxval;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        maxval = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
        }
    }
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    float divisor = 0.f;
    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      divisor += __expf(a[row*w + i] - maxval);
    }
    for (int mask = 16; mask>0; mask/=2)
    {
      divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
    }

    if (ty%32 == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        for (int mask = 16; mask>0; mask/=2)
        {
          divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        }
    }
    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

    for (int i = ty; i<w; i+=BLOCK_DIM_Y)
    {
      b[row*w + i] = __expf(a[row*w + i]-maxval)/divisor;
    }
  }
}

```

`__shfl_xor_sync(...)` allows each thread to access the bit values in the samce warp.





softmax:
float_count: 5T*B

