# CUDA Self-Attention Implementation

This repository contains a CUDA implementation of the self-attention mechanism, with both CPU and GPU versions. It includes utilities for profiling, testing, and comparing performance.

---
## Generate the data
The input and test data are generated using python

### installing the dependencies 

1. Install dependencies using Conda:
```bash
conda install --yes --file requirements.txt
```
### generate inputs and reference output
to generate the inputs
```
python pytorch/generate/generate_input_qkv.py --data_dir ./data
```
to generate the outputs
```
python pytorch/generate/generate_attention_output.py --data_dir ./data
```

## Generate attention weights using CUDA
### Switch to H100 
```
h100sh 
```
### Load the NVIDIA HPC SDK module 
```
module load nvhpc/24.11
```
### GPU Self-Attention benchmark
benchmark self-attention CUDA implementaion and print results 
```
nvc++ -cuda attention_benchmark.cu linear.c attention_cuda.cu attention.c -o attention_benchmark -lm -lcublas -lcurand
```
```
attention_benchmark
```
### OpenMP  benchmark
benchmark generating attention weights using openmp with 500 threads and save the results in `./results`
```
g++ openmp_benchmarck.c linear.c attention.c -o openmp_benchmarck -fopenmp
```
```
OMP_NUM_THREADS=500 openmp_benchmarck
```
open `pytorch/plots/plot_openmp.ipynb` to plot the results.

### GPU Softmax benchmark
benchmark softmax CUDA implementaion and save results in `./results`
```
nvc++ -cuda softmax_gpu_test.cu linear.c attention_cuda.cu attention.c -o softmax_gpu_test -lm -lcublas -lcurand
```
```
softmax_gpu_test
```
open `pytorch/plots/plot_softmax_cuda_perf.ipynb` to plot the results.


### Generate attention weights using CUDA 
```
nvc++ -cuda generate_attention_weights.cu linear.c attention_cuda.cu attention.c -o generate_attention_weights -lm -lcublas -lcurand
```
this will generate attention weights
```
generate_attention_weights
```

## Test against reference output
to test the generated attention weights  by CUDA (see above on how to generate), run the following 
```
python pytorch/test/test_output.py --data_dir ./data 
```
if the difference between the weights are greater than `1e-4`, an exception will be thrown. 




