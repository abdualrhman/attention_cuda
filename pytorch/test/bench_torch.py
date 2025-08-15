import argparse
import os
import sys
import numpy as np
import torch


import torch
from torch.nn import functional as F


def torch_attention(x, key, query, value, H):
    k = key(x)      # (B,T,H)
    q = query(x)    # (B,T,H)
    v = value(x)    # (B,T,H)
    wei = (q @ k.transpose(-2, -1)) * (H ** -0.5)  # (B,T,T)
    T = x.size(1)
    tril = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    out = wei @ v                                  # (B,T,H)
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
args = parser.parse_args()
data_dir = args.data_dir

with open(os.path.join(data_dir, "tensor_shape.txt"), "r") as f:
    line = f.readline()
    B, T, C, H = map(int, line.strip().split())


x_data = np.fromfile(os.path.join(data_dir, "x.bin"), dtype=np.float32)
W_k_data = np.fromfile(os.path.join(data_dir,'W_k.bin'), dtype=np.float32)
W_q_data = np.fromfile(os.path.join(data_dir,'W_q.bin'), dtype=np.float32)
W_v_data = np.fromfile(os.path.join(data_dir,'W_v.bin'), dtype=np.float32)


W_k_tensor = torch.tensor(W_k_data).reshape(H, C)
W_q_tensor = torch.tensor(W_q_data).reshape(H, C)
W_v_tensor = torch.tensor(W_v_data).reshape(H, C)

key = torch.nn.Linear(C, H, bias=False)
query = torch.nn.Linear(C, H, bias=False)
value = torch.nn.Linear(C, H, bias=False)
with torch.no_grad():
    key.weight.copy_(W_k_tensor)
    query.weight.copy_(W_q_tensor)
    value.weight.copy_(W_v_tensor)
    
x = torch.tensor(x_data).reshape(B,T,C)

def bench_torch(B, T, C, H, runs=50, warmup=10, dtype=torch.float32):
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(B, T, C, device=device, dtype=dtype)
    key = torch.nn.Linear(C, H, bias=False, device=device, dtype=dtype)
    query = torch.nn.Linear(C, H, bias=False, device=device, dtype=dtype)
    value = torch.nn.Linear(C, H, bias=False, device=device, dtype=dtype)

    # warmup
    for _ in range(warmup):
        _ = torch_attention(x, key, query, value, H)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(runs):
        start.record()
        _ = torch_attention(x, key, query, value, H)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))   # ms

    return sum(times_ms)/len(times_ms)   


avg_ms = bench_torch(B, T, C, H)
print(f"PyTorch avg latency: {avg_ms:.3f} ms")