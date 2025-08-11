import torch
import numpy as np
import argparse
import os
import torch.nn as nn

torch.manual_seed(1337)
# 64 256 384 64
parser = argparse.ArgumentParser(description="Generate random Q, K, V tensors and save them as .npy files.")
parser.add_argument("--B", default=64, type=int, help="Batch size of the tensors")
parser.add_argument("--T", default=256,type=int, help="Sequence length of the tensors (time)")
parser.add_argument("--C", default=384, type=int, help="Embedding dimension of the tensors (channel)")
parser.add_argument("--H", default=64, type=int, help="Head size")
parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")

args = parser.parse_args()

B = args.B
T = args.T
C = args.C
H = args.H
data_dir = args.data_dir

x=torch.randn(B, T, C)
Q = torch.randn(B, T, H, dtype=torch.float32)
K = torch.randn(B, T, H, dtype=torch.float32)
V = torch.randn(B, T, H, dtype=torch.float32)


key = nn.Linear(C, H, bias=False, dtype=torch.float32)
query = nn.Linear(C, H, bias=False, dtype=torch.float32)
value = nn.Linear(C, H, bias=False, dtype=torch.float32)

W_k = key.weight.detach().cpu().numpy()
W_q = query.weight.detach().cpu().numpy()
W_v = value.weight.detach().cpu().numpy()


x.numpy().tofile(os.path.join(data_dir, "x.bin"))
Q.numpy().tofile(os.path.join(data_dir, "q.bin"))
K.numpy().tofile(os.path.join(data_dir, "k.bin"))
V.numpy().tofile(os.path.join(data_dir, "v.bin"))

W_k.tofile(os.path.join(data_dir, "W_k.bin"))
W_q.tofile(os.path.join(data_dir, "W_q.bin"))
W_v.tofile(os.path.join(data_dir, "W_v.bin"))


with open(os.path.join(data_dir, "tensor_shape.txt"), "w") as f:
    f.write(f"{B} {T} {C} {H}")
print("data generated", B, T, C, H)