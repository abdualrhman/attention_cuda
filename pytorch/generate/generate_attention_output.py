import argparse
import os
import numpy as np
import torch 
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
args = parser.parse_args()
data_dir = args.data_dir

with open(os.path.join(data_dir, "tensor_shape.txt"), "r") as f:
    line = f.readline()
    B, T, C, H = map(int, line.strip().split())

# k_data = np.fromfile(os.path.join(data_dir, "k.bin"), dtype=np.float32)
# q_data = np.fromfile(os.path.join(data_dir, "q.bin"), dtype=np.float32)
# v_data = np.fromfile(os.path.join(data_dir, "v.bin"), dtype=np.float32)
x_data = np.fromfile(os.path.join(data_dir, "x.bin"), dtype=np.float32)
W_k_data = np.fromfile(os.path.join(data_dir,'W_k.bin'), dtype=np.float32)
W_q_data = np.fromfile(os.path.join(data_dir,'W_q.bin'), dtype=np.float32)
W_v_data = np.fromfile(os.path.join(data_dir,'W_v.bin'), dtype=np.float32)

# k = torch.tensor(k_data).reshape(B,T,H)
# q = torch.tensor(q_data).reshape(B,T,H)
# v = torch.tensor(v_data).reshape(B,T,H)
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
k = key(x) 
q = query(x)
v = value(x)


wei = q @ k.transpose(-2, -1) * H ** -0.5 # (B,T,16) @ (B,16,T) --> (B, T, T)
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

out = wei @ v 

out.detach().numpy().tofile(os.path.join(data_dir, "attention_weights.bin"))

print("attention weights saved to", os.path.join(data_dir, "attention_weights.bin"))
