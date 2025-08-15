
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