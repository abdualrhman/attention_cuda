import numpy as np
import os 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
args = parser.parse_args()

data_dir = args.data_dir

with open(os.path.join(data_dir, "tensor_shape.txt"), "r") as f:
    line = f.readline()
    B, T, C, H = map(int, line.strip().split())
    
out_c = np.fromfile(os.path.join(data_dir, "output.bin"), dtype=np.float32).reshape(B, T, H)
out_ref = np.fromfile(os.path.join(data_dir, "attention_weights.bin"), dtype=np.float32).reshape(B, T, H)

# Compare
abs_diff = np.abs(out_c - out_ref)
max_diff = np.max(abs_diff)
print(f"Max absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {np.mean(abs_diff):.6f}")


print(out_c)
print("=========")
print(out_ref)

assert np.allclose(out_c, out_ref, atol=1e-4), "Mismatch!"
