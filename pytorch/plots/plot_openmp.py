import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory")
args = parser.parse_args()
results_dir = args.results_dir

df = pd.read_csv(os.path.join(results_dir, 'opnemmp_test.csv'))

plt.plot(df["Threads"], df["FLOP/s"] / 1e9, marker='o')
plt.xlabel("Number of Threads")
plt.ylabel("FLOP/s (GFLOP/s)")
plt.title("Matrix Multiplication Performance vs Threads")
plt.grid(True)
plt.show()