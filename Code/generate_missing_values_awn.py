import numpy as np
import pandas as pd
import os

data = pd.read_csv("Data/awn/awn_norm.csv")

mask = ~np.isnan(data.to_numpy())

# for i in range(mask.shape[0]):
#     temp = mask[i].copy().reshape(-1, 2)
#     index = np.random.randint(0, temp.shape[0])
#     temp[index] = 0
#     mask[i] = temp.reshape(-1)
K = 7
rng = None
N = 67
L = 288
B, D = mask.shape
# if D != N * K:
#     raise ValueError(f"D must equal N*K (got D={D}, N={N}, K={K})")

# out = mask
num_blocks = (B + L - 1) // L
rng = np.random.default_rng(rng)

chosen = rng.integers(0, N, size=num_blocks)
for b in range(num_blocks):
    r0 = b * L
    r1 = min((b + 1) * L, B)
    idx = int(chosen[b])
    c0 = idx * K
    c1 = (idx + 1) * K
    mask[r0:r1, c0:c1] = 0
if not os.path.isdir("Data/awn/mask"):
    os.makedirs("Data/awn/mask")
np.savetxt("Data/awn/mask/awn_mask.csv", mask, fmt="%d", delimiter=",")