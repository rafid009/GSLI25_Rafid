import numpy as np
import pandas as pd

data = pd.read_csv("Data/nacse/nacse_norm.csv")

mask = ~np.isnan(data)

for i in range(mask.shape[0]):
    temp = mask[i].copy().reshape(-1, 2)
    index = np.random.randint(0, temp.shape[0])
    temp[index] = 0
    mask[i] = temp

np.savetxt("Data/nacse/mask/nacse_mask.csv", mask, fmt="%d", delimiter=",")