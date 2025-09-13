import numpy as np
import pandas as pd
import os

# data = pd.read_csv()
# locations = pd.read_csv()

data = pd.read_csv("Data/OR_temps.csv")
data.replace(-9999.0, np.nan, inplace=True)
data.drop('index', axis=1, inplace=True)
locations = pd.read_csv("Data/OR_temps_loc.csv")
# print(locations.to_numpy()[0, 1:])
df_locs = pd.DataFrame(locations.to_numpy()[0, 1:].reshape(-1, 3)[:, :-1], columns=['longitude', 'latitude'])
df_locs.to_csv("Data/nacse_stations.csv", index=False)

df_norm = data.to_numpy().copy().reshape(-1, 2)

print(f"df norm: {df_norm}")
mean = np.nanmean(df_norm, axis=0)
std = np.nanstd(df_norm, axis=0)

df_norm = (df_norm - mean) / std

df_norm = pd.DataFrame(df_norm.reshape(-1, 179*2))
if not os.path.isdir("Data/nacse"):
    os.makedirs("Data/nacse")
df_norm.to_csv("Data/nacse/nacse_norm.csv", index=False)
