import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#First define the column names for the dataset. There are 26 columns in total. We are going to use the following format: 2 + 3 + 21 = 26. 
col_names = (
    ["engine_unit", "cycle"] + 
    [f"operational_setting {i}" for i in range(1,4)] + 
    [f"sensor_measurement {i}" for i in range(1,22)]
)

#Load the training dataset
train_FD001_df = pd.read_csv(
    r"C:\Users\20243314\OneDrive - TU Eindhoven\Desktop\Datasets For Projects\Project 3 Predictive Maintenance NASA-CMAPSS\CMaps\train_FD001.txt",
    sep = r"\s+",
    header = None,
    names = col_names
)


#Group by engine_unit, find the max cycle for each engine unit, and substract the current cycle from the max cycle to get RUL. For each maximum cycle, collect them in a list. 
max_cycles = train_FD001_df.groupby("engine_unit")["cycle"].max().tolist()


#Plot a histogram of the max cycles per each engine unit.
plt.hist(max_cycles, bins=30, edgecolor="black", rwidth=0.8)
plt.xlabel("Engine lifetime (max cycles)")
plt.ylabel("Number of engines")
plt.title("Distribution of Engine Lifetimes in Train_FD001", fontweight ="bold", fontsize = 18 )
plt.show()
