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

#Create a dataframe for engine unit 1
engine_1_df = train_FD001_df[train_FD001_df["engine_unit"] == 1]
print(engine_1_df)

#Plot the sensor measurements for engine unit 1 across all cycles.
plt.figure(figsize=(15,10))
plt.plot(engine_1_df["cycle"], engine_1_df["sensor_measurement 8"], label = "Sensor Measurement 8")
plt.xlabel("Cycle")
plt.ylabel("Sensor Measurement 8")
plt.title("Sensor Measurement 8 across Cycles for Engine Unit 1")
plt.show()