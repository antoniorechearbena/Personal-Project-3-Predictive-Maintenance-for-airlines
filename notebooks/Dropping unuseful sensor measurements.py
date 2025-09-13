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

#Drop the unuseful sensor measurements: sensor_measurement 1, 5, 6, 10, 16, 18, 19.
train_FD001_df = train_FD001_df.drop(columns = [
    "sensor_measurement 1", 
    "sensor_measurement 5", 
    "sensor_measurement 6", 
    "sensor_measurement 10", 
    "sensor_measurement 16", 
    "sensor_measurement 18", 
    "sensor_measurement 19"
])

print(train_FD001_df)