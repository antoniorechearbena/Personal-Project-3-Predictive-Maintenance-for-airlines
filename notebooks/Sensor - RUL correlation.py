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


#Add RUL column to the dataframe.
train_FD001_df["RUL"] = train_FD001_df.groupby("engine_unit")["cycle"].transform("max") - train_FD001_df["cycle"]

#Keep only the sensor measurements and RUL columns for correlation analysis. 
correlation_df = train_FD001_df.drop(columns = ["engine_unit", "cycle", "operational_setting 1", "operational_setting 2", "operational_setting 3"])

#Calculate the correlation matrix, using Pearson correlation coefficient.
correlation_matrix = correlation_df.corr(method="pearson")
print(correlation_matrix)

#Plot the correlation matrix as a barchart.
plt.figure(figsize=(12,8))
correlation_with_RUL = correlation_matrix["RUL"].drop("RUL")
correlation_with_RUL.plot(kind="bar", color="skyblue", edgecolor="black")
plt.xlabel("Sensor Measurements")
plt.ylabel("Correlation with RUL")
plt.title("Correlation of Sensor Measurements with Remaining Useful Life (RUL)", fontweight ="bold", fontsize = 18 )
plt.axhline(0, color="black", linewidth=0.8)       
plt.show()


