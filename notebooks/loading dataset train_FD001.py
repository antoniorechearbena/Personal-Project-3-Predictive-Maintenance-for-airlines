import pandas as pd

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

#Finding RUL (Remaining Useful Life) for each engine unit in the training dataset.
#Group by engine_unit, find the max cycle for each engine unit, and substract the current cycle from the max cycle to get RUL.
train_FD001_df["RUL"] = train_FD001_df.groupby("engine_unit")["cycle"].transform(max) - train_FD001_df["cycle"]

print(train_FD001_df)