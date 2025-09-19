import pandas as pd
import matplotlib.pyplot as plt
from loading_dataset_train_FD001 import get_df
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, os, json

#Import the dataframe from loading_dataset_train_FD001.py.
train_df = get_df()

#Define features X and target Y.
X = train_df.drop(columns = ["engine_unit", "RUL", "cycle", "operational_setting 1", "operational_setting 2", "operational_setting 3"], errors = "ignore")
Y = train_df["RUL"]

#Split by engine_unit to avoid data leakage. 
units = train_df["engine_unit"].unique()
train_units, test_units = train_test_split(units, test_size = 0.2, random_state = 42, shuffle = True)

train_mask = train_df["engine_unit"].isin(train_units)
X_train, X_test = X[train_mask], X[train_mask == False]
Y_train, Y_test = Y[train_mask], Y[train_mask == False]

#Scale features. Random forest is not sensitive to feature scaling, but makes comparison with linear regression easier.
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns, index = X_test.index)

#Train Random Forest Model
rf = RandomForestRegressor(
    n_estimators = 1000,
    max_depth = 50,
    random_state = 42,
    n_jobs = -1,
    min_samples_leaf = 100,
    max_features = "sqrt"

)
rf.fit(X_train, Y_train)

#Evaluate based on train/test split.
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

train_mae = mean_absolute_error(Y_train, y_train_pred)
test_mae = mean_absolute_error(Y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(Y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
train_r2 = r2_score(Y_train, y_train_pred)
test_r2 = r2_score(Y_test, y_test_pred)

#Build plot to visualize residuals.


#Show results for Random Forest on Train dataset.




#Start with model testing on official test set.
def load_and_prepare_test_fd001(test_path, rul_path, feature_cols, scaler):
    col_names = (
        ["engine_unit", "cycle"] +
        [f"operational_setting {i}" for i in range(1, 4)] +
        [f"sensor_measurement {i}" for i in range(1, 22)]
    )

    test_df = pd.read_csv(test_path, sep = r"\s+", header = None, names = col_names)

    drop_cols = [
        "sensor_measurement 1", "sensor_measurement 5", "sensor_measurement 6",
        "sensor_measurement 10", "sensor_measurement 16",
        "sensor_measurement 18", "sensor_measurement 19",
        "operational_setting 1", "operational_setting 2", "operational_setting 3",
    ]

    test_df = test_df.drop(columns = drop_cols, errors = "ignore")

    #Keep only the last cycle for each engine unit. 
    test_last = (
        test_df.sort_values(["engine_unit", "cycle"])   
        .groupby("engine_unit", as_index = False)
        .tail(1)
        .reset_index(drop = True)
    )

    y_test_true = pd.read_csv(rul_path, header = None).iloc[:,0].to_numpy()
    test_last = test_last.sort_values("engine_unit").reset_index(drop = True)
    test_last["RUL_true"] = y_test_true

    X_test_raw = test_last[feature_cols].copy()
    X_test_final = pd.DataFrame(scaler.transform(X_test_raw), columns = feature_cols, index = X_test_raw.index)

    return X_test_final, y_test_true, test_last


#Invoce the function to load and prepare the test set.
feature_cols = X_train.columns.tolist()

test_path = r"C:\Users\20243314\OneDrive - TU Eindhoven\Desktop\Datasets For Projects\Project 3 Predictive Maintenance NASA-CMAPSS\CMaps\test_FD001.txt"
rul_path = r"C:\Users\20243314\OneDrive - TU Eindhoven\Desktop\Datasets For Projects\Project 3 Predictive Maintenance NASA-CMAPSS\CMaps\RUL_FD001.txt"


X_test_final, y_test_true, test_last = load_and_prepare_test_fd001(
    test_path = test_path,
    rul_path = rul_path,
    feature_cols = feature_cols,
    scaler = scaler
)

y_pred_official = rf.predict(X_test_final)

#Metrics and plots.
off_mae  = mean_absolute_error(y_test_true, y_pred_official)
off_rmse = np.sqrt(mean_squared_error(y_test_true, y_pred_official))
off_r2   = r2_score(y_test_true, y_pred_official)
print(f"\nOFFICIAL TEST  —  MAE: {off_mae:.2f}, RMSE: {off_rmse:.2f}, R²: {off_r2:.2f}")

# Parity + residuals plots (optional)
plt.figure(figsize=(6,6))
plt.scatter(y_test_true, y_pred_official, alpha=0.6)
mn, mx = y_test_true.min(), y_test_true.max()
plt.plot([mn, mx], [mn, mx], "k--", lw=2, label="Perfect Prediction")
plt.xlabel("True RUL"); plt.ylabel("Predicted RUL"); plt.title("RF – True vs Pred (Official Test)")
plt.legend(); plt.show()

plt.figure(figsize=(10,5))
plt.scatter(y_pred_official, y_test_true - y_pred_official, alpha=0.5)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Predicted RUL"); plt.ylabel("Residual (True - Pred)")
plt.title("RF – Residuals on Official Test"); plt.show()