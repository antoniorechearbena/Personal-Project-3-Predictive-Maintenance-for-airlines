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
    n_estimators = 500,
    max_depth = None,
    random_state = 42,
    n_jobs = -1

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
plt.figure(figsize=(6,4))
plt.scatter(y_test_pred, Y_test - y_test_pred, alpha=0.5)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Predicted RUL")
plt.ylabel("True Pred RUL")
plt.title("Random Forest Residuals")
plt.show()

print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
print(f"Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}")


