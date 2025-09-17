import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from loading_dataset_train_FD001 import get_df
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib, json, os 


#Import the dataframe from loading_dataset_train_FD001.py.
normalize_linear_regression_df = get_df()

print(normalize_linear_regression_df)
#Define features X and target Y.
X = normalize_linear_regression_df.drop(columns = ["engine_unit", "RUL", "cycle", "operational_setting 1", "operational_setting 2", "operational_setting 3"], errors = "ignore")
Y = normalize_linear_regression_df["RUL"]


#Split by engine_unit to avoid data leakage.
units = normalize_linear_regression_df["engine_unit"].unique()
train_units, test_units = train_test_split(units, test_size = 0.2, random_state = 42, shuffle = True)

#Create train and test sets based on engine_unit split. 
train_mask = normalize_linear_regression_df["engine_unit"].isin(train_units)

X_train = X[train_mask]
X_test = X[train_mask == False]
Y_train = Y[train_mask]
Y_test = Y[train_mask == False]

#Fit scaler on training data feature only, then transform both sets. Keep column names after scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X.columns, index = X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns, index = X_test.index)

#Create and fit the linear regression model.
reg = LinearRegression()
reg.fit(X_train, Y_train)

#Make predictions on both sets. 
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

#Add standard metrics to evaluate the model.
train_mae = mean_absolute_error(Y_train, y_train_pred)
test_mae = mean_absolute_error(Y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(Y_train, y_train_pred)) 
test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
train_r2 = r2_score(Y_train, y_train_pred)
test_r2 = r2_score(Y_test, y_test_pred)

plt.figure(figsize=(6,4))
plt.scatter(y_test_pred, Y_test - y_test_pred, alpha=0.5)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Predicted RUL")
plt.ylabel("Residual (True - Pred)")
plt.title("Residuals on Test Set")
plt.show()

print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
print(f"Train R^2: {train_r2:.2f}, Test R^2: {test_r2:.2f}")    

os.makedirs("artifacts", exist_ok = True)

joblib.dump(reg, "artifacts/lr_fd001.joblib")
joblib.dump(scaler, "artifacts/scaler_fd001.joblib")

feature_cols = X_train.columns.tolist()  
with open("artifacts/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)