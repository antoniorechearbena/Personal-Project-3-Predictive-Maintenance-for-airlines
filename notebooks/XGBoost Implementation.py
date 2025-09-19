import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from loading_dataset_train_FD001 import get_df


#Features to improve the performance's model.
#Rolling Statistics
def add_rolling_means(df: pd.DataFrame, sensor_cols: list, window: int = 5):
    out = df.copy()
    for col in sensor_cols:
        out[f"{col}_mean_{window}"] = (
            out.groupby("engine_unit")[col]
                .rolling(window = window, min_periods = 1)
                .mean()
                .reset_index(level = 0, drop = True)
        )
    return out


#Rate of Change
def add_differences(df: pd.DataFrame, sensor_cols: list):
    out = df.copy()
    for col in sensor_cols:
        out[f"{col}_diff"] = (
            out.groupby("engine_unit")[col].diff().fillna(0.0)
        )
    return out

#Normalized Cycle 
def add_cycle_normalized(df: pd.DataFrame):
    out = df.copy()
    out["cycle_norm"] = out.groupby("engine_unit")["cycle"].transform(lambda s: s / s.max())
    return out
    
#Drop near constants.
def drop_near_constant(df: pd.DataFrame,threshold: float = 1e-8):
    variances = df.var(numeric_only = True)
    keep = variances[variances > threshold].index.tolist()
    keep = list(set(keep) | set([c for c in df.columns if df[c].dtype == "O"]))
    return df[keep]



#Load train set
train_df = get_df()

sensor_cols = [c for c in train_df.columns if c.startswith("sensor_measurement")]
base_drop = ["engine_unit", "RUL", "cycle", "operational_setting 1", "operational_setting 2", "operational_setting 3"]

fe_df = add_rolling_means(train_df, sensor_cols, window = 5)
fe_df = add_differences(fe_df, sensor_cols)
fe_df = add_cycle_normalized(fe_df)
fe_df = drop_near_constant(fe_df)
#Define features and target
X = fe_df.drop(columns = base_drop, errors = "ignore")
Y = fe_df["RUL"]

#Split by engine_unit to avoid leakage
units = fe_df["engine_unit"].unique()
train_units, val_units = train_test_split(units, test_size=0.2, random_state=42, shuffle=True)

mask = fe_df["engine_unit"].isin(train_units)
X_train, X_val = X[mask], X[~mask]
Y_train, Y_val = Y[mask], Y[~mask]

#Early stopping via xgb.cv
def cv_best_n(params, X_train, y_train, num_boost_round=150, nfold=5, es_rounds=100, seed=42):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    cv = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        early_stopping_rounds=es_rounds,
        verbose_eval=False,
        seed=seed
    )
    best_n = cv.shape[0]
    best_rmse = float(cv["test-rmse-mean"].iloc[-1])
    return best_n, best_rmse

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.01,               
    "max_depth": 20,
    "min_child_weight": 10,
    "subsample": 0.4,
    "colsample_bytree": 0.4,
    "gamma": 0.5,
    "reg_alpha": 0.0,
    "reg_lambda": 5.0,
    "seed": 42,
}

best_n, cv_rmse = cv_best_n(xgb_params, X_train, Y_train)
print(f"[XGB] CV best_n={best_n}, CV test-RMSE={cv_rmse:.3f}")

xgb_model = XGBRegressor(
    n_estimators=best_n,
    learning_rate=xgb_params["eta"],
    max_depth=xgb_params["max_depth"],
    min_child_weight=xgb_params["min_child_weight"],
    subsample=xgb_params["subsample"],
    colsample_bytree=xgb_params["colsample_bytree"],
    gamma=xgb_params["gamma"],
    reg_alpha=xgb_params["reg_alpha"],
    reg_lambda=xgb_params["reg_lambda"],
    objective="reg:squarederror",
    eval_metric="rmse",
    n_jobs=-1,
    random_state=42,
)
xgb_model.fit(X_train, Y_train)

#Internal evaluation
y_tr_pred = xgb_model.predict(X_train)
y_val_pred = xgb_model.predict(X_val)

tr_mae  = mean_absolute_error(Y_train, y_tr_pred)
va_mae  = mean_absolute_error(Y_val,   y_val_pred)
tr_rmse = np.sqrt(mean_squared_error(Y_train, y_tr_pred))
va_rmse = np.sqrt(mean_squared_error(Y_val,   y_val_pred))
tr_r2   = r2_score(Y_train, y_tr_pred)
va_r2   = r2_score(Y_val,   y_val_pred)

#print(f"XGB (internal) -> Train MAE: {tr_mae:.2f}, RMSE: {tr_rmse:.2f}, R²: {tr_r2:.2f}")
#print(f"XGB (internal) ->  Test MAE: {va_mae:.2f}, RMSE: {va_rmse:.2f}, R²: {va_r2:.2f}")




#XGBoost Implementation on Test Dataset
def load_official_fd001(test_path: str, rul_path: str, feature_cols: list):
    col_names = (
        ["engine_unit", "cycle"]
        + [f"operational_setting {i}" for i in range(1, 3 + 1)]
        + [f"sensor_measurement {i}" for i in range(1, 22)]
    )
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)

    drop_cols = ["engine_unit", "cycle",
                 "operational_setting 1", "operational_setting 2", "operational_setting 3"]
    
    
    sensors_t = [c for c in test_df.columns if c.startswith("sensor_measurement")]
    fe_t = add_rolling_means(test_df, sensors_t, window=5)
    fe_t = add_differences(fe_t, sensors_t)
    fe_t = add_cycle_normalized(fe_t)
    fe_t = drop_near_constant(fe_t)

    last_rows = (
        fe_t.sort_values(["engine_unit", "cycle"])
               .groupby("engine_unit", as_index=False)
               .tail(1)
               .reset_index(drop=True)
    )

    
    X_official = last_rows.drop(columns=drop_cols, errors="ignore")
    X_official = X_official.reindex(columns = feature_cols, fill_value=0.0)
    y_true = pd.read_csv(rul_path, header=None).iloc[:, 0].to_numpy()
    return X_official, y_true

feature_cols = X_train.columns.tolist()
test_path = r"C:\Users\20243314\OneDrive - TU Eindhoven\Desktop\Datasets For Projects\Project 3 Predictive Maintenance NASA-CMAPSS\CMaps\test_FD001.txt"
rul_path  = r"C:\Users\20243314\OneDrive - TU Eindhoven\Desktop\Datasets For Projects\Project 3 Predictive Maintenance NASA-CMAPSS\CMaps\RUL_FD001.txt"

X_official, y_true_off = load_official_fd001(test_path, rul_path, feature_cols)
y_pred_off = xgb_model.predict(X_official)

#Results + Plotting
off_mae  = mean_absolute_error(y_true_off, y_pred_off)
off_rmse = np.sqrt(mean_squared_error(y_true_off, y_pred_off))
off_r2   = r2_score(y_true_off, y_pred_off)
print(f"\nOFFICIAL TEST  —  MAE: {off_mae:.2f}, RMSE: {off_rmse:.2f}, R²: {off_r2:.2f}")

plt.figure(figsize=(6,6))
plt.scatter(y_true_off, y_pred_off, alpha=0.6)
mn, mx = y_true_off.min(), y_true_off.max()
plt.plot([mn, mx], [mn, mx], "k--", lw=2, label="Perfect Prediction")
plt.xlabel("True RUL"); plt.ylabel("Predicted RUL")
plt.title("XGB – True vs Pred (Official Test)"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,5))
plt.scatter(y_pred_off, y_true_off - y_pred_off, alpha=0.5)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Predicted RUL"); plt.ylabel("Residual (True - Pred)")
plt.title("XGB – Residuals on Official Test"); plt.tight_layout(); plt.show()
