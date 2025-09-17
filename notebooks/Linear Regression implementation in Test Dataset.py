import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib, json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_and_prepare_test_fd001(
    test_path: str,
    rul_path: str,
    feature_cols: list,
    scaler
):
    """
    Returns:
      X_test_final  -> scaled features for the last-cycle row of each engine
      y_test_true   -> true RUL for those rows (from RUL_FD001.txt)
      test_last     -> the unscaled last-cycle dataframe (useful for debugging)
    """

    col_names = (
        ["engine_unit", "cycle"] +
        [f"operational_setting {i}" for i in range(1, 4)] +
        [f"sensor_measurement {i}" for i in range(1, 22)]
    )

    test_df = pd.read_csv(
        test_path, sep=r"\s+", header=None, names=col_names
    )

    drop_cols = [
        "sensor_measurement 1", "sensor_measurement 5", "sensor_measurement 6",
        "sensor_measurement 10", "sensor_measurement 16",
        "sensor_measurement 18", "sensor_measurement 19",
        "operational_setting 1", "operational_setting 2", "operational_setting 3",
    ]
    test_df = test_df.drop(columns=drop_cols)

    
    test_last = (
        test_df.sort_values(["engine_unit", "cycle"])
               .groupby("engine_unit", as_index=False)
               .tail(1)
               .reset_index(drop=True)
    )

    y_test_true = pd.read_csv(rul_path, header=None).iloc[:, 0].to_numpy()
    test_last = test_last.sort_values("engine_unit").reset_index(drop=True)
    test_last["RUL_true"] = y_test_true

    
    X_test_raw = test_last[feature_cols].copy()

    X_test_final = scaler.transform(X_test_raw)

    return X_test_final, y_test_true, test_last

reg = joblib.load("artifacts/lr_fd001.joblib")
scaler = joblib.load("artifacts/scaler_fd001.joblib")
with open("artifacts/feature_cols.json", "r") as f:
    feature_cols = json.load(f)

X_test_final, y_test_true, test_last = load_and_prepare_test_fd001(
    test_path=r"C:\Users\20243314\OneDrive - TU Eindhoven\Desktop\Datasets For Projects\Project 3 Predictive Maintenance NASA-CMAPSS\CMaps\test_FD001.txt",
    rul_path=r"C:\Users\20243314\OneDrive - TU Eindhoven\Desktop\Datasets For Projects\Project 3 Predictive Maintenance NASA-CMAPSS\CMaps\RUL_FD001.txt",
    feature_cols=feature_cols,
    scaler=scaler
)

y_test_pred = reg.predict(X_test_final)

#Plot the results.
plt.figure(figsize=(6,4))
plt.scatter(y_test_pred, y_test_true - y_test_pred, alpha=0.5)
plt.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'k--', lw=2, label = "Perfect Prediction")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("Linear Regression Predictions on Test Set")
plt.legend()
plt.show()


# Evaluate
mae  = mean_absolute_error(y_test_true, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
r2 = r2_score(y_test_true, y_test_pred)
print(f"FINAL (official test) — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R^2: {r2:.2f}")
