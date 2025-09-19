#  Predictive Maintenance on NASA CMAPSS (FD001)

This project focuses on predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA CMAPSS dataset (FD001 subset). Accurate RUL predictions are critical in aviation maintenance planning, helping reduce costs, increase safety, and avoid unexpected failures.  

---

##  Dataset

- **Source:** NASA CMAPSS (FD001)  
- **Train set:** Engine sensor data until failure  
- **Test set:** Engine sensor data truncated before failure + true RULs provided separately  
- **Features:** Operational settings and 21 sensor measurements  
- **Target:** Remaining Useful Life (RUL)  

---

## Models Implemented

### 1. Linear Regression (Baseline)
- **What it is:** A simple model that assumes a straight-line relationship between sensors and RUL.  
- **Why:** Serves as a baseline for comparison.  
- **Result:** Poor performance (high MAE/RMSE, negative R²).  

### 2. Random Forest
- **What it is:** An ensemble of decision trees, averaging multiple independent trees.  
- **Why:** Handles non-linearities and noise better than linear regression.  
- **Result:** MAE = 23.1, RMSE = 30.3, R² = 0.47.  

### 3. XGBoost (Gradient Boosting Trees)
- **What it is:** A boosting method that builds trees sequentially, each one correcting errors from the previous.  
- **Why:** More flexible and powerful, widely used in industry and Kaggle competitions.  
- **Feature engineering added:**  
  - Rolling averages of sensor values (short-term trends)  
  - First differences (rate of change)  
  - Normalized cycles (relative engine age)  
- **Result (best config):** MAE = 19.9, RMSE = 23.9, R² = 0.67.  

---

##  Results Summary

| Model              | MAE ↓  | RMSE ↓ | R² ↑  |
|--------------------|--------|--------|-------|
| Linear Regression  | ~50+   | Very high | Negative |
| Random Forest      | 23.1   | 30.3   | 0.47  |
| XGBoost (baseline) | 24.4   | 28.5   | 0.53  |
| XGBoost (tuned)    | 19.9   | 23.9   | 0.67  |

 **Key Insight:** XGBoost consistently outperformed Random Forest, especially after feature engineering and tuning.

---

##  Project Workflow

1. **Data Loading:** Preprocessed CMAPSS FD001 into pandas DataFrames.  
2. **Feature Engineering:** Added rolling means, sensor differences, and cycle normalization.  
3. **Model Training:**  
   - Linear Regression with scikit-learn  
   - Random Forest with scikit-learn  
   - XGBoost with xgboost library  
4. **Hyperparameter Tuning:** Used cross-validation + early stopping to find the best number of boosting rounds.  
5. **Evaluation:** Compared MAE, RMSE, and R² on the official FD001 test set.  
6. **Visualization:**  
   - True vs Predicted RUL parity plots  
   - Residual plots  

---

##  Example Plots

**XGBoost Parity Plot (Official Test)**  
(Add your screenshot here)

**XGBoost Residuals (Official Test)**  
(Add your screenshot here)

---

##  Lessons Learned

- Linear models are too simple for this problem.  
- Tree ensembles (RF, XGB) capture non-linear sensor interactions effectively.  
- Feature engineering (rolling stats, cycle normalization) is crucial for extracting degradation signals.  
- Cross-validation and early stopping prevent overfitting.  
- Slight variations in results are expected due to stochastic sampling in ensembles, but the overall trend (XGB > RF > Linear) remained consistent.  

---

##  Next Steps

- Extend feature engineering with domain-specific health indicators.  
- Implement **LSTM/GRU deep learning models** to capture sequential dependencies in engine degradation.  
- Explore model ensembling (XGBoost + RF + neural nets).  

---

##  Tech Stack

- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- XGBoost  

---


## Author
Antonio Reche Cazorla.
Linkedln: www.linkedin.com/in/antoniorechecazorla

