import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys, os

def train_k_fold():
    sys.stdout = open(os.devnull, "w")

    # Load your data
    df = pd.read_csv('processed_stats.csv')

    # Define features and target
    features = ['FGM', 'FGA', 'FG_PCT', 'FG3A', 'FG3M', 'FG3_PCT', 
                'FTM', 'OP_DRTG', 'OP_PACE', 'FGM_FGA_Interaction', 'FTM_FG_PCT_Interaction']

    target = 'Points'

    X = df[features].values
    y = df[target].values

    # Initialize scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    import joblib
    joblib.dump(scaler, 'scaler.pkl')

    # Define Leave-One-Out Cross-Validation (LOOCV)
    loo = LeaveOneOut()

    # Initialize lists to store results
    fold_mse = []
    fold_mae = []

    # Define model creation function
    def create_model():
        model = Ridge(alpha=0.01)  # Regularization strength, equivalent to kernel_regularizer in NN
        return model

    # Perform LOOCV
    for train_idx, val_idx in loo.split(X):
        # Split the data into training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create a new instance of the model
        model = create_model()

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model on the validation data
        y_val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        fold_mse.append(val_mse)
        fold_mae.append(val_mae)

    # Save the final model (coefficients and intercept for Ridge regression)
    np.savez('parlay_prophet_model_k_fold.npz', coefficients=model.coef_, intercept=model.intercept_)
    sys.stdout = sys.__stdout__

    # print(f"Average MSE: {np.mean(fold_mse):.4f}")
    # print(f"Average MAE: {np.mean(fold_mae):.4f}")

    return np.mean(fold_mae)
