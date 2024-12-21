import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from tensorflow.keras.metrics import MeanAbsoluteError # type: ignore
from sklearn.model_selection import KFold
import sys, os

def train_k_fold_nn():
    sys.stdout = open(os.devnull, "w")

    # Load your data
    df = pd.read_csv('processed_stats.csv')

    # Define features and target
    features = ['FGM', 'FGA', 'FG_PCT', 'FG3A', 'FG3M', 'FG3_PCT', 
                'FTM', 'OP_DRTG', 'OP_PACE', 'FGM_FGA_Interaction', 'FTM_FG_PCT_Interaction']

    target = 'Points'

    X = df[features].values
    y = df[target].values
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # Initialize scaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    import joblib
    joblib.dump(scaler, 'scaler.pkl')

    # Initialize lists to store results
    fold_mse = []
    fold_mae = []

    # Define model creation function
    def create_model():
        model = Sequential([
        Dense(64, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])

    # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )  # Regularization strength, equivalent to kernel_regularizer in NN
        return model
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Split the data into training and validation sets
        X_train, X_test = X[train_idx], X[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]

        # Create a new instance of the model
        model = create_model()

        # Train the model
        history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=120,
        batch_size=2,
        verbose=0
        )

        # Evaluate the model on the validation data
        test_loss, test_mae = model.evaluate(X_test, y_test)

        fold_mse.append(test_loss)
        fold_mae.append(test_mae)

    # Save the final model (coefficients and intercept for Ridge regression)
    sys.stdout = sys.__stdout__
    model.save('parlay_prophet_model_nn.keras')
    return np.mean(fold_mae)
