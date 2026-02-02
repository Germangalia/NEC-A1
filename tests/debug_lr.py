#!/usr/bin/env python3
"""
Debug script with different learning rates
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from part2_bp_implementation.NeuralNet import NeuralNet

np.random.seed(0)

# Load preprocessed datasets generated in Part 1
X_train_val = np.load("./dataset/preprocessed/X_train_val.npy")
y_train_val = np.load("./dataset/preprocessed/y_train_val.npy")
X_test = np.load("./dataset/preprocessed/X_test.npy")
y_test = np.load("./dataset/preprocessed/y_test.npy")

# Ensure targets are 2D column vectors
if y_train_val.ndim == 1:
    y_train_val = y_train_val.reshape(-1, 1)
if y_test.ndim == 1:
    y_test = y_test.reshape(-1, 1)

# Standardize the target variable y
y_mean = y_train_val.mean(axis=0)
y_std = y_train_val.std(axis=0)
y_std[y_std == 0] = 1.0

y_train_val_scaled = (y_train_val - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

n_features = X_train_val.shape[1]

# Test different learning rates
learning_rates = [0.001, 0.01, 0.1, 1.0]

for lr in learning_rates:
    print(f"\n{'='*60}")
    print(f"Learning Rate: {lr}")
    print(f"{'='*60}")

    nn = NeuralNet(
        layers=[n_features, 10, 1],
        learning_rate=lr,
        momentum=0.0,
        fact='sigmoid',
        l1_reg=0.0,
        l2_reg=0.0,
    )

    # Train
    train_losses, val_losses = nn.fit(
        X_train_val,
        y_train_val_scaled,
        epochs=20,
        validation_split=0.2,
    )

    # Predict
    y_pred_scaled = nn.predict(X_test)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_pred_original = y_pred_scaled * y_std + y_mean

    # Metrics
    mse = float(np.mean((y_test - y_pred_original) ** 2))
    mae = float(np.mean(np.abs(y_test - y_pred_original)))
    correlation = np.corrcoef(y_test.flatten(), y_pred_original.flatten())[0, 1]
    pred_std = np.std(y_pred_original)

    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Prediction std: {pred_std:.4f}")
    print(f"Correlation: {correlation:.4f}")