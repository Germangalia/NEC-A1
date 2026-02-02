#!/usr/bin/env python3
"""
Debug script testing different configurations
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

# Ensure targets are 2D column vectors (n_samples, 1)
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

# Test different configurations
configs = [
    {"hidden_layers": [10], "epochs": 20, "learning_rate": 0.01, "momentum": 0.0, "activation": "sigmoid"},
    {"hidden_layers": [10], "epochs": 20, "learning_rate": 0.01, "momentum": 0.0, "activation": "tanh"},
    {"hidden_layers": [10], "epochs": 20, "learning_rate": 0.01, "momentum": 0.0, "activation": "relu"},
]

for i, config in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Configuration {i+1}/{len(configs)}")
    print(f"{'='*60}")
    print(f"Hidden layers: {config['hidden_layers']}")
    print(f"Activation: {config['activation']}")
    print(f"Epochs: {config['epochs']}")

    layers = [n_features] + config["hidden_layers"] + [1]

    # Instantiate the custom BP model
    nn = NeuralNet(
        layers=layers,
        learning_rate=config["learning_rate"],
        momentum=config["momentum"],
        fact=config["activation"],
        l1_reg=0.0,
        l2_reg=0.0,
    )

    # Train
    train_losses, val_losses = nn.fit(
        X_train_val,
        y_train_val_scaled,
        epochs=config["epochs"],
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

    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Correlation: {correlation:.4f}")

    # Check if loss decreased significantly
    loss_decrease = (train_losses[0] - train_losses[-1]) / train_losses[0]
    print(f"Loss decrease: {loss_decrease:.2%}")