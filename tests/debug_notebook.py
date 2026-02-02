#!/usr/bin/env python3
"""
Debug script using the same data as the notebook
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

print(f"Train+val X shape: {X_train_val.shape}, y shape: {y_train_val.shape}")
print(f"Test      X shape: {X_test.shape}, y shape: {y_test.shape}")

# Standardize the target variable y according to BP.v2
y_mean = y_train_val.mean(axis=0)
y_std = y_train_val.std(axis=0)
y_std[y_std == 0] = 1.0  # safety

y_train_val_scaled = (y_train_val - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

print("\nTarget standardization:")
print(f"  y_mean: {y_mean}")
print(f"  y_std : {y_std}")
print(f"  y_train_val_scaled mean: {y_train_val_scaled.mean():.4f}, std: {y_train_val_scaled.std():.4f}")
print(f"  y_test_scaled mean: {y_test_scaled.mean():.4f}, std: {y_test_scaled.std():.4f}")

# Test with a simple configuration (the best one from the notebook)
config = {"hidden_layers": [10], "epochs": 50, "learning_rate": 0.01, "momentum": 0.0, "activation": "sigmoid"}

n_features = X_train_val.shape[1]
hidden_layers = config["hidden_layers"]
epochs = config["epochs"]
lr = config["learning_rate"]
momentum = config["momentum"]
activation = config["activation"]

layers = [n_features] + hidden_layers + [1]

print(f"\nTraining configuration:")
print(f"  Layers: {layers}")
print(f"  Epochs: {epochs}")
print(f"  Learning rate: {lr}")
print(f"  Momentum: {momentum}")
print(f"  Activation: {activation}")

# Instantiate the custom BP model
nn = NeuralNet(
    layers=layers,
    learning_rate=lr,
    momentum=momentum,
    fact=activation,
    l1_reg=0.0,
    l2_reg=0.0,
)

print("\nTraining neural network...")
train_losses, val_losses = nn.fit(
    X_train_val,
    y_train_val_scaled,
    epochs=epochs,
    validation_split=0.2,
)

print(f"\nFinal training loss: {train_losses[-1]:.6f}")
print(f"Final validation loss: {val_losses[-1]:.6f}")

# Predictions on test (scaled)
y_pred_scaled = nn.predict(X_test)
print(f"\nPrediction shape: {y_pred_scaled.shape}")

if y_pred_scaled.ndim == 1:
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)

# Descale predictions back to original units
y_pred_original = y_pred_scaled * y_std + y_mean

print(f"Descaled prediction shape: {y_pred_original.shape}")
print(f"First 5 actual values: {y_test[:5].flatten()}")
print(f"First 5 predicted values: {y_pred_original[:5].flatten()}")

# Compute metrics
mse = float(np.mean((y_test - y_pred_original) ** 2))
mae = float(np.mean(np.abs(y_test - y_pred_original)))
eps = 1e-8
mape = float(np.mean(np.abs((y_test - y_pred_original) / (y_test + eps))) * 100.0)

print(f"\nMetrics:")
print(f"  MSE: {mse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  MAPE: {mape:.4f}%")

# Check if predictions are not constant
pred_std = np.std(y_pred_original)
print(f"\nPrediction std: {pred_std:.4f}")
if pred_std < 0.01:
    print("WARNING: Predictions are nearly constant!")
else:
    print("OK: Predictions have reasonable variance")

# Check if predictions follow the trend
correlation = np.corrcoef(y_test.flatten(), y_pred_original.flatten())[0, 1]
print(f"Correlation between y_test and y_pred: {correlation:.4f}")
if correlation < 0.1:
    print("WARNING: Low correlation between predictions and targets!")
else:
    print("OK: Predictions correlate with targets")