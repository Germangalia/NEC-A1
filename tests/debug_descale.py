#!/usr/bin/env python3
"""
Debug script to check descaling
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

print("Original y statistics:")
print(f"  Mean: {y_train_val.mean():.4f}")
print(f"  Std: {y_train_val.std():.4f}")
print(f"  Min: {y_train_val.min():.4f}")
print(f"  Max: {y_train_val.max():.4f}")

y_train_val_scaled = (y_train_val - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

print("\nScaled y statistics:")
print(f"  Mean: {y_train_val_scaled.mean():.4f}")
print(f"  Std: {y_train_val_scaled.std():.4f}")
print(f"  Min: {y_train_val_scaled.min():.4f}")
print(f"  Max: {y_train_val_scaled.max():.4f}")

print("\nScaling parameters:")
print(f"  y_mean: {y_mean} (shape: {y_mean.shape})")
print(f"  y_std: {y_std} (shape: {y_std.shape})")

# Train a simple model
n_features = X_train_val.shape[1]
nn = NeuralNet(
    layers=[n_features, 10, 1],
    learning_rate=0.01,
    momentum=0.0,
    fact='sigmoid',
    l1_reg=0.0,
    l2_reg=0.0,
)

print("\nTraining...")
train_losses, val_losses = nn.fit(
    X_train_val,
    y_train_val_scaled,
    epochs=20,
    validation_split=0.2,
)

# Predictions
y_pred_scaled = nn.predict(X_test)
print(f"\nPrediction shape: {y_pred_scaled.shape}")
print(f"Prediction statistics (scaled):")
print(f"  Mean: {y_pred_scaled.mean():.4f}")
print(f"  Std: {y_pred_scaled.std():.4f}")
print(f"  Min: {y_pred_scaled.min():.4f}")
print(f"  Max: {y_pred_scaled.max():.4f}")

if y_pred_scaled.ndim == 1:
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)

# Check descaling formula
print("\nDescaling check:")
print(f"y_pred_scaled shape: {y_pred_scaled.shape}")
print(f"y_std shape: {y_std.shape}")
print(f"y_mean shape: {y_mean.shape}")

# Manual descaling with broadcasting check
y_pred_original = y_pred_scaled * y_std + y_mean

print(f"\nDescaled prediction shape: {y_pred_original.shape}")
print(f"Descaled prediction statistics:")
print(f"  Mean: {y_pred_original.mean():.4f}")
print(f"  Std: {y_pred_original.std():.4f}")
print(f"  Min: {y_pred_original.min():.4f}")
print(f"  Max: {y_pred_original.max():.4f}")

print(f"\nActual y_test statistics:")
print(f"  Mean: {y_test.mean():.4f}")
print(f"  Std: {y_test.std():.4f}")
print(f"  Min: {y_test.min():.4f}")
print(f"  Max: {y_test.max():.4f}")

# First few values
print(f"\nFirst 5 values:")
print(f"  Actual: {y_test[:5].flatten()}")
print(f"  Predicted: {y_pred_original[:5].flatten()}")

# Check if there's a constant bias
print(f"\nChecking for constant bias:")
print(f"  Mean difference: {(y_pred_original.mean() - y_test.mean()):.4f}")
print(f"  Pred mean - actual mean: {(y_pred_original.mean() - y_test.mean()):.4f}")