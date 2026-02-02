#!/usr/bin/env python3
"""
Debug script to verify NeuralNet functionality
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from part2_bp_implementation.NeuralNet import NeuralNet

# Set random seed for reproducibility
np.random.seed(42)

# Create simple test data
n_samples = 100
n_features = 10

# Generate simple linear relationship: y = 2*x1 + 3*x2 + noise
X = np.random.randn(n_samples, n_features)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.1

# Reshape y to (n_samples, 1)
y = y.reshape(-1, 1)

# Normalize y for training
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

print("Test data shapes:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")
print(f"y_scaled: {y_scaled.shape}")
print(f"y_mean: {y_mean:.4f}, y_std: {y_std:.4f}")

# Create and train neural network
nn = NeuralNet(
    layers=[n_features, 5, 1],
    learning_rate=0.01,
    momentum=0.0,
    fact='relu',
    l1_reg=0.0,
    l2_reg=0.0,
)

print("\nTraining neural network...")
train_losses, val_losses = nn.fit(
    X, y_scaled, epochs=50, validation_split=0.2
)

print(f"\nFinal training loss: {train_losses[-1]:.6f}")
print(f"Final validation loss: {val_losses[-1]:.6f}")

# Make predictions
y_pred_scaled = nn.predict(X)
y_pred = y_pred_scaled * y_std + y_mean

print(f"\nPrediction shape: {y_pred.shape}")
print(f"First 5 actual values: {y[:5].flatten()}")
print(f"First 5 predicted values: {y_pred[:5].flatten()}")

# Calculate metrics
mse = np.mean((y - y_pred) ** 2)
mae = np.mean(np.abs(y - y_pred))
print(f"\nMSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# Check if predictions are not constant
pred_std = np.std(y_pred)
print(f"Prediction std: {pred_std:.4f}")
if pred_std < 0.01:
    print("WARNING: Predictions are nearly constant!")
else:
    print("OK: Predictions have reasonable variance")

# Check if predictions follow the trend
correlation = np.corrcoef(y.flatten(), y_pred.flatten())[0, 1]
print(f"Correlation between y and y_pred: {correlation:.4f}")
if correlation < 0.1:
    print("WARNING: Low correlation between predictions and targets!")
else:
    print("OK: Predictions correlate with targets")