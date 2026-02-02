#!/usr/bin/env python3
"""
Debug script to check forward propagation dimensions
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from part2_bp_implementation.NeuralNet import NeuralNet

np.random.seed(0)

# Create simple test data
n_samples = 5
n_features = 10

X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples, 1)

print("Input shapes:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

# Create neural network
nn = NeuralNet(
    layers=[n_features, 5, 1],
    learning_rate=0.01,
    momentum=0.0,
    fact='relu',
    l1_reg=0.0,
    l2_reg=0.0,
)

print("\nNetwork architecture:")
print(f"Layers: {nn.n}")
print(f"Number of layers: {nn.L}")

print("\nWeight matrix shapes:")
for l in range(1, nn.L):
    print(f"w[{l}]: {nn.w[l].shape}")

print("\nBias shapes:")
for l in range(nn.L):
    print(f"theta[{l}]: {nn.theta[l].shape}")

# Forward propagation
print("\nForward propagation:")
predictions = nn.forward_propagation(X)
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions:\n{predictions}")

# Check internal shapes
print("\nInternal activation shapes:")
for l in range(nn.L):
    print(f"xi[{l}]: {nn.xi[l].shape}")

print("\nInternal field shapes:")
for l in range(nn.L):
    print(f"h[{l}]: {nn.h[l].shape}")

# Backward propagation
print("\nBackward propagation:")
nn.backward_propagation(X, y)

print("\nDelta shapes:")
for l in range(nn.L):
    print(f"delta[{l}]: {nn.delta[l].shape}")

# Update weights
print("\nWeight update:")
nn.update_weights_and_thresholds()

print("\nWeight changes shapes:")
for l in range(1, nn.L):
    print(f"d_w[{l}]: {nn.d_w[l].shape}")

print("\nBias changes shapes:")
for l in range(1, nn.L):
    print(f"d_theta[{l}]: {nn.d_theta[l].shape}")

print("\nBias changes (first values):")
for l in range(1, nn.L):
    print(f"d_theta[{l}]: {nn.d_theta[l].flatten()[:3]}")
