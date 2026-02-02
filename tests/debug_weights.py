#!/usr/bin/env python3
"""
Debug script to check weight initialization
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from part2_bp_implementation.NeuralNet import NeuralNet

np.random.seed(0)

# Create neural network with sigmoid activation
nn_sigmoid = NeuralNet(
    layers=[134, 10, 1],
    learning_rate=0.01,
    momentum=0.0,
    fact='sigmoid',
    l1_reg=0.0,
    l2_reg=0.0,
)

print("Sigmoid network weight statistics:")
print(f"w[1] mean: {nn_sigmoid.w[1].mean():.6f}, std: {nn_sigmoid.w[1].std():.6f}")
print(f"w[1] min: {nn_sigmoid.w[1].min():.6f}, max: {nn_sigmoid.w[1].max():.6f}")
print(f"w[2] mean: {nn_sigmoid.w[2].mean():.6f}, std: {nn_sigmoid.w[2].std():.6f}")
print(f"w[2] min: {nn_sigmoid.w[2].min():.6f}, max: {nn_sigmoid.w[2].max():.6f}")
print(f"theta[1] mean: {nn_sigmoid.theta[1].mean():.6f}, std: {nn_sigmoid.theta[1].std():.6f}")
print(f"theta[2] mean: {nn_sigmoid.theta[2].mean():.6f}, std: {nn_sigmoid.theta[2].std():.6f}")

# Create neural network with relu activation
nn_relu = NeuralNet(
    layers=[134, 10, 1],
    learning_rate=0.01,
    momentum=0.0,
    fact='relu',
    l1_reg=0.0,
    l2_reg=0.0,
)

print("\nReLU network weight statistics:")
print(f"w[1] mean: {nn_relu.w[1].mean():.6f}, std: {nn_relu.w[1].std():.6f}")
print(f"w[1] min: {nn_relu.w[1].min():.6f}, max: {nn_relu.w[1].max():.6f}")
print(f"w[2] mean: {nn_relu.w[2].mean():.6f}, std: {nn_relu.w[2].std():.6f}")
print(f"w[2] min: {nn_relu.w[2].min():.6f}, max: {nn_relu.w[2].max():.6f}")
print(f"theta[1] mean: {nn_relu.theta[1].mean():.6f}, std: {nn_relu.theta[1].std():.6f}")
print(f"theta[2] mean: {nn_relu.theta[2].mean():.6f}, std: {nn_relu.theta[2].std():.6f}")

# Create neural network with tanh activation
nn_tanh = NeuralNet(
    layers=[134, 10, 1],
    learning_rate=0.01,
    momentum=0.0,
    fact='tanh',
    l1_reg=0.0,
    l2_reg=0.0,
)

print("\nTanh network weight statistics:")
print(f"w[1] mean: {nn_tanh.w[1].mean():.6f}, std: {nn_tanh.w[1].std():.6f}")
print(f"w[1] min: {nn_tanh.w[1].min():.6f}, max: {nn_tanh.w[1].max():.6f}")
print(f"w[2] mean: {nn_tanh.w[2].mean():.6f}, std: {nn_tanh.w[2].std():.6f}")
print(f"w[2] min: {nn_tanh.w[2].min():.6f}, max: {nn_tanh.w[2].max():.6f}")
print(f"theta[1] mean: {nn_tanh.theta[1].mean():.6f}, std: {nn_tanh.theta[1].std():.6f}")
print(f"theta[2] mean: {nn_tanh.theta[2].mean():.6f}, std: {nn_tanh.theta[2].std():.6f}")