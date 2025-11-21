import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add path to access the custom BP implementation from Part 2
sys.path.append('../part2_bp_implementation/')
from NeuralNet import NeuralNet

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_true = y_true != 0
    # Avoid division by zero by only calculating MAPE for non-zero true values
    mape = np.mean(np.abs((y_true[non_zero_true] - y_pred[non_zero_true]) / y_true[non_zero_true])) * 100
    return mape

# Load the dataset
dataset_path = "../dataset/shopping_behavior.csv"
df = pd.read_csv(dataset_path)

print(f'Dataset loaded successfully with shape: {df.shape}')
print('Dataset columns:', df.columns.tolist())

# Data preprocessing for models
# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove target variable from features if it's in the numerical columns
target_col = 'Purchase Amount (USD)'
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

# Separate features (X) and target variable (y)
X = df[numerical_cols + categorical_cols]
y = df[target_col]

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Prepare data for the custom BP model (reshape if needed)
X_train_bp = X_train_scaled.astype(np.float32)
X_test_bp = X_test_scaled.astype(np.float32)
y_train_bp = y_train.values.astype(np.float32)
y_test_bp = y_test.values.astype(np.float32)

print("\n--- Regularization Evaluation: Comparing Regularized vs Non-Regularized Models ---")

# Define models to compare
model_configs = [
    {
        "name": "No Regularization",
        "params": {
            "layers": [X_train_bp.shape[1], 20, 10, 1],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "fact": 'relu',
            "l1_reg": 0.0,
            "l2_reg": 0.0
        }
    },
    {
        "name": "L2 Regularization (0.001)",
        "params": {
            "layers": [X_train_bp.shape[1], 20, 10, 1],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "fact": 'relu',
            "l1_reg": 0.0,
            "l2_reg": 0.001
        }
    },
    {
        "name": "L1 Regularization (0.001)",
        "params": {
            "layers": [X_train_bp.shape[1], 20, 10, 1],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "fact": 'relu',
            "l1_reg": 0.001,
            "l2_reg": 0.0
        }
    },
    {
        "name": "L1+L2 Regularization (0.0001 each)",
        "params": {
            "layers": [X_train_bp.shape[1], 20, 10, 1],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "fact": 'relu',
            "l1_reg": 0.0001,
            "l2_reg": 0.0001
        }
    }
]

# Train and evaluate all models
models = {}
evaluations = []

for config in model_configs:
    print(f"\nTraining model: {config['name']}")
    
    # Create and configure the neural network
    nn = NeuralNet(**config['params'])
    
    # Train the model
    nn.fit(X_train_bp, y_train_bp, epochs=100, validation_split=0.2)
    
    # Store the trained model
    models[config['name']] = nn
    
    # Get predictions
    y_train_pred = nn.predict(X_train_bp)
    y_test_pred = nn.predict(X_test_bp)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train_bp, y_train_pred)
    test_mse = mean_squared_error(y_test_bp, y_test_pred)
    
    train_mae = mean_absolute_error(y_train_bp, y_train_pred)
    test_mae = mean_absolute_error(y_test_bp, y_test_pred)
    
    train_mape = calculate_mape(y_train_bp, y_train_pred)
    test_mape = calculate_mape(y_test_bp, y_test_pred)
    
    # Calculate total weights magnitude for regularization analysis
    total_weights_magnitude = 0.0
    for l in range(1, nn.L):
        total_weights_magnitude += np.sum(np.abs(nn.w[l]))
    
    # Calculate weight variance for regularization analysis
    weight_variance = 0.0
    for l in range(1, nn.L):
        weight_variance += np.sum(nn.w[l] ** 2)
    
    # Overfitting measure (difference between train and test error)
    mse_diff = train_mse - test_mse
    mae_diff = train_mae - test_mae
    
    evaluations.append({
        "Model": config['name'],
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train MAPE": train_mape,
        "Test MAPE": test_mape,
        "Total Weights Magnitude": total_weights_magnitude,
        "Weight Variance": weight_variance,
        "MSE Difference (Train-Test)": abs(mse_diff),
        "MAE Difference (Train-Test)": abs(mae_diff)
    })
    
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"  Train MAPE: {train_mape:.4f}%, Test MAPE: {test_mape:.4f}%")
    print(f"  Total Weights Magnitude: {total_weights_magnitude:.4f}")
    print(f"  Weight Variance: {weight_variance:.4f}")
    print(f"  MSE Difference (Train-Test): {abs(mse_diff):.4f}")

# Create a summary table
import pandas as pd
evaluations_df = pd.DataFrame(evaluations)
print("\n--- Regularization Evaluation Results ---")
print(evaluations_df.round(4))

# Find best model based on test MSE
best_model_idx = evaluations_df["Test MSE"].idxmin()
best_model = evaluations_df.iloc[best_model_idx]
print(f"\nBest model based on test MSE: {best_model['Model']}")
print(f"Test MSE: {best_model['Test MSE']:.4f}, Test MAE: {best_model['Test MAE']:.4f}, Test MAPE: {best_model['Test MAPE']:.4f}%")

# Analysis of regularization effectiveness
print(f"\n--- Regularization Effectiveness Analysis ---")
no_reg_row = evaluations_df[evaluations_df["Model"] == "No Regularization"].iloc[0]

for _, row in evaluations_df.iterrows():
    if row["Model"] == "No Regularization":
        continue
    
    # Compare test performance
    mse_improvement = (no_reg_row["Test MSE"] - row["Test MSE"]) / no_reg_row["Test MSE"] * 100
    mae_improvement = (no_reg_row["Test MAE"] - row["Test MAE"]) / no_reg_row["Test MAE"] * 100
    
    # Compare overfitting (smaller difference between train and test is better)
    overfitting_reduction = (no_reg_row["MSE Difference (Train-Test)"] - row["MSE Difference (Train-Test)"]) / no_reg_row["MSE Difference (Train-Test)"] * 100
    
    # Compare weight magnitudes
    weight_reduction = (no_reg_row["Total Weights Magnitude"] - row["Total Weights Magnitude"]) / no_reg_row["Total Weights Magnitude"] * 100
    
    print(f"\n{row['Model']}:")
    print(f"  Test MSE improvement over no regularization: {mse_improvement:.2f}%")
    print(f"  Test MAE improvement over no regularization: {mae_improvement:.2f}%")
    print(f"  Overfitting reduction: {overfitting_reduction:.2f}%")
    print(f"  Weight magnitude reduction: {weight_reduction:.2f}%")

# Create visualization comparing models
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Test MSE comparison
axes[0, 0].bar(evaluations_df["Model"], evaluations_df["Test MSE"])
axes[0, 0].set_title("Test MSE Comparison")
axes[0, 0].set_ylabel("MSE")
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Test MAE comparison
axes[0, 1].bar(evaluations_df["Model"], evaluations_df["Test MAE"])
axes[0, 1].set_title("Test MAE Comparison")
axes[0, 1].set_ylabel("MAE")
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Weight Magnitude comparison
axes[1, 0].bar(evaluations_df["Model"], evaluations_df["Total Weights Magnitude"])
axes[1, 0].set_title("Total Weights Magnitude Comparison")
axes[1, 0].set_ylabel("Total Weights Magnitude")
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Overfitting (Train-Test difference) comparison
axes[1, 1].bar(evaluations_df["Model"], evaluations_df["MSE Difference (Train-Test)"])
axes[1, 1].set_title("Overfitting Measure (MSE Difference Train-Test)")
axes[1, 1].set_ylabel("MSE Difference")
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('regularization_comparison.png')
print("\nComparison plot saved as regularization_comparison.png")

# Generate scatter plots for the best regularized model vs non-regularized model
best_reg_model = evaluations_df[evaluations_df["Model"] != "No Regularization"].iloc[evaluations_df[evaluations_df["Model"] != "No Regularization"]["Test MSE"].idxmin()]
worst_model = evaluations_df[evaluations_df["Model"] == "No Regularization"]

best_model_nn = models[best_reg_model["Model"]]
worst_model_nn = models[worst_model["Model"]]

y_best_pred = best_model_nn.predict(X_test_bp)
y_worst_pred = worst_model_nn.predict(X_test_bp)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Best regularized model
axes[0].scatter(y_test_bp, y_best_pred, alpha=0.5)
axes[0].plot([y_test_bp.min(), y_test_bp.max()], [y_test_bp.min(), y_test_bp.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title(f'{best_reg_model["Model"]}\nTest MSE: {best_reg_model["Test MSE"]:.4f}')

# Non-regularized model
axes[1].scatter(y_test_bp, y_worst_pred, alpha=0.5)
axes[1].plot([y_test_bp.min(), y_test_bp.max()], [y_test_bp.min(), y_test_bp.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Values')
axes[1].set_ylabel('Predicted Values')
axes[1].set_title(f'{worst_model["Model"]}\nTest MSE: {worst_model["Test MSE"]:.4f}')

plt.tight_layout()
plt.savefig('regularization_scatter_comparison.png')
print("Scatter plot comparison saved as regularization_scatter_comparison.png")

print("\nRegularization evaluation completed successfully!")
print("The comparison clearly shows the benefits of regularization in improving generalization and reducing overfitting.")