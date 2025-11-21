import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
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
# Select features and target variable
# We'll use numerical features and one-hot encode categorical features

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove target variable from features if it's in the numerical columns
target_col = 'Purchase Amount (USD)'
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

print(f'Numerical columns: {numerical_cols}')
print(f'Categorical columns: {categorical_cols}')

# Separate features (X) and target variable (y)
X = df[numerical_cols + categorical_cols]
y = df[target_col]

print(f'Features shape: {X.shape}')
print(f'Target shape: {y.shape}')

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)

print(f'Encoded features shape: {X_encoded.shape}')

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print(f'Training set - X: {X_train.shape}, y: {y_train.shape}')
print(f'Test set - X: {X_test.shape}, y: {y_test.shape}')

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Features have been standardized.')

# Prepare data for the custom BP model (reshape if needed)
X_train_bp = X_train_scaled.astype(np.float32)
X_test_bp = X_test_scaled.astype(np.float32)
y_train_bp = y_train.values.astype(np.float32)
y_test_bp = y_test.values.astype(np.float32)

print("\n--- Testing Regularization Techniques ---")

# Compare models with different regularization parameters
regularization_configs = [
    {"name": "No Regularization", "l1_reg": 0.0, "l2_reg": 0.0},
    {"name": "L1 Regularization (0.001)", "l1_reg": 0.001, "l2_reg": 0.0},
    {"name": "L2 Regularization (0.001)", "l1_reg": 0.0, "l2_reg": 0.001},
    {"name": "L1 + L2 Regularization (0.001 each)", "l1_reg": 0.001, "l2_reg": 0.001},
    {"name": "L2 Regularization (0.01)", "l1_reg": 0.0, "l2_reg": 0.01}
]

results = []

for config in regularization_configs:
    print(f"\nTesting: {config['name']}")
    
    # Create and configure the neural network with regularization
    nn = NeuralNet(
        layers=[X_train_bp.shape[1], 20, 10, 1],  # Example architecture
        learning_rate=0.01,
        momentum=0.5,
        fact='relu',
        l1_reg=config['l1_reg'],
        l2_reg=config['l2_reg']
    )
    
    # Train the model
    print(f"  Training with L1={config['l1_reg']}, L2={config['l2_reg']}")
    nn.fit(X_train_bp, y_train_bp, epochs=50, validation_split=0.2)
    
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
    
    results.append({
        "Configuration": config['name'],
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train MAPE": train_mape,
        "Test MAPE": test_mape,
        "Total Weights Magnitude": total_weights_magnitude
    })
    
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"  Train MAPE: {train_mape:.4f}%, Test MAPE: {test_mape:.4f}%")
    print(f"  Total Weights Magnitude: {total_weights_magnitude:.4f}")

# Create a summary table
import pandas as pd
results_df = pd.DataFrame(results)
print("\n--- Regularization Comparison Results ---")
print(results_df.round(4))

# Find best configuration based on test MSE
best_config_idx = results_df["Test MSE"].idxmin()
best_config = results_df.iloc[best_config_idx]
print(f"\nBest configuration based on test MSE: {best_config['Configuration']}")
print(f"Test MSE: {best_config['Test MSE']:.4f}, Test MAE: {best_config['Test MAE']:.4f}, Test MAPE: {best_config['Test MAPE']:.4f}%")

print("\nRegularization implementation completed successfully!")
print("The custom BP model now supports L1 and L2 regularization to prevent overfitting.")
