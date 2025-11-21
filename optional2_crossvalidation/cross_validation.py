import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
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

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Convert to the right format for our neural network
X_scaled = X_scaled.astype(np.float32)
y_values = y.values.astype(np.float32)

print(f'Data prepared: X shape {X_scaled.shape}, y shape {y_values.shape}')

print("\n--- K-Fold Cross-Validation Implementation ---")

# Set up k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Define model configurations to test with cross-validation
model_configs = [
    {
        "name": "Basic BP Model",
        "params": {
            "layers": [X_scaled.shape[1], 15, 8, 1],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "fact": 'relu',
            "l1_reg": 0.0,
            "l2_reg": 0.0
        }
    },
    {
        "name": "BP Model with L2 Regularization",
        "params": {
            "layers": [X_scaled.shape[1], 15, 8, 1],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "fact": 'relu',
            "l1_reg": 0.0,
            "l2_reg": 0.001
        }
    },
    {
        "name": "BP Model with L1 Regularization",
        "params": {
            "layers": [X_scaled.shape[1], 15, 8, 1],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "fact": 'relu',
            "l1_reg": 0.001,
            "l2_reg": 0.0
        }
    }
]

# Initialize results dictionary
cv_results = {}

for config in model_configs:
    print(f"\nPerforming {k_folds}-fold CV for: {config['name']}")
    
    fold_mse_scores = []
    fold_mae_scores = []
    fold_mape_scores = []
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        print(f"  Fold {fold + 1}/{k_folds}")
        
        # Split data
        X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
        y_train_fold, y_val_fold = y_values[train_idx], y_values[val_idx]
        
        # Create and train model
        nn = NeuralNet(**config['params'])
        nn.fit(X_train_fold, y_train_fold, epochs=50, validation_split=0.0)  # No internal validation split since we're doing CV
        
        # Make predictions
        y_pred = nn.predict(X_val_fold)
        
        # Calculate metrics
        mse = mean_squared_error(y_val_fold, y_pred)
        mae = mean_absolute_error(y_val_fold, y_pred)
        mape = calculate_mape(y_val_fold, y_pred)
        
        fold_mse_scores.append(mse)
        fold_mae_scores.append(mae)
        fold_mape_scores.append(mape)
        
        print(f"    MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%")
    
    # Calculate mean and std for each metric
    mean_mse = np.mean(fold_mse_scores)
    std_mse = np.std(fold_mse_scores)
    mean_mae = np.mean(fold_mae_scores)
    std_mae = np.std(fold_mae_scores)
    mean_mape = np.mean(fold_mape_scores)
    std_mape = np.std(fold_mape_scores)
    
    cv_results[config['name']] = {
        'MSE': (mean_mse, std_mse),
        'MAE': (mean_mae, std_mae),
        'MAPE': (mean_mape, std_mape),
        'Fold_MSEs': fold_mse_scores,
        'Fold_MAEs': fold_mae_scores,
        'Fold_MAPEs': fold_mape_scores
    }
    
    print(f"  {config['name']} - CV Results:")
    print(f"    MSE: {mean_mse:.4f} (+/- {std_mse * 2:.4f})")
    print(f"    MAE: {mean_mae:.4f} (+/- {std_mae * 2:.4f})")
    print(f"    MAPE: {mean_mape:.4f}% (+/- {std_mape * 2:.4f}%)")

# Create a summary table
import pandas as pd
summary_data = []
for model_name, results in cv_results.items():
    summary_data.append({
        "Model": model_name,
        "Mean MSE": f"{results['MSE'][0]:.4f}",
        "Std MSE": f"{results['MSE'][1]:.4f}",
        "Mean MAE": f"{results['MAE'][0]:.4f}",
        "Std MAE": f"{results['MAE'][1]:.4f}",
        "Mean MAPE": f"{results['MAPE'][0]:.4f}%",
        "Std MAPE": f"{results['MAPE'][1]:.4f}%"
    })

summary_df = pd.DataFrame(summary_data)
print("\n--- K-Fold Cross-Validation Summary ---")
print(summary_df)

# Identify best model based on mean CV MSE
best_model_name = min(cv_results.keys(), key=lambda x: cv_results[x]['MSE'][0])
best_mse = cv_results[best_model_name]['MSE'][0]
best_std = cv_results[best_model_name]['MSE'][1]

print(f"\nBest model based on CV MSE: {best_model_name}")
print(f"Mean CV MSE: {best_mse:.4f} (+/- {best_std * 2:.4f})")

# Additional analysis: Compare regularization effectiveness using CV
if "Basic BP Model" in cv_results and "BP Model with L2 Regularization" in cv_results:
    basic_mse = cv_results["Basic BP Model"]['MSE'][0]
    reg_mse = cv_results["BP Model with L2 Regularization"]['MSE'][0]
    improvement = (basic_mse - reg_mse) / basic_mse * 100
    
    print(f"\nL2 Regularization Effectiveness (vs Basic Model):")
    print(f"MSE improvement: {improvement:.2f}%")

print("\nK-fold cross-validation implementation completed successfully!")
print(f"The implementation tested {len(model_configs)} different model configurations with {k_folds}-fold CV.")
print("Cross-validation provides more reliable estimates of model performance by using multiple train/validation splits.")
