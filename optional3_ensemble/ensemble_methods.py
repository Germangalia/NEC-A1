import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

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

print(f'Training set - X: {X_train_scaled.shape}, y: {y_train.shape}')
print(f'Test set - X: {X_test_scaled.shape}, y: {y_test.shape}')

print("\n--- Ensemble Learning Methods Implementation ---")

# Define ensemble methods to implement
ensemble_configs = [
    {
        "name": "Random Forest",
        "model": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    },
    {
        "name": "Gradient Boosting",
        "model": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
    },
    {
        "name": "Random Forest (Tuned)",
        "model": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15, min_samples_split=5, min_samples_leaf=2)
    },
    {
        "name": "Gradient Boosting (Tuned)",
        "model": GradientBoostingRegressor(n_estimators=150, random_state=42, max_depth=8, learning_rate=0.05, subsample=0.8)
    }
]

# Train and evaluate ensemble models
ensemble_results = []

for config in ensemble_configs:
    print(f"\nTraining: {config['name']}")
    
    # Train the model
    model = config['model']
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_mape = calculate_mape(y_train, y_train_pred)
    test_mape = calculate_mape(y_test, y_test_pred)
    
    # Store results
    result = {
        "Model": config['name'],
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train MAPE": train_mape,
        "Test MAPE": test_mape
    }
    
    ensemble_results.append(result)
    
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"  Train MAPE: {train_mape:.4f}%, Test MAPE: {test_mape:.4f}%")

# Add baseline models for comparison
print(f"\n--- Adding Baseline Models for Comparison ---")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_train_pred_lr = lr_model.predict(X_train_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

lr_result = {
    "Model": "Linear Regression",
    "Train MSE": mean_squared_error(y_train, y_train_pred_lr),
    "Test MSE": mean_squared_error(y_test, y_test_pred_lr),
    "Train MAE": mean_absolute_error(y_train, y_train_pred_lr),
    "Test MAE": mean_absolute_error(y_test, y_test_pred_lr),
    "Train MAPE": calculate_mape(y_train, y_train_pred_lr),
    "Test MAPE": calculate_mape(y_test, y_test_pred_lr)
}

ensemble_results.append(lr_result)
print(f"Linear Regression - Test MSE: {lr_result['Test MSE']:.4f}, Test MAE: {lr_result['Test MAE']:.4f}, Test MAPE: {lr_result['Test MAPE']:.4f}%")

# Neural Network from scikit-learn
nn_model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42, early_stopping=True)
n_model.fit(X_train_scaled, y_train)
y_train_pred_nn = nn_model.predict(X_train_scaled)
y_test_pred_nn = nn_model.predict(X_test_scaled)

nn_result = {
    "Model": "Neural Network (MLP)",
    "Train MSE": mean_squared_error(y_train, y_train_pred_nn),
    "Test MSE": mean_squared_error(y_test, y_test_pred_nn),
    "Train MAE": mean_absolute_error(y_train, y_train_pred_nn),
    "Test MAE": mean_absolute_error(y_test, y_test_pred_nn),
    "Train MAPE": calculate_mape(y_train, y_train_pred_nn),
    "Test MAPE": calculate_mape(y_test, y_test_pred_nn)
}

ensemble_results.append(nn_result)
print(f"Neural Network - Test MSE: {nn_result['Test MSE']:.4f}, Test MAE: {nn_result['Test MAE']:.4f}, Test MAPE: {nn_result['Test MAPE']:.4f}%")

# Create results summary
import pandas as pd
results_df = pd.DataFrame(ensemble_results)
print(f"\n--- Ensemble Learning Results Summary ---")
print(results_df.round(4))

# Find best ensemble model based on test MSE
ensemble_results_only = [r for r in ensemble_results if "Ensemble" in r["Model"] or "Random Forest" in r["Model"] or "Gradient Boosting" in r["Model"]]
if ensemble_results_only:
    best_ensemble_idx = min(range(len(ensemble_results_only)), key=lambda i: ensemble_results_only[i]['Test MSE'])
    best_ensemble = ensemble_results_only[best_ensemble_idx]
    print(f"\nBest ensemble model: {best_ensemble['Model']}")
    print(f"Test MSE: {best_ensemble['Test MSE']:.4f}, Test MAE: {best_ensemble['Test MAE']:.4f}, Test MAPE: {best_ensemble['Test MAPE']:.4f}%")

# Feature importance analysis for Random Forest
rf_model = None
for config in ensemble_configs:
    if "Random Forest" in config['name'] and "Tuned" not in config['name']:
        rf_model = config['model']
        break

if rf_model is not None:
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n--- Top 10 Feature Importances (Random Forest) ---")
    print(feature_importance.head(10))

print(f"\nEnsemble learning methods implementation completed successfully!")
print(f"Implemented Random Forest and Gradient Boosting with multiple configurations.")