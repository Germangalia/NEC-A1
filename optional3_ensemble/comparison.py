import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

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

print(f'Training set - X: {X_train_scaled.shape}, y: {y_train.shape}')
print(f'Test set - X: {X_test_scaled.shape}, y: {y_test.shape}')

print("\n--- Comprehensive Model Comparison ---")

# Define all models to compare
all_models = [
    {
        "name": "Linear Regression (MLR-F)",
        "model": LinearRegression(),
        "requires_scaling": True
    },
    {
        "name": "Neural Network (BP-F)",
        "model": MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42, early_stopping=True),
        "requires_scaling": True
    },
    {
        "name": "Random Forest",
        "model": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        "requires_scaling": False
    },
    {
        "name": "Gradient Boosting",
        "model": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1),
        "requires_scaling": False
    },
]

# Train and evaluate all models
all_results = []

# Add the custom BP model
print("Training Custom BP Model...")
bp_model = NeuralNet(
    layers=[X_train_bp.shape[1], 20, 10, 1],
    learning_rate=0.01,
    momentum=0.5,
    fact='relu',
    l1_reg=0.001,
    l2_reg=0.001
)
bp_model.fit(X_train_bp, y_train_bp, epochs=100, validation_split=0.2)

y_train_pred_bp = bp_model.predict(X_train_bp)
y_test_pred_bp = bp_model.predict(X_test_bp)

bp_result = {
    "Model": "Custom BP Model",
    "Train MSE": mean_squared_error(y_train_bp, y_train_pred_bp),
    "Test MSE": mean_squared_error(y_test_bp, y_test_pred_bp),
    "Train MAE": mean_absolute_error(y_train_bp, y_train_pred_bp),
    "Test MAE": mean_absolute_error(y_test_bp, y_test_pred_bp),
    "Train MAPE": calculate_mape(y_train_bp, y_train_pred_bp),
    "Test MAPE": calculate_mape(y_test_bp, y_test_pred_bp)
}

all_results.append(bp_result)
print(f"Custom BP Model - Test MSE: {bp_result['Test MSE']:.4f}, Test MAE: {bp_result['Test MAE']:.4f}, Test MAPE: {bp_result['Test MAPE']:.4f}%")

# Train and evaluate traditional models
for config in all_models:
    print(f"Training: {config['name']}")
    
    model = config['model']
    
    if config['requires_scaling']:
        # Use scaled data
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    else:
        # Use original data (no scaling needed for tree-based models)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
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
    
    all_results.append(result)
    
    print(f"  Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test MAPE: {test_mape:.4f}%")

# Create comprehensive results summary
import pandas as pd
all_results_df = pd.DataFrame(all_results)
print(f"\n--- Comprehensive Model Comparison Results ---")
print(all_results_df.round(4))

# Identify best model based on test MSE
best_model_idx = all_results_df["Test MSE"].idxmin()
best_model = all_results_df.iloc[best_model_idx]
print(f"\nBest overall model based on Test MSE: {best_model['Model']}")
print(f"Test MSE: {best_model['Test MSE']:.4f}, Test MAE: {best_model['Test MAE']:.4f}, Test MAPE: {best_model['Test MAPE']:.4f}%")

# Analyze ensemble vs other methods
ensemble_models = ['Random Forest', 'Gradient Boosting']
traditional_models = ['Linear Regression (MLR-F)', 'Neural Network (BP-F)', 'Custom BP Model']

ensemble_results = all_results_df[all_results_df['Model'].isin(ensemble_models)]
traditional_results = all_results_df[all_results_df['Model'].isin(traditional_models)]

print(f"\n--- Ensemble Methods vs Traditional Methods Analysis ---")
print(f"Ensemble Methods Average Test MSE: {ensemble_results['Test MSE'].mean():.4f}")
print(f"Traditional Methods Average Test MSE: {traditional_results['Test MSE'].mean():.4f}")

if ensemble_results['Test MSE'].mean() < traditional_results['Test MSE'].mean():
    print("Ensemble methods perform better on average than traditional methods (based on MSE)")
else:
    print("Traditional methods perform better on average than ensemble methods (based on MSE)")

# Create visualization comparing all models
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Test MSE comparison
axes[0, 0].bar(all_results_df["Model"], all_results_df["Test MSE"])
axes[0, 0].set_title("Test MSE Comparison")
axes[0, 0].set_ylabel("MSE")
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Test MAE comparison
axes[0, 1].bar(all_results_df["Model"], all_results_df["Test MAE"])
axes[0, 1].set_title("Test MAE Comparison")
axes[0, 1].set_ylabel("MAE")
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Test MAPE comparison
axes[1, 0].bar(all_results_df["Model"], all_results_df["Test MAPE"])
axes[1, 0].set_title("Test MAPE Comparison")
axes[1, 0].set_ylabel("MAPE (%)")
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Model grouping comparison
model_types = ['Ensemble', 'Traditional']
avg_mse_by_type = [
    ensemble_results['Test MSE'].mean(),
    traditional_results['Test MSE'].mean()
]
axes[1, 1].bar(model_types, avg_mse_by_type)
axes[1, 1].set_title("Average Test MSE by Model Type")
axes[1, 1].set_ylabel("Average MSE")

plt.tight_layout()
plt.savefig('model_comparison_all.png')
print("\nComparison plot saved as model_comparison_all.png")

# Generate scatter plots for the top 3 models
sorted_models = all_results_df.sort_values('Test MSE')
top_3_models = sorted_models.head(3)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (_, model_row) in enumerate(top_3_models.iterrows()):
    model_name = model_row['Model']
    
    # Get predictions for the specific model
    if model_name == "Custom BP Model":
        y_pred = bp_model.predict(X_test_bp)
    elif model_name == "Linear Regression (MLR-F)":
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        y_pred = lr_model.predict(X_test_scaled)
    elif model_name == "Neural Network (BP-F)":
        nn_model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42, early_stopping=True)
        nn_model.fit(X_train_scaled, y_train)
        y_pred = nn_model.predict(X_test_scaled)
    elif model_name == "Random Forest":
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
    elif model_name == "Gradient Boosting":
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
        gb_model.fit(X_train, y_train)
        y_pred = gb_model.predict(X_test)
    
    axes[i].scatter(y_test, y_pred, alpha=0.5)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[i].set_xlabel('Actual Values')
    axes[i].set_ylabel('Predicted Values')
    axes[i].set_title(f'{model_name}\nTest MSE: {model_row["Test MSE"]:.4f}')

plt.tight_layout()
plt.savefig('top_models_scatter.png')
print("Top models scatter plot saved as top_models_scatter.png")

# Analysis of ensemble method advantages
print(f"\n--- Analysis of Ensemble Method Advantages ---")
for _, row in all_results_df.iterrows():
    if "Random Forest" in row['Model'] or "Gradient Boosting" in row['Model']:
        # Calculate overfitting measure (difference between train and test error)
        mse_diff = abs(row['Train MSE'] - row['Test MSE'])
        print(f"{row['Model']}:\n")
        print(f"  Overfitting measure (|Train MSE - Test MSE|): {mse_diff:.4f}")
        print(f"  Generalization ability: {'Good' if mse_diff < 0.1 else 'Fair' if mse_diff < 0.5 else 'Poor'}")

print(f"\nEnsemble methods comparison completed successfully!")
print("The analysis compares ensemble methods (Random Forest, Gradient Boosting) with other implemented models.")
