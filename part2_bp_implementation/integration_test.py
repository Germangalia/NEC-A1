import numpy as np
from NeuralNet import NeuralNet
import pandas as pd
import pickle

def load_preprocessed_data():
    """
    Load preprocessed data from part1_preprocessing
    """
    import pandas as pd
    import os
    
    # First, try to load the actual dataset from the provided file
    dataset_path = "../dataset/shopping_behavior.csv"
    
    if os.path.exists(dataset_path):
        # Load the actual dataset
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        
        # Display basic info about the dataset
        print("Dataset columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head())
        
        # Select relevant features for the model
        # Based on the requirements, we need at least 10 input features and 1 output feature
        # The target variable is "Purchase Amount (USD)"
        
        # Handle categorical variables by encoding them
        df_encoded = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 
                              'Subscription Status', 'Discount Applied', 'Payment Method', 
                              'Frequency of Purchases']
        
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
        
        # Define features (input) and target (output)
        feature_columns = ['Age', 'Gender', 'Item Purchased', 'Category', 'Location',
                          'Review Rating', 'Subscription Status', 'Discount Applied',
                          'Previous Purchases', 'Payment Method', 'Frequency of Purchases']
        
        # Filter to only include columns that exist in the dataset
        available_features = [col for col in feature_columns if col in df_encoded.columns]
        
        X = df_encoded[available_features].values
        y = df_encoded['Purchase Amount (USD)'].values.reshape(-1, 1)
        
        print(f"Features used: {available_features}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        return X, y
    else:
        print(f"Dataset file not found at {dataset_path}")
        print("The dataset needs to be downloaded or accessed.")
        
        # If kagglehub is not available, show an error and stop execution
        print("Error: Dataset file not found and kagglehub is not available.")
        print("Please ensure the dataset file exists at the specified location.")
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

def integrate_with_preprocessed_data():
    """
    Integrate the custom BP model with preprocessed data
    """
    print("Loading preprocessed data...")
    X, y = load_preprocessed_data()
    
    print(f"Data shape - X: {X.shape}, y: {y.shape}")
    print(f"Sample of X: {X[:5]}")
    print(f"Sample of y: {y[:5]}")
    
    # Normalize the features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    # Normalize the target
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_normalized = (y - y_mean) / y_std
    
    # Split the data into training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.2, random_state=42
    )
    
    print(f"Training set - X: {X_train.shape}, y: {y_train.shape}")
    print(f"Test set - X: {X_test.shape}, y: {y_test.shape}")
    
    # Get the number of features from the dataset
    n_features = X_train.shape[1]
    print(f"Number of features in dataset: {n_features}")
    
    # Create and configure the neural network
    # Using n_features input nodes to match the actual dataset
    layers = [n_features, 15, 8, 1]  # [input_size, hidden1, hidden2, output]
    nn = NeuralNet(
        layers=layers,
        learning_rate=0.01,
        momentum=0.9,
        fact='sigmoid'  # Using sigmoid as activation function
    )
    
    print("Starting training...")
    
    # Train the model
    train_losses, val_losses = nn.fit(
        X=X_train,
        y=y_train,
        epochs=50,
        validation_split=0.2  # Use 20% of training data for validation
    )
    
    print("Training completed!")
    
    # Make predictions on the test set
    print("Making predictions on test set...")
    y_pred_normalized = nn.predict(X_test)
    
    # Denormalize predictions to original scale
    y_pred = y_pred_normalized * y_std + y_mean
    y_test_original = y_test * y_std + y_mean
    
    # Calculate performance metrics
    mse = np.mean((y_test_original - y_pred) ** 2)
    mae = np.mean(np.abs(y_test_original - y_pred))
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
    
    print(f"Test Set Performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    # Get loss evolution
    train_error_evolution, val_error_evolution = nn.loss_epochs()
    print(f"Loss evolution arrays shape - Train: {train_error_evolution.shape}, Val: {val_error_evolution.shape}")
    
    return nn, (X_train, X_test, y_train, y_test), (train_losses, val_losses)

if __name__ == "__main__":
    # Run the integration
    model, data, losses = integrate_with_preprocessed_data()
    
    print("\nIntegration with preprocessed data completed successfully!")
    print("The custom BP model has been trained and evaluated.")
