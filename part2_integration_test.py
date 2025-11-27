import numpy as np
from part2_bp_implementation.NeuralNet import NeuralNet  # Part 2 BP model


def load_preprocessed_data():
    """
    Load preprocessed data generated in Part 1 from .npy files.

    Requirement mapping (Part 1 -> Part 2 bridge):
    - Uses the preprocessed / normalized input features X generated in Part 1.
    - Reuses the random 80% / 20% train+validation / test split already
      performed in Part 1, so Part 2 does not re-split or re-preprocess
      the raw dataset again.
    - Ensures that all models in Part 3 (BP, BP-F, MLR-F) work on exactly
      the same preprocessed data.
    """

    # Load matrices saved by part1_generate_files.py.
    # Shapes:
    #   X_train_val: (n_train_val, n_features)
    #   y_train_val: (n_train_val, 1)
    #   X_test     : (n_test, n_features)
    #   y_test     : (n_test, 1)
    X_train_val = np.load("./dataset/preprocessed/X_train_val.npy")
    y_train_val = np.load("./dataset/preprocessed/y_train_val.npy")
    X_test = np.load("./dataset/preprocessed/X_test.npy")
    y_test = np.load("./dataset/preprocessed/y_test.npy")

    # Ensure y arrays are 2D column vectors (n_samples, 1), in case they were
    # saved as 1D. This matches the expected interface in NeuralNet.
    if y_train_val.ndim == 1:
        y_train_val = y_train_val.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    print(f"Train+val X shape: {X_train_val.shape}, y shape: {y_train_val.shape}")
    print(f"Test      X shape: {X_test.shape}, y shape: {y_test.shape}")

    return X_train_val, y_train_val, X_test, y_test


def integrate_with_preprocessed_data():
    """
    Integrate the custom BP model (Part 2) with the dataset preprocessed in Part 1.

    Requirement mapping (Part 2: Implementation of BP):
    - Receives one preprocessed dataset as input (X, y) and uses it to train the
      multilayer network implemented from scratch.
    - Uses a separate test set (20% hold-out from Part 1) only for final evaluation.
    - Scales the output variable according to BP.v2 (standardization) and
      descales it after prediction on the test set.
    - Calls NeuralNet.fit(X, y) with a validation split so the network internally
      separates training and validation data (as required in Part 2).
    """

    # -------------------------------------------------------------------------
    # Step 1: Get normalized inputs and 80/20 split from preprocessed files
    # -------------------------------------------------------------------------
    # X_train_val, X_test are already normalized (StandardScaler) in Part 1.
    # This satisfies the "input normalization" requirement of BP.v2 for X.
    X_train_val, y_train_val, X_test, y_test = load_preprocessed_data()

    # -------------------------------------------------------------------------
    # Step 2: Standardize the target variable using train+val statistics only
    # -------------------------------------------------------------------------
    # BP.v2 preprocessing requirement:
    # "Standardize/scale the desired output variables and then descale when evaluating."
    y_mean = y_train_val.mean(axis=0)
    y_std = y_train_val.std(axis=0)

    # Avoid division by zero in case of zero variance (very unlikely but safe).
    y_std[y_std == 0] = 1.0

    y_train_val_scaled = (y_train_val - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    print("Output standardization:")
    print(f"  y_mean: {y_mean}")
    print(f"  y_std : {y_std}")

    # -------------------------------------------------------------------------
    # Step 3: Configure the multilayer network architecture and hyperparameters
    # -------------------------------------------------------------------------
    # Part 2 requirement: the code must support arbitrary multilayer networks,
    # encoded by an array n with the number of units in each layer.
    n_features = X_train_val.shape[1]
    layers = [n_features, 15, 8, 1]  # Example architecture: input - 2 hidden - output

    # The NeuralNet constructor receives:
    # - Number of units in each layer (layers array -> n).
    # - Learning rate and momentum.
    # - Activation function name (fact).
    # This matches the constructor requirements in Part 2.
    nn = NeuralNet(
        layers=layers,
        learning_rate=0.01,
        momentum=0.9,
        fact="sigmoid",  # Allowed activation functions: sigmoid / relu / linear / tanh
        l1_reg=0.0,      # Regularization parameters kept at 0.0 for basic Part 2
        l2_reg=0.0,
    )

    print(f"Neural network architecture (layers): {layers}")
    print(f"Learning rate: {nn.learning_rate}, Momentum: {nn.momentum}, Activation: {nn.fact}")

    # -------------------------------------------------------------------------
    # Step 4: Train the network with training + validation split
    # -------------------------------------------------------------------------
    # Part 2 requirement:
    # - The code receives one dataset and, using a percentage of data, divides
    #   it into training and validation. In this implementation, the percentage
    #   is given to fit() via validation_split.
    epochs = 50  # Number of epochs (can be tuned in Part 3 hyperparameter search)

    print("Starting training...")
    train_losses, val_losses = nn.fit(
        X=X_train_val,
        y=y_train_val_scaled,
        epochs=epochs,
        validation_split=0.2,  # 20% of (train+val) used as validation set
    )
    print("Training completed.")

    # -------------------------------------------------------------------------
    # Step 5: Predict on test set and descale outputs
    # -------------------------------------------------------------------------
    # Listing 1 in BP.v2: after training, feed-forward all test patterns, descale
    # the predictions and evaluate them.
    print("Making predictions on test set...")
    y_pred_scaled = nn.predict(X_test)

    # Ensure predictions are 2D column vectors for consistent descaling
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    # Descale predictions back to original units
    y_pred = y_pred_scaled * y_std + y_mean

    # y_test is already in the original scale (we only scaled y_test_scaled for training)
    y_test_original = y_test

    # -------------------------------------------------------------------------
    # Step 6: Compute performance metrics (MSE, MAE, MAPE)
    # -------------------------------------------------------------------------
    # These metrics are explicitly required in Part 3, but we can already compute
    # them here for the custom BP model.
    mse = float(np.mean((y_test_original - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_test_original - y_pred)))
    mape = float(np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100)

    print("Test set performance (custom BP):")
    print(f"  MSE : {mse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")

    # -------------------------------------------------------------------------
    # Step 7: Obtain loss evolution for plotting
    # -------------------------------------------------------------------------
    # Part 2 + Part 3 requirement:
    # - loss_epochs() must return the evolution of training and validation errors
    #   for each epoch, so they can be plotted.
    train_error_evolution, val_error_evolution = nn.loss_epochs()
    print(
        f"Loss evolution shapes - Train: {train_error_evolution.shape}, "
        f"Val: {val_error_evolution.shape}"
    )

    # Return everything needed for later analysis (e.g. Jupyter notebooks in Part 3)
    return (
        nn,
        (X_train_val, X_test, y_train_val_scaled, y_test_scaled),
        (train_losses, val_losses),
        (mse, mae, mape),
    )


if __name__ == "__main__":
    # Run the integration to train and evaluate the BP model on the Shopping dataset.
    # This script corresponds to the Part 2 integration over the preprocessed files
    # generated in Part 1.
    model, data, losses, metrics = integrate_with_preprocessed_data()

    print("\nIntegration with preprocessed data completed successfully.")
    print("The custom BP model has been trained and evaluated on the test set.")
