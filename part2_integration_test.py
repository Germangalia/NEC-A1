import numpy as np
from part2_bp_implementation.NeuralNet import NeuralNet


def load_preprocessed_data():
    X_train_val = np.load("./dataset/preprocessed/X_train_val.npy")
    y_train_val = np.load("./dataset/preprocessed/y_train_val.npy")
    X_test = np.load("./dataset/preprocessed/X_test.npy")
    y_test = np.load("./dataset/preprocessed/y_test.npy")

    if y_train_val.ndim == 1:
        y_train_val = y_train_val.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    print(f"Train+val X shape: {X_train_val.shape}, y shape: {y_train_val.shape}")
    print(f"Test      X shape: {X_test.shape}, y shape: {y_test.shape}")

    return X_train_val, y_train_val, X_test, y_test


def integrate_with_preprocessed_data():
    X_train_val, y_train_val, X_test, y_test = load_preprocessed_data()

    y_mean = y_train_val.mean(axis=0)
    y_std = y_train_val.std(axis=0)
    y_std[y_std == 0] = 1.0

    y_train_val_scaled = (y_train_val - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    print("Output standardization:")
    print(f"  y_mean: {y_mean}")
    print(f"  y_std : {y_std}")

    n_features = X_train_val.shape[1]
    layers = [n_features, 15, 8, 1]

    nn = NeuralNet(
        layers=layers,
        learning_rate=0.01,
        momentum=0.9,
        fact="sigmoid",
        l1_reg=0.0,
        l2_reg=0.0,
    )

    print(f"Neural network architecture (layers): {layers}")
    print(f"Learning rate: {nn.learning_rate}, Momentum: {nn.momentum}, Activation: {nn.fact}")

    epochs = 50

    print("Starting training...")
    train_losses, val_losses = nn.fit(
        X=X_train_val,
        y=y_train_val_scaled,
        epochs=epochs,
        validation_split=0.2,
    )
    print("Training completed.")

    print("Making predictions on test set...")
    y_pred_scaled = nn.predict(X_test)

    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    y_pred = y_pred_scaled * y_std + y_mean
    y_test_original = y_test

    mse = float(np.mean((y_test_original - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_test_original - y_pred)))
    mape = float(np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100)

    print("Test set performance (custom BP):")
    print(f"  MSE : {mse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")

    train_error_evolution, val_error_evolution = nn.loss_epochs()
    print(
        f"Loss evolution shapes - Train: {train_error_evolution.shape}, "
        f"Val: {val_error_evolution.shape}"
    )

    return (
        nn,
        (X_train_val, X_test, y_train_val_scaled, y_test_scaled),
        (train_losses, val_losses),
        (mse, mae, mape),
    )


if __name__ == "__main__":
    model, data, losses, metrics = integrate_with_preprocessed_data()
    print("\nIntegration with preprocessed data completed successfully.")
    print("The custom BP model has been trained and evaluated on the test set.")
