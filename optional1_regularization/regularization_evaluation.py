import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from part2_bp_implementation.NeuralNet import NeuralNet


def load_preprocessed_data():
    """
    Load the preprocessed train+validation and test sets from Part 1.

    This ensures that the regularization study uses the same data splits
    as Part 2 (BP), Part 3 (model comparison) and the parameter_tuning script.
    """

    X_train_val = np.load("./../dataset/preprocessed/X_train_val.npy")
    y_train_val = np.load("./../dataset/preprocessed/y_train_val.npy")
    X_test = np.load("./../dataset/preprocessed/X_test.npy")
    y_test = np.load("./../dataset/preprocessed/y_test.npy")

    if y_train_val.ndim == 1:
        y_train_val = y_train_val.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    return X_train_val, y_train_val, X_test, y_test


def regression_metrics(y_true, y_pred):
    """
    Compute MSE, MAE and MAPE between y_true and y_pred in original units.
    """

    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)

    return mse, mae, mape


def train_and_evaluate_model(
    name,
    l1_reg,
    l2_reg,
    X_train_val,
    y_train_val_scaled,
    X_test,
    y_test_scaled,
    y_train_original,
    y_test_original,
    y_mean,
    y_std,
    epochs=100,
):
    """
    Train one BP model with a specific regularization configuration
    and compute detailed diagnostics.

    Outputs:
    - Train and test MSE/MAE/MAPE (in original units).
    - Overall weight magnitude (sum |w|, sum w^2).
    - Difference between train and test errors to quantify overfitting.
    - Final network and test predictions (for scatter plots).
    """

    # -------------------------------------------------------------------------
    # Step 1: Define network architecture and regularization
    # -------------------------------------------------------------------------
    n_features = X_train_val.shape[1]
    layers = [n_features, 20, 10, 1]

    nn = NeuralNet(
        layers=layers,
        learning_rate=0.01,
        momentum=0.5,
        fact="relu",
        l1_reg=l1_reg,
        l2_reg=l2_reg,
    )

    print(f"\nTraining model: {name} (L1={l1_reg}, L2={l2_reg})")

    # -------------------------------------------------------------------------
    # Step 2: Train the network with standardized outputs
    # -------------------------------------------------------------------------
    nn.fit(
        X_train_val,
        y_train_val_scaled,
        epochs=epochs,
        validation_split=0.2,
    )

    # -------------------------------------------------------------------------
    # Step 3: Compute predictions for train+val and test
    # -------------------------------------------------------------------------
    y_train_pred_scaled = nn.predict(X_train_val)
    y_test_pred_scaled = nn.predict(X_test)

    if y_train_pred_scaled.ndim == 1:
        y_train_pred_scaled = y_train_pred_scaled.reshape(-1, 1)
    if y_test_pred_scaled.ndim == 1:
        y_test_pred_scaled = y_test_pred_scaled.reshape(-1, 1)

    # Descale back to original units
    y_train_pred = y_train_pred_scaled * y_std + y_mean
    y_test_pred = y_test_pred_scaled * y_std + y_mean

    # -------------------------------------------------------------------------
    # Step 4: Compute train and test metrics (MSE, MAE, MAPE)
    # -------------------------------------------------------------------------
    train_mse, train_mae, train_mape = regression_metrics(
        y_train_original, y_train_pred
    )
    test_mse, test_mae, test_mape = regression_metrics(
        y_test_original, y_test_pred
    )

    # -------------------------------------------------------------------------
    # Step 5: Compute weight magnitudes and overfitting indicators
    # -------------------------------------------------------------------------
    total_abs_weights = 0.0
    total_sq_weights = 0.0
    for l in range(1, nn.L):
        total_abs_weights += np.sum(np.abs(nn.w[l]))
        total_sq_weights += np.sum(nn.w[l] ** 2)

    mse_diff = abs(train_mse - test_mse)
    mae_diff = abs(train_mae - test_mae)

    return {
        "Model": name,
        "L1_reg": l1_reg,
        "L2_reg": l2_reg,
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train MAPE": train_mape,
        "Test MAPE": test_mape,
        "Total |w|": total_abs_weights,
        "Total w^2": total_sq_weights,
        "MSE diff (train-test)": mse_diff,
        "MAE diff (train-test)": mae_diff,
        # Objects used later for scatter plots
        "nn": nn,
        "y_test_pred": y_test_pred,
    }


def main():
    """
    Evaluate the effect of regularization on a small set of representative models.

    Requirement mapping (Optional 1):
    - Compare at least:
        * a model without regularization,
        * one with only L2,
        * one with only L1,
        * and one with combined L1+L2.
    - Analyse:
        * test performance (MSE, MAE, MAPE),
        * magnitude of weights,
        * overfitting (difference between train and test errors).
    - Produce plots for the written report (bars and scatter plots).
    """

    np.random.seed(0)

    # -------------------------------------------------------------------------
    # Step 1: Load data and standardize output (same as other parts)
    # -------------------------------------------------------------------------
    X_train_val, y_train_val, X_test, y_test = load_preprocessed_data()

    y_mean = y_train_val.mean(axis=0)
    y_std = y_train_val.std(axis=0)
    y_std[y_std == 0] = 1.0

    y_train_val_scaled = (y_train_val - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    # -------------------------------------------------------------------------
    # Step 2: Define a small set of regularization configurations to compare
    # -------------------------------------------------------------------------
    # The exact numeric values can be adjusted based on the tuning results.
    model_configs = [
        ("No regularization", 0.0, 0.0),
        ("L2 (0.001)", 0.0, 0.001),
        ("L1 (0.001)", 0.001, 0.0),
        ("L1+L2 (0.0001)", 0.0001, 0.0001),
    ]

    evaluations = []
    for name, l1_reg, l2_reg in model_configs:
        res = train_and_evaluate_model(
            name=name,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            X_train_val=X_train_val,
            y_train_val_scaled=y_train_val_scaled,
            X_test=X_test,
            y_test_scaled=y_test_scaled,
            y_train_original=y_train_val,
            y_test_original=y_test,
            y_mean=y_mean,
            y_std=y_std,
            epochs=100,
        )
        evaluations.append(res)

    # Build a summary DataFrame excluding objects (nn, y_test_pred)
    eval_df = pd.DataFrame(
        [{k: v for k, v in e.items() if k not in ("nn", "y_test_pred")} for e in evaluations]
    )

    print("\n--- Regularization evaluation summary ---")
    print(eval_df)

    eval_df.to_csv("./regularization_evaluation_summary.csv", index=False)
    print("Saved summary to regularization_evaluation_summary.csv")

    # -------------------------------------------------------------------------
    # Step 3: Plot aggregated metrics for the report
    # -------------------------------------------------------------------------

    # Test MSE vs model
    plt.figure(figsize=(10, 4))
    plt.bar(eval_df["Model"], eval_df["Test MSE"])
    plt.ylabel("Test MSE")
    plt.title("Test MSE vs regularization configuration")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("reg_test_mse.png")

    # Total |w| vs model (weight shrinkage)
    plt.figure(figsize=(10, 4))
    plt.bar(eval_df["Model"], eval_df["Total |w|"])
    plt.ylabel("Sum |weights|")
    plt.title("Total weight magnitude vs regularization configuration")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("reg_weight_magnitude.png")

    # Overfitting measure |MSE_train - MSE_test|
    plt.figure(figsize=(10, 4))
    plt.bar(eval_df["Model"], eval_df["MSE diff (train-test)"])
    plt.ylabel("|MSE_train - MSE_test|")
    plt.title("Overfitting measure vs regularization configuration")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("reg_overfitting.png")

    # -------------------------------------------------------------------------
    # Step 4: Scatter plots for the non-regularized and best-regularized models
    # -------------------------------------------------------------------------
    no_reg = next(e for e in evaluations if e["Model"] == "No regularization")
    best_reg = min(
        [e for e in evaluations if e["Model"] != "No regularization"],
        key=lambda e: e["Test MSE"],
    )

    # Scatter: no regularization
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, no_reg["y_test_pred"], alpha=0.4)
    mn = min(y_test.min(), no_reg["y_test_pred"].min())
    mx = max(y_test.max(), no_reg["y_test_pred"].max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("Real value (Purchase Amount USD)")
    plt.ylabel("Predicted value (Purchase Amount USD)")
    plt.title("No regularization – Test predictions vs real")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_no_regularization.png")

    # Scatter: best regularized model
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_reg["y_test_pred"], alpha=0.4)
    mn = min(y_test.min(), best_reg["y_test_pred"].min())
    mx = max(y_test.max(), best_reg["y_test_pred"].max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("Real value (Purchase Amount USD)")
    plt.ylabel("Predicted value (Purchase Amount USD)")
    plt.title(f"{best_reg['Model']} – Test predictions vs real")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_best_regularized.png")

    print(
        "\nSaved plots:\n"
        "  reg_test_mse.png\n"
        "  reg_weight_magnitude.png\n"
        "  reg_overfitting.png\n"
        "  scatter_no_regularization.png\n"
        "  scatter_best_regularized.png"
    )


if __name__ == "__main__":
    main()
