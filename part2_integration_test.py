import numpy as np
from part2_bp_implementation.NeuralNet import NeuralNet


def load_pre_data():
    # TODO: check paths, not sure if these files are already generated
    X_tv = np.load("./data_proc/tmp/X_tv.npy")
    y_tv = np.load("./data_proc/tmp/y_tv.npy")
    X_tst = np.load("./data_proc/tmp/X_tst.npy")
    y_tst = np.load("./data_proc/tmp/y_tst.npy")

    if y_tv.ndim == 1:
        y_tv = y_tv.reshape(-1, 1)
    if y_tst.ndim == 1:
        y_tst = y_tst.reshape(-1, 1)

    print(f"train_val shapes: {X_tv.shape}, {y_tv.shape}")
    print(f"test shapes: {X_tst.shape}, {y_tst.shape}")

    return X_tv, y_tv, X_tst, y_tst


def integrate_data():
    X_tv, y_tv, X_tst, y_tst = load_pre_data()

    y_m = y_tv.mean(axis=0)
    y_s = y_tv.std(axis=0)
    y_s[y_s == 0] = 1

    y_tv_s = (y_tv - y_m) / y_s
    y_tst_s = (y_tst - y_m) / y_s

    print("output scaling:")
    print(" mean:", y_m)
    print(" std :", y_s)

    feats = X_tv.shape[1]
    layers_cfg = [feats, 10, 6, 1]

    #TODO Check files
    net = NeuralNet(
        layers=layers_cfg,
        learning_rate=0.015,
        momentum=0.8,
        fact="sigm",            # TODO verify implementation
        l1_reg=0.0,
        l2_reg=0.0,
    )

    print("Current architecture:", layers_cfg)
    print("lr:", net.learning_rate, "mom:", net.momentum)

    ep = 30  # provisional, probably too few

    print("Training (still without early stopping)...")
    tr_loss, vl_loss = net.fit(
        X=X_tv,
        y=y_tv_s,
        epochs=ep,
        validation_split=0.25,
    )
    print("Training finished (early version).")

    print("Predicting on test set...")
    y_pred_s = net.predict(X_tst)

    if y_pred_s.ndim == 1:
        y_pred_s = y_pred_s.reshape(-1, 1)

    y_pred = y_pred_s * y_s + y_m

    mse = float(np.mean((y_tst - y_pred)**2))
    mae = float(np.mean(np.abs(y_tst - y_pred)))
    # dividing by zero here may cause issues, needs correction later
    mape = float(np.mean(np.abs((y_tst - y_pred) / (y_tst + 1e-9))) * 100)

    print("Performance (initial version):")
    print(" MSE :", mse)
    print(" MAE :", mae)
    print(" MAPE:", mape)

    hist_tr, hist_val = net.loss_epochs()
    print("history shapes:", hist_tr.shape, hist_val.shape)

    return (
        net,
        (X_tv, X_tst, y_tv_s, y_tst_s),
        (tr_loss, vl_loss),
        (mse, mae, mape),
    )


if __name__ == "__main__":
    model, data, losses, metrics = integrate_data()
    print("Preliminary integration completed.")
