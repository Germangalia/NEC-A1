import numpy as np
from part1_preprocess_dataset.ShoppingPreprocessor import ShoppingPreprocessor

def generate_and_save_preprocessed_files():
    preprocessor = ShoppingPreprocessor(
        csv_path="./dataset/shopping_behavior.csv",
        target_col="Purchase Amount (USD)",
        test_size=0.2,
        random_state=42,
    )

    data = preprocessor.get_data()

    X_train_val = data["X_train_val"]
    y_train_val = data["y_train_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    if y_train_val.ndim == 1:
        y_train_val = y_train_val.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    print("Saving preprocessed datasets to .npy files...")
    print(f"  X_train_val: {X_train_val.shape}")
    print(f"  y_train_val: {y_train_val.shape}")
    print(f"  X_test     : {X_test.shape}")
    print(f"  y_test     : {y_test.shape}")

    np.save("./dataset/preprocessed/X_train_val.npy", X_train_val)
    np.save("./dataset/preprocessed/y_train_val.npy", y_train_val)
    np.save("./dataset/preprocessed/X_test.npy", X_test)
    np.save("./dataset/preprocessed/y_test.npy", y_test)

    print("Preprocessed files generated successfully.")
    print("Files: X_train_val.npy, y_train_val.npy, X_test.npy, y_test.npy")

if __name__ == "__main__":
    generate_and_save_preprocessed_files()
