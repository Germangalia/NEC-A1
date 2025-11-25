import numpy as np
from part1_preprocess_dataset.ShoppingPreprocessor import ShoppingPreprocessor


def generate_and_save_preprocessed_files():
    """
    Generate preprocessed / normalized datasets for Part 2 and Part 3.

    Requirement mapping (Assignment Part 1):
    - Uses ShoppingPreprocessor to:
        * load the original CSV dataset,
        * correctly encode categorical variables,
        * normalize numerical variables,
        * perform the random 80% / 20% train+test split (with shuffling).
    - Saves the resulting matrices to disk as .npy files so that the analysis
      part (BP, MLR-F, BP-F, Optional 1) always uses exactly the same data.
      This implements: "generate the preprocessed / normalized files that are
      going to be the input of your analysis part".
    """

    # Instantiate the preprocessor with the same configuration used throughout
    # the assignment:
    # - same dataset path,
    # - same real-valued target column (regression),
    # - same 80/20 split and random seed to ensure reproducibility.
    preprocessor = ShoppingPreprocessor(
        csv_path="./dataset/shopping_behavior.csv",
        target_col="Purchase Amount (USD)",
        test_size=0.2,
        random_state=42,
    )

    # get_data() returns:
    # - X_train_val, X_test: already normalized input features
    # - y_train_val, y_test: original target values (float32)
    # - feature_names, numeric_features, categorical_features: metadata
    data = preprocessor.get_data()

    X_train_val = data["X_train_val"]   # shape (n_train_val, n_features)
    y_train_val = data["y_train_val"]   # shape (n_train_val,)
    X_test = data["X_test"]             # shape (n_test, n_features)
    y_test = data["y_test"]             # shape (n_test,)

    # Ensure targets are stored as column vectors (n_samples, 1) so that they
    # match the expected shape in the BP implementation (NeuralNet).
    if y_train_val.ndim == 1:
        y_train_val = y_train_val.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    print("Saving preprocessed datasets to .npy files...")
    print(f"  X_train_val: {X_train_val.shape}")
    print(f"  y_train_val: {y_train_val.shape}")
    print(f"  X_test     : {X_test.shape}")
    print(f"  y_test     : {y_test.shape}")

    # Save the preprocessed datasets as NumPy binary files.
    # These files will be loaded later by the Part 2 integration script.
    np.save("./dataset/preprocessed/X_train_val.npy", X_train_val)
    np.save("./dataset/preprocessed/y_train_val.npy", y_train_val)
    np.save("./dataset/preprocessed/X_test.npy", X_test)
    np.save("./dataset/preprocessed/y_test.npy", y_test)

    print("Preprocessed files generated successfully.")
    print("Files: X_train_val.npy, y_train_val.npy, X_test.npy, y_test.npy")


if __name__ == "__main__":
    # Running this script corresponds to "completing Part 1" in terms of code:
    # the preprocessed / normalized files are generated and stored on disk.
    generate_and_save_preprocessed_files()
