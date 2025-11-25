import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class ShoppingPreprocessor:
    def __init__(
        self,
        csv_path,
        target_col="Purchase Amount (USD)",
        test_size=0.2,
        random_state=42,
    ):
        # Path to the original raw CSV file (Part 1: selected dataset)
        self.csv_path = csv_path

        # Name of the target/output variable (Part 1: real-valued prediction variable)
        self.target_col = target_col

        # Fraction of data used as test set (Part 1: 80% train+val, 20% test)
        self.test_size = test_size

        # Seed to make the random split reproducible
        self.random_state = random_state

        # Internal attributes to store the raw data and the preprocessing pipeline
        self.raw_df = None
        self.preprocessor = None  # ColumnTransformer that applies scaling + one-hot

        # Arrays for preprocessed data (Part 1: “preprocessed / normalized files”)
        self.X_train_val = None
        self.X_test = None
        self.y_train_val = None
        self.y_test = None
        self.feature_names_ = None

    def load_data(self):
        """
        Load the raw CSV file into a pandas DataFrame.
        Part 1: this is the starting point after selecting the dataset.
        """
        self.raw_df = pd.read_csv(self.csv_path)

    def _build_preprocessor(self, df):
        """
        Create the ColumnTransformer with:
        - standardization for numerical features
        - one-hot encoding for categorical features

        Part 1:
        - “represent correctly categorical values” → OneHotEncoder
        - “data normalization/transformation” → StandardScaler on numerical features
        """
        df = df.copy()

        # Drop pure identifier column (it does not contain useful information for prediction)
        # Part 1: remove ID so it is not used as an input feature
        if "Customer ID" in df.columns:
            df = df.drop(columns=["Customer ID"])

        # Numerical features (excluding the target)
        # Part 1: numerical inputs that will be scaled/standardized
        numeric_features = ["Age", "Review Rating", "Previous Purchases"]
        numeric_features = [
            c for c in numeric_features
            if c in df.columns and c != self.target_col
        ]

        # Categorical features = all remaining features except target and numeric ones
        # Part 1: categorical inputs that will be one-hot encoded
        categorical_features = [
            c for c in df.columns
            if c not in numeric_features + [self.target_col]
        ]

        # Transformer for numerical features: standardization (mean 0, std 1)
        numeric_transformer = StandardScaler()

        # Transformer for categorical features: one-hot encoding
        # handle_unknown="ignore" ensures robustness if new categories appear
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        # ColumnTransformer that applies the two transformations in parallel
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Save the feature names for later inspection
        self.numeric_features_ = numeric_features
        self.categorical_features_ = categorical_features

    def preprocess_and_split(self):
        """
        Full preprocessing pipeline:

        1) Load raw data if it has not been loaded.
        2) Build the preprocessing pipeline (scaling + one-hot).
        3) Split data into train+validation and test (80% / 20%, shuffled).
        4) Apply the preprocessing pipeline to X.

        Part 1:
        - Random 80% / 20% split, shuffling the original data.
        - Generate preprocessed / normalized input data for later analysis.
        """
        # Load data if needed
        if self.raw_df is None:
            self.load_data()

        df = self.raw_df.copy()

        # Check that the target column exists in the dataset
        # Part 1: ensure that the chosen prediction variable is present and real-valued
        if self.target_col not in df.columns:
            raise ValueError(f"Target {self.target_col} not found in the CSV.")

        # Build the preprocessing pipeline using the current DataFrame
        self._build_preprocessor(df)

        # Drop the ID column again (if present) before splitting
        if "Customer ID" in df.columns:
            df = df.drop(columns=["Customer ID"])

        # Separate inputs X and target y
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].astype(float).values  # regression target

        # Random 80% / 20% split into train+validation and test
        # Part 1 requirement:
        # “Select randomly 80% of the patterns for training and validation,
        #  and the remaining 20% for test; it is important to shuffle the original data.”
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        # Fit the preprocessing pipeline using ONLY the train+validation data
        # Part 1: normalization / transformation learned only from training data
        X_train_val_proc = self.preprocessor.fit_transform(X_train_val)

        # Apply the same transformation to the test set
        X_test_proc = self.preprocessor.transform(X_test)

        # Ensure dense NumPy arrays (ColumnTransformer may return a sparse matrix).
        # This is important so that the custom BP implementation in Part 2
        # can work with standard dense arrays (n_samples, n_features) without
        # crashing when converting types.
        if hasattr(X_train_val_proc, "toarray"):
            X_train_val_proc = X_train_val_proc.toarray()
            X_test_proc = X_test_proc.toarray()

        # Store the resulting NumPy arrays (float32 for later use in neural networks)
        # Part 1 requirement: "generate the preprocessed / normalized files that are
        # going to be the input of your analysis part (BP, MLR-F, BP-F, Optional 1)".
        self.X_train_val = X_train_val_proc.astype(np.float32)
        self.X_test = X_test_proc.astype(np.float32)
        self.y_train_val = y_train_val.astype(np.float32)
        self.y_test = y_test.astype(np.float32)

        # Build the list of feature names after preprocessing
        # (numerical + one-hot encoded categorical)
        num_names = self.numeric_features_
        cat_names = list(
            self.preprocessor.named_transformers_["cat"]
            .get_feature_names_out(self.categorical_features_)
        )
        self.feature_names_ = num_names + cat_names

    def get_data(self):
        """
        Return all objects needed for the rest of the assignment:

        - X_train_val, y_train_val: preprocessed data for training and validation
        - X_test, y_test: preprocessed data for final testing
        - feature_names: names of all input features after preprocessing
        - numeric_features, categorical_features: original feature lists

        Part 1:
        - This corresponds to “generate the preprocessed / normalized files that are
          going to be the input of your analysis part” (BP, MLR-F, BP-F, Optional 1).
        """
        # If preprocessing has not been run yet, run it now
        if self.X_train_val is None:
            self.preprocess_and_split()

        return {
            "X_train_val": self.X_train_val,
            "y_train_val": self.y_train_val,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "feature_names": self.feature_names_,
            "numeric_features": getattr(self, "numeric_features_", None),
            "categorical_features": getattr(self, "categorical_features_", None),
        }
