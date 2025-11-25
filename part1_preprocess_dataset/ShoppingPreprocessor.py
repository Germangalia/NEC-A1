import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class ShoppingPreprocessor:
    def __init__(self, csv_path, target_col="Purchase Amount (USD)", test_size=0.2, random_state=42):
        self.csv_path = csv_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.raw_df = None
        self.preprocessor = None
        self.X_train_val = None
        self.X_test = None
        self.y_train_val = None
        self.y_test = None
        self.feature_names_ = None

    def load_data(self):
        self.raw_df = pd.read_csv(self.csv_path)

    def _build_preprocessor(self, df):
        df = df.copy()
        if "Customer ID" in df.columns:
            df = df.drop(columns=["Customer ID"])
        numeric_features = ["Age", "Review Rating", "Previous Purchases"]
        numeric_features = [c for c in numeric_features if c in df.columns and c != self.target_col]
        categorical_features = [c for c in df.columns if c not in numeric_features + [self.target_col]]
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        self.numeric_features_ = numeric_features
        self.categorical_features_ = categorical_features

    def preprocess_and_split(self):
        if self.raw_df is None:
            self.load_data()
        df = self.raw_df.copy()
        if self.target_col not in df.columns:
            raise ValueError(f"Target {self.target_col} not found in the CSV.")
        self._build_preprocessor(df)
        if "Customer ID" in df.columns:
            df = df.drop(columns=["Customer ID"])
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].astype(float).values
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )
        X_train_val_proc = self.preprocessor.fit_transform(X_train_val)
        X_test_proc = self.preprocessor.transform(X_test)
        if hasattr(X_train_val_proc, "toarray"):
            X_train_val_proc = X_train_val_proc.toarray()
            X_test_proc = X_test_proc.toarray()
        self.X_train_val = X_train_val_proc.astype(np.float32)
        self.X_test = X_test_proc.astype(np.float32)
        self.y_train_val = y_train_val.astype(np.float32)
        self.y_test = y_test.astype(np.float32)
        num_names = self.numeric_features_
        cat_names = list(
            self.preprocessor.named_transformers_["cat"].get_feature_names_out(self.categorical_features_)
        )
        self.feature_names_ = num_names + cat_names

    def get_data(self):
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
