import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ExtendedMinMaxScalar(MinMaxScaler):
    def __init__(self, feature_range=(0, 1), copy=True):
        super().__init__(feature_range=feature_range, copy=copy)

    def fit(self, X, y=None):
        # Compute min and max
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        # Modify the range to include 10% below min and 10% above max
        self.min_ = self.min_ - 0.1 * self.min_
        self.max_ = self.max_ + 0.1 * self.max_

        # Set data_min_ and data_max_ used in transformation
        self.data_min_ = self.min_
        self.data_max_ = self.max_

        # Return the fitted scaler
        return self

    def transform(self, X):
        # Apply the custom transformation based on modified range
        X_std = (X - self.data_min_) / (self.data_max_ - self.data_min_)
        X_scaled = (
            X_std * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )
        return X_scaled

    def inverse_transform(self, X):
        # Revert the scaling
        X_inv_std = (X - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
        X_inv = X_inv_std * (self.data_max_ - self.data_min_) + self.data_min_
        return X_inv

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
