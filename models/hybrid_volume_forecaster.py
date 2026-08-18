import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HybridVolumeForecaster:
    def __init__(self, seasonal_periods=7, rf_params=None, use_bayesian_intervals=True):
        self.seasonal_periods = seasonal_periods
        self.rf_params = rf_params or {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": 42,
        }
        self.use_bayesian_intervals = use_bayesian_intervals

        self.hw_model = None
        self.rf_model = None
        self.bayes_model = None
        self.feature_columns_ = None
        self.last_series_ = None

    def _fit_holtwinters(self, y: pd.Series):
        self.hw_model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
        return self.hw_model

    def _create_features(self, y: pd.Series, hw_fitted: pd.Series = None, is_training: bool = True):
        """
        Unified feature engineering for both training and prediction.
        """
        df = pd.DataFrame(index=y.index)
        df["y"] = y

        # ---- Calendar features ----
        df["dow"] = y.index.dayofweek
        df["dom"] = y.index.day
        df["month"] = y.index.month
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        df["is_month_start"] = y.index.is_month_start.astype(int)
        df["is_month_end"] = y.index.is_month_end.astype(int)

        # ---- Holt-Winters components ----
        if hw_fitted is not None:
            df["hw_fitted"] = hw_fitted
            # We keep the last known level/trend and cycle the seasonal component
            df["hw_level"] = self.hw_model.level.iloc[-1]
            df["hw_trend"] = self.hw_model.trend.iloc[-1] if hasattr(self.hw_model, "trend") else 0.0

            # Seasonal component (cyclic)
            season = self.hw_model.season
            season_len = len(season)
            # Map each date to the corresponding seasonal position
            positions = np.arange(len(y)) % self.seasonal_periods
            # Use the last full seasonal cycle as reference
            last_season = season.iloc[-self.seasonal_periods :].values
            df["hw_season"] = last_season[positions % self.seasonal_periods]
        else:
            df["hw_fitted"] = np.nan
            df["hw_level"] = np.nan
            df["hw_trend"] = np.nan
            df["hw_season"] = np.nan

        # ---- Lag features ----
        for lag in [1, 2, 3, 7, 14, 21]:
            df[f"lag_{lag}"] = y.shift(lag)

        # ---- Rolling statistics ----
        for window in [7, 14, 28]:
            shifted = y.shift(1)
            df[f"roll_mean_{window}"] = shifted.rolling(window).mean()
            df[f"roll_std_{window}"] = shifted.rolling(window).std()
            df[f"roll_max_{window}"] = shifted.rolling(window).max()
            df[f"roll_min_{window}"] = shifted.rolling(window).min()

        if is_training and hw_fitted is not None:
            df["target_residual"] = df["y"] - df["hw_fitted"]
        else:
            df["target_residual"] = np.nan

        return df

    def fit(self, series: pd.Series):
        if not isinstance(series, pd.Series):
            raise ValueError("Input must be a pandas Series with DatetimeIndex")

        series = series.sort_index()
        self.last_series_ = series.copy()

        # 1. Fit Holt-Winters
        self._fit_holtwinters(series)
        hw_fitted = self.hw_model.fittedvalues

        # 2. Build features
        feat = self._create_features(series, hw_fitted, is_training=True)
        feat = feat.dropna()

        self.feature_columns_ = [
            c for c in feat.columns if c not in ["y", "target_residual"]
        ]

        X = feat[self.feature_columns_]
        y_resid = feat["target_residual"]

        # 3. Random Forest on residual
        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.rf_model.fit(X, y_resid)

        # 4. Bayesian layer (optional)
        if self.use_bayesian_intervals:
            rf_pred = self.rf_model.predict(X).reshape(-1, 1)
            self.bayes_model = BayesianRidge()
            self.bayes_model.fit(rf_pred, y_resid)

        return self

    def predict(self, steps: int = 30, alpha: float = 0.1):
        """
        Clean recursive multi-step forecast.
        """
        if self.hw_model is None or self.rf_model is None:
            raise RuntimeError("Model has not been fitted yet.")

        # Baseline Holt-Winters forecast
        hw_forecast = self.hw_model.forecast(steps)

        future_index = pd.date_range(
            start=self.last_series_.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )

        # Growing history (starts with training data)
        history = self.last_series_.copy()

        predictions = []
        lowers = []
        uppers = []

        for i, date in enumerate(future_index):
            # Temporary series including the current step (filled with HW value for feature calculation)
            temp_y = pd.concat([history, pd.Series([hw_forecast.iloc[i]], index=[date])])

            # Create features for this single date
            feat = self._create_features(temp_y, hw_fitted=None, is_training=False)

            # Override HW components for this future point
            feat.loc[date, "hw_fitted"] = hw_forecast.iloc[i]
            feat.loc[date, "hw_level"] = self.hw_model.level.iloc[-1]
            feat.loc[date, "hw_trend"] = (
                self.hw_model.trend.iloc[-1] if hasattr(self.hw_model, "trend") else 0.0
            )

            # Seasonal component (cyclic)
            season_pos = (len(history) + i) % self.seasonal_periods
            last_season = self.hw_model.season.iloc[-self.seasonal_periods :].values
            feat.loc[date, "hw_season"] = last_season[season_pos]

            # Select only the current row and required columns
            X_row = feat.loc[[date], self.feature_columns_]

            # Predict residual correction
            rf_corr = self.rf_model.predict(X_row)[0]
            final_pred = hw_forecast.iloc[i] + rf_corr

            # Uncertainty intervals
            if self.use_bayesian_intervals and self.bayes_model is not None:
                mean, std = self.bayes_model.predict(
                    np.array([[rf_corr]]), return_std=True
                )
                z = 1.645  # approx 90%
                lower = final_pred - z * std[0]
                upper = final_pred + z * std[0]
            else:
                lower = final_pred * 0.90
                upper = final_pred * 1.10

            predictions.append(final_pred)
            lowers.append(lower)
            uppers.append(upper)

            # Update history with the final prediction (important for next lags/rolling)
            history.loc[date] = final_pred

        result = pd.DataFrame(
            {
                "y_pred": predictions,
                "y_lower": lowers,
                "y_upper": uppers,
                "hw_baseline": hw_forecast.values,
            },
            index=future_index,
        )

        return result

    def evaluate(self, test_series: pd.Series):
        pred_df = self.predict(steps=len(test_series))
        y_true = test_series.values
        y_pred = pred_df["y_pred"].values

        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100,
        }
