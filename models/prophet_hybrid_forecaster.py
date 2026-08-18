import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


class ProphetHybridForecaster:
    """
    Hybrid forecasting model combining Facebook's Prophet with Random Forest residual modeling
    and Bayesian uncertainty intervals.
    
    Architecture:
    1. Prophet: Captures trend, seasonality, and holiday effects
    2. Random Forest: Learns residual patterns not captured by Prophet
    3. Bayesian Ridge: Provides calibrated uncertainty intervals
    """
    
    def __init__(
        self,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        rf_params=None,
        use_bayesian_intervals=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.90,
    ):
        """
        Parameters
        ----------
        yearly_seasonality : bool
            Whether to fit yearly seasonality
        weekly_seasonality : bool
            Whether to fit weekly seasonality
        daily_seasonality : bool
            Whether to fit daily seasonality
        rf_params : dict
            Random Forest hyperparameters
        use_bayesian_intervals : bool
            Whether to use Bayesian Ridge for uncertainty intervals
        changepoint_prior_scale : float
            Controls flexibility of trend changes
        seasonality_prior_scale : float
            Controls strength of seasonality
        interval_width : float
            Confidence interval width (e.g., 0.90 for 90%)
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        self.use_bayesian_intervals = use_bayesian_intervals
        
        self.rf_params = rf_params or {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": 42,
        }
        
        self.prophet_model = None
        self.rf_model = None
        self.bayes_model = None
        self.feature_columns_ = None
        self.last_series_ = None
        self.training_dates_ = None

    def _fit_prophet(self, df: pd.DataFrame):
        """Fit Prophet model to training data."""
        self.prophet_model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=self.interval_width,
        )
        self.prophet_model.fit(df)
        return self.prophet_model

    def _create_features(self, y: pd.Series, prophet_fitted: pd.Series = None, is_training: bool = True):
        """
        Create features for Random Forest from Prophet components and time series statistics.
        """
        df = pd.DataFrame(index=y.index)
        df["y"] = y

        # ---- Calendar features ----
        df["dow"] = y.index.dayofweek
        df["dom"] = y.index.day
        df["month"] = y.index.month
        df["quarter"] = y.index.quarter
        df["doy"] = y.index.dayofyear
        df["week"] = y.index.isocalendar().week
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        df["is_month_start"] = y.index.is_month_start.astype(int)
        df["is_month_end"] = y.index.is_month_end.astype(int)

        # ---- Prophet components ----
        if prophet_fitted is not None:
            df["prophet_fitted"] = prophet_fitted
        else:
            df["prophet_fitted"] = np.nan

        # ---- Lag features ----
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f"lag_{lag}"] = y.shift(lag)

        # ---- Rolling statistics ----
        for window in [7, 14, 28]:
            shifted = y.shift(1)
            df[f"roll_mean_{window}"] = shifted.rolling(window).mean()
            df[f"roll_std_{window}"] = shifted.rolling(window).std()
            df[f"roll_max_{window}"] = shifted.rolling(window).max()
            df[f"roll_min_{window}"] = shifted.rolling(window).min()
            df[f"roll_median_{window}"] = shifted.rolling(window).median()

        # ---- Differencing features ----
        df["diff_1"] = y.diff(1)
        df["diff_7"] = y.diff(7)
        
        # ---- Volatility features ----
        df["volatility_7"] = y.rolling(7).std()
        df["volatility_14"] = y.rolling(14).std()

        if is_training and prophet_fitted is not None:
            df["target_residual"] = df["y"] - df["prophet_fitted"]
        else:
            df["target_residual"] = np.nan

        return df

    def add_holidays(self, holidays_df: pd.DataFrame):
        """
        Add custom holidays to the model.
        
        Parameters
        ----------
        holidays_df : pd.DataFrame
            DataFrame with columns: 'holiday', 'ds', 'lower_window', 'upper_window'
        """
        if self.prophet_model is None:
            self.prophet_model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                interval_width=self.interval_width,
                holidays=holidays_df,
            )
        else:
            self.prophet_model.holidays = holidays_df

    def add_regressors(self, df: pd.DataFrame, regressor_cols: list):
        """
        Add external regressors to Prophet model.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the regressor columns
        regressor_cols : list
            List of column names to add as regressors
        """
        if self.prophet_model is None:
            raise RuntimeError("Fit the model first before adding regressors")
        
        for col in regressor_cols:
            self.prophet_model.add_regressor(col)

    def fit(self, series: pd.Series, exogenous_df: pd.DataFrame = None):
        """
        Fit the hybrid forecaster.
        
        Parameters
        ----------
        series : pd.Series
            Time series with DatetimeIndex
        exogenous_df : pd.DataFrame, optional
            DataFrame with external regressors (must have 'ds' column)
        """
        if not isinstance(series, pd.Series):
            raise ValueError("Input must be a pandas Series with DatetimeIndex")

        series = series.sort_index()
        self.last_series_ = series.copy()
        self.training_dates_ = series.index

        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            "ds": series.index,
            "y": series.values
        })

        # Add exogenous variables if provided
        if exogenous_df is not None:
            prophet_df = prophet_df.merge(exogenous_df, on="ds", how="left")

        # 1. Fit Prophet
        self._fit_prophet(prophet_df)
        prophet_forecast = self.prophet_model.predict(prophet_df[["ds"] + list(exogenous_df.columns) if exogenous_df is not None else ["ds"]])
        prophet_fitted = prophet_forecast["yhat"].values

        # 2. Build features
        feat = self._create_features(series, prophet_fitted, is_training=True)
        feat = feat.dropna()

        self.feature_columns_ = [
            c for c in feat.columns if c not in ["y", "target_residual"]
        ]

        X = feat[self.feature_columns_]
        y_resid = feat["target_residual"]

        # 3. Fit Random Forest on residuals
        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.rf_model.fit(X, y_resid)

        # 4. Bayesian layer for uncertainty calibration
        if self.use_bayesian_intervals:
            rf_pred = self.rf_model.predict(X).reshape(-1, 1)
            self.bayes_model = BayesianRidge()
            self.bayes_model.fit(rf_pred, y_resid)

        return self

    def predict(self, steps: int = 30, exogenous_future: pd.DataFrame = None):
        """
        Generate multi-step forecasts with uncertainty intervals.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        exogenous_future : pd.DataFrame, optional
            Future values of external regressors
            
        Returns
        -------
        pd.DataFrame
            Forecast with columns: y_pred, y_lower, y_upper, prophet_baseline, rf_correction
        """
        if self.prophet_model is None or self.rf_model is None:
            raise RuntimeError("Model has not been fitted yet.")

        # Create future dates
        future_index = pd.date_range(
            start=self.last_series_.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )

        # Prepare future dataframe for Prophet
        future_df = pd.DataFrame({"ds": future_index})
        
        if exogenous_future is not None:
            future_df = future_df.merge(exogenous_future, on="ds", how="left")

        # Prophet baseline forecast
        prophet_forecast = self.prophet_model.predict(future_df)
        prophet_pred = prophet_forecast["yhat"].values

        # Growing history for feature engineering
        history = self.last_series_.copy()

        predictions = []
        lowers = []
        uppers = []
        rf_corrections = []

        for i, date in enumerate(future_index):
            # Temporary series for feature creation
            temp_y = pd.concat([history, pd.Series([prophet_pred[i]], index=[date])])

            # Create features
            feat = self._create_features(temp_y, prophet_fitted=None, is_training=False)

            # Override Prophet component
            feat.loc[date, "prophet_fitted"] = prophet_pred[i]

            # Extract features for this timestep
            X_row = feat.loc[[date], self.feature_columns_]

            # Predict residual correction
            rf_corr = self.rf_model.predict(X_row)[0]
            final_pred = prophet_pred[i] + rf_corr

            # Uncertainty intervals
            if self.use_bayesian_intervals and self.bayes_model is not None:
                mean, std = self.bayes_model.predict(
                    np.array([[rf_corr]]), return_std=True
                )
                z = 1.645  # ~90% confidence
                lower = final_pred - z * std[0]
                upper = final_pred + z * std[0]
            else:
                # Use Prophet's native intervals
                prophet_lower = prophet_forecast.iloc[i]["yhat_lower"]
                prophet_upper = prophet_forecast.iloc[i]["yhat_upper"]
                lower = prophet_lower + rf_corr
                upper = prophet_upper + rf_corr

            predictions.append(final_pred)
            lowers.append(lower)
            uppers.append(upper)
            rf_corrections.append(rf_corr)

            # Update history with prediction
            history.loc[date] = final_pred

        result = pd.DataFrame(
            {
                "y_pred": predictions,
                "y_lower": lowers,
                "y_upper": uppers,
                "prophet_baseline": prophet_pred,
                "rf_correction": rf_corrections,
            },
            index=future_index,
        )

        return result

    def evaluate(self, test_series: pd.Series, exogenous_test: pd.DataFrame = None):
        """
        Evaluate model on test data.
        
        Parameters
        ----------
        test_series : pd.Series
            Test time series
        exogenous_test : pd.DataFrame, optional
            Test external regressors
            
        Returns
        -------
        dict
            Evaluation metrics (MAE, RMSE, MAPE)
        """
        pred_df = self.predict(steps=len(test_series), exogenous_future=exogenous_test)
        y_true = test_series.values
        y_pred = pred_df["y_pred"].values

        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100,
            "Mean_Prediction_Width": (pred_df["y_upper"] - pred_df["y_lower"]).mean(),
        }

    def get_prophet_components(self):
        """
        Extract and return Prophet's decomposed components.
        
        Returns
        -------
        dict
            Dictionary containing trend, weekly, yearly components
        """
        if self.prophet_model is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        return {
            "trend": self.prophet_model.trend,
            "yearly": self.prophet_model.yearly_seasonality,
            "weekly": self.prophet_model.weekly_seasonality,
        }

    def get_feature_importance(self):
        """
        Get Random Forest feature importance for residual prediction.
        
        Returns
        -------
        pd.Series
            Feature importance ranked by importance
        """
        if self.rf_model is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        importance = pd.Series(
            self.rf_model.feature_importances_,
            index=self.feature_columns_
        ).sort_values(ascending=False)
        
        return importance
