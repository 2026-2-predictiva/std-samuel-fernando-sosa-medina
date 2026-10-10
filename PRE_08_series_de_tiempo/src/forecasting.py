"""Pronostico de la serie de tiempo Sutter con Holt-Winters."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

FOLDER = "PRE_08_series_de_tiempo"
TEST_SIZE = 24


def load_data():
    dataset = pd.read_csv(f"{FOLDER}/data/sutter.csv")
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.set_index("date")
    dataset.index.freq = "MS"
    return dataset["yt_true"]


def train_test_split(series, test_size=TEST_SIZE):
    return series.iloc[:-test_size], series.iloc[-test_size:]


def fit_model(train):
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
    )
    return model.fit()


def main():
    series = load_data()
    train, test = train_test_split(series)

    fitted = fit_model(train)
    forecasts = fitted.forecast(len(test))

    metrics = pd.DataFrame(
        [
            {
                "mae": mean_absolute_error(test, forecasts),
                "rmse": np.sqrt(mean_squared_error(test, forecasts)),
                "mape": (np.abs((test - forecasts) / test)).mean() * 100,
            }
        ]
    )
    metrics.to_csv(f"{FOLDER}/submission/metrics.csv", index=False)

    forecasts_df = pd.DataFrame(
        {
            "date": test.index,
            "yt_true": test.values,
            "yt_forecast": forecasts.values,
        }
    )
    forecasts_df.to_csv(f"{FOLDER}/submission/forecasts.csv", index=False)


if __name__ == "__main__":
    main()
