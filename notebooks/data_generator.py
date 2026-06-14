import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

def geometric_adstock_np(x: np.ndarray, alpha: float, l_max: int = 8, normalize: bool = True) -> np.ndarray:
    """Pure numpy geometric adstock transformation."""
    weights = np.array([alpha ** i for i in range(l_max)])
    if normalize:
        weights = weights / weights.sum()
    
    result = np.zeros_like(x)
    for t in range(len(x)):
        for l in range(min(l_max, t + 1)):
            result[t] += weights[l] * x[t - l]
    return result

def logistic_saturation_np(x: np.ndarray, lam: float) -> np.ndarray:
    """Pure numpy logistic saturation transformation."""
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))

def data_generator(start_date, periods, channels, spend_scalar, adstock_alphas, saturation_lamdas, betas, freq="W"):
    """
    Generates a synthetic dataset for MMM with trend, seasonality, and channel-specific contributions.
    """
    # 0. Create time dimension
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    df = pd.DataFrame({'date': date_range})

    # 1. Trend
    df["trend"] = (np.linspace(start=0.0, stop=20, num=periods) + 5) ** (1 / 8) - 1

    # 2. Seasonality
    df["seasonality"] = 0.1 * np.sin(2 * np.pi * df.index / 52)

    # 3. Demand
    df["demand"] = df["trend"] * (1 + df["seasonality"]) + np.random.normal(0, 0.10, periods)
    df["demand"] = df["demand"] * 1000

    # 4. Demand proxy
    df["demand_proxy"] = np.abs(df["demand"] * np.random.normal(1, 0.10, periods))

    # 5. Initialize sales
    df["sales"] = df["demand"]

    # 6. Channel contributions
    for i, channel in enumerate(channels):

        # Raw spend
        df[f"{channel}_spend_raw"] = df["demand"] * spend_scalar[i]
        df[f"{channel}_spend_raw"] = np.abs(df[f"{channel}_spend_raw"] * np.random.normal(1, 0.30, periods))

        # Scale spend
        scaler = MaxAbsScaler().fit(df[f"{channel}_spend_raw"].values.reshape(-1, 1))
        df[f"{channel}_spend"] = scaler.transform(df[f"{channel}_spend_raw"].values.reshape(-1, 1)).flatten()

        # Adstock
        df[f"{channel}_adstock"] = geometric_adstock_np(
            x=df[f"{channel}_spend"].values,
            alpha=adstock_alphas[i],
            l_max=8,
            normalize=True
        )

        # Saturation
        df[f"{channel}_saturated"] = logistic_saturation_np(
            x=df[f"{channel}_adstock"].values,
            lam=saturation_lamdas[i]
        )

        # Contribution
        df[f"{channel}_sales"] = df[f"{channel}_saturated"] * betas[i]
        df["sales"] += df[f"{channel}_sales"]

    return df