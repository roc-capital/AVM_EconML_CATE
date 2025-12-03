#!/usr/bin/env python

import numpy as np
import pandas as pd
import pymc as pm


def load_data():
    """
    Load your AVM dataframe here.
    Replace the path / format with whatever you're actually using.
    """
    # EXAMPLE: Parquet
    df = pd.read_parquet("/cate_scenario_analyses/data/inference_df.parquet")

    return df

def build_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns we need and drop rows with missing values.
    """

    cols = [
        "SALEAMT",
        "HALF_BATHS_COAL",
        "LIVINGAREASQFT_COAL",
        "LOTSIZESQFT_COAL",
        "BEDROOMS_MLS",
        "FULL_BATHS_COAL",
        "TOTAL_BATHS_COAL",
        "GARAGE_SPACES_COAL",
        "FIREPLACE_COUNT_MLS",
        "YEARBUILT_COAL",
        "EFFECTIVEYEARBUILT_COAL",
        "LATITUDE",
        "LONGITUDE",
        "HPI_MID",
        "YEAR",
    ]

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    df_model = df[cols].copy()

    # Drop rows with NaNs in any modeling column to keep PyMC happy
    df_model = df_model.dropna(subset=cols)

    return df_model


def run_pymc_model(df_model: pd.DataFrame):
    """
    Run a Bayesian linear regression in PyMC:
        SALEAMT ~ HALF_BATHS_COAL + covariates
    and print the posterior for the half-bath effect.
    """

    # Outcome
    y = df_model["SALEAMT"].values

    # --- Treatment (continuous: number of half baths) ---
    t = df_model["HALF_BATHS_COAL"].values

    # If instead you want a binary treatment (any half bath vs none), use:
    # df_model = df_model.copy()
    # df_model["treated_halfbath"] = (df_model["HALF_BATHS_COAL"] >= 1).astype(int)
    # t = df_model["treated_halfbath"].values

    covariate_names = [
        "LIVINGAREASQFT_COAL",
        "LOTSIZESQFT_COAL",
        "BEDROOMS_MLS",
        "FULL_BATHS_COAL",
        "TOTAL_BATHS_COAL",
        "GARAGE_SPACES_COAL",
        "FIREPLACE_COUNT_MLS",
        "YEARBUILT_COAL",
        "EFFECTIVEYEARBUILT_COAL",
        "LATITUDE",
        "LONGITUDE",
        "HPI_MID",
        "YEAR",
    ]

    X = df_model[covariate_names].values

    # Optional: standardize covariates to help sampling
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0  # avoid divide by zero
    X_scaled = (X - X_mean) / X_std

    # Also center the outcome to help with priors
    y_mean = y.mean()
    y_centered = y - y_mean

    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10.0)                 # intercept on centered scale
        beta_t = pm.Normal("beta_t", mu=0, sigma=10.0)               # effect of half baths
        betas = pm.Normal("betas", mu=0, sigma=5.0, shape=X_scaled.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=10.0)

        mu = alpha + beta_t * t + pm.math.dot(X_scaled, betas)

        # Likelihood (centered outcome)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_centered)

        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,          # IMPORTANT: avoid multiprocessing issues
            target_accept=0.9,
            random_seed=42,
            progressbar=True,
        )

    # Extract posterior samples for beta_t
    beta_t_samples = idata.posterior["beta_t"].values.flatten()
    beta_t_mean = beta_t_samples.mean()
    beta_t_ci = np.percentile(beta_t_samples, [2.5, 97.5])

    print("\n=== PyMC Results: Effect of HALF_BATHS_COAL on SALEAMT ===")
    print("Posterior mean (centered scale):", beta_t_mean)
    print("95% credible interval (centered scale):", beta_t_ci)

    # Since y was centered (SALEAMT - mean), the beta_t is still in dollar units.
    # Interpretation:
    #   For each additional half bath, SALEAMT changes by ~beta_t dollars on average,
    #   conditional on the included covariates.

    return idata, beta_t_mean, beta_t_ci


def main():
    df = load_data()
    df_model = build_clean_dataframe(df)
    print(f"Using {len(df_model)} rows after dropping missing values.")
    N_SUB = 10000
    if len(df_model) > N_SUB:
        df_model = df_model.sample(N_SUB, random_state=42)
        print(f"Subsampled to {len(df_model)} rows for PyMC.")

    run_pymc_model(df_model)


if __name__ == "__main__":
    main()
