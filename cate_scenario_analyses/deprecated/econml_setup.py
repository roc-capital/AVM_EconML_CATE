#!/usr/bin/env python

import numpy as np
import pandas as pd

from econml.dml import LinearDML
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data():
    """
    Load your AVM dataframe here.
    Replace with your real path / format.
    """
    # Example:
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
    df_model = df_model.dropna(subset=cols)

    return df_model


def run_econml_linear_dml(df_model: pd.DataFrame):
    """
    Use EconML LinearDML to estimate the causal effect of HALF_BATHS_COAL on SALEAMT,
    controlling for the other house features.
    """

    # Outcome
    Y = df_model["SALEAMT"].values

    # Treatment (continuous: number of half baths)
    T = df_model["HALF_BATHS_COAL"].values.reshape(-1, 1)

    # Controls W (confounders)
    control_cols = [
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
    W = df_model[control_cols].values

    # No X for heterogeneity (we just want global effect),
    # so we set X=None and put all confounders into W.
    X = None

    # First-stage models for outcome and treatment
    # Pipeline: Standardize -> LassoCV
    model_y = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=3, random_state=42)),
        ]
    )

    model_t = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=3, random_state=42)),
        ]
    )

    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=False,  # HALF_BATHS_COAL is numeric/continuous
        cv=3,
        random_state=42,
    )

    print("Fitting LinearDML (this may take a bit on large data)...")
    est.fit(Y, T, X=X, W=W)

    # ATE of increasing HALF_BATHS_COAL by 1 (e.g., from k to k+1)
    # For linear-in-treatment estimators, T0=0, T1=1 gives per-unit effect.
    ate = est.ate(X=X, T0=0, T1=1)
    ate_lb, ate_ub = est.ate_interval(X=X, T0=0, T1=1)

    print("\n=== EconML LinearDML: Effect of HALF_BATHS_COAL on SALEAMT ===")
    print(f"ATE (per +1 half bath): {ate:,.2f}")
    print(f"95% CI: [{ate_lb:,.2f}, {ate_ub:,.2f}]")
    print(
        "\nInterpretation: Increasing HALF_BATHS_COAL by 1 (e.g., adding one half bath) "
        "changes SALEAMT by this many dollars on average, after adjusting for the controls."
    )

    return est, ate, (ate_lb, ate_ub)


def main():
    df = load_data()
    df_model = build_clean_dataframe(df)
    print(f"Using {len(df_model)} rows after dropping missing values.")

    # If runtime is too slow, you can subsample here, e.g.:
    # df_model = df_model.sample(50000, random_state=42)
    # print(f"Subsampled to {len(df_model)} rows for EconML.")

    run_econml_linear_dml(df_model)


if __name__ == "__main__":
    main()
