#!/usr/bin/env python
"""
End-to-end AVM / EconML script:
- Load inference dataframe
- Clean & prepare features
- Estimate ZIP-level CATEs of HALF_BATHS_COAL on SALEAMT
- Produce figures summarizing effects by ZIP
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from econml.dml import LinearDML
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Data loading
# ============================================================

DATA_PATH = "/cate_scenario_analyses/data/inference_df.parquet"
# ^^ Change this if your parquet lives somewhere else


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load your AVM dataframe.
    """
    print(f"Loading data from: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded dataframe with shape: {df.shape}")
    return df


# ============================================================
# 2. Cleaning / preparation
# ============================================================

def build_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns we need and drop rows with missing values.
    Also:
      - Set ZIP_COAL as string
      - Encode ZIP_COAL to an integer code: ZIP_INT
    """

    cols = [
        "SALEAMT",
        "HALF_BATHS_COAL",
        "ZIP_COAL",
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
    before = len(df_model)
    df_model = df_model.dropna(subset=cols)
    after = len(df_model)
    print(f"Dropped {before - after} rows with missing values; remaining: {after}")

    # Ensure ZIP is string, then encode to integer codes
    df_model["ZIP_COAL"] = df_model["ZIP_COAL"].astype(str)
    df_model["ZIP_INT"] = df_model["ZIP_COAL"].astype("category").cat.codes

    return df_model


# ============================================================
# 3. EconML: ZIP-level CATEs
# ============================================================

def run_econml_zip_cates(df_model: pd.DataFrame) -> Tuple[LinearDML, pd.DataFrame, pd.Series]:
    """
    Use EconML LinearDML to estimate the effect of HALF_BATHS_COAL on SALEAMT,
    allowing the effect to vary by ZIP (heterogeneity in X=ZIP_INT), and
    aggregate effects to the ZIP level.
    """

    # Outcome
    Y = df_model["SALEAMT"].values

    # Treatment (continuous: number of half baths)
    T = df_model["HALF_BATHS_COAL"].values.reshape(-1, 1)

    # Confounders W (controls)
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

    # Heterogeneity feature X: ZIP code encoded as integer
    X = df_model[["ZIP_INT"]].values  # shape (n_samples, 1)

    # First-stage models for outcome and treatment: StandardScaler + Lasso
    model_y = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=0.001, max_iter=5000, random_state=42)),
        ]
    )

    model_t = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=0.001, max_iter=5000, random_state=42)),
        ]
    )

    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=False,  # HALF_BATHS_COAL is continuous
        cv=2,
        random_state=42,
    )

    print("Fitting LinearDML with ZIP as heterogeneity feature...")
    est.fit(Y, T, X=X, W=W)

    # Per-row CATE: effect of increasing HALF_BATHS_COAL by 1
    cate_per_row = est.effect(X, T0=0, T1=1).reshape(-1)

    # Build a dataframe for aggregation and plotting
    df_effects = pd.DataFrame(
        {
            "ZIP_COAL": df_model["ZIP_COAL"].values,
            "ZIP_INT": df_model["ZIP_INT"].values,
            "cate_per_plus1_halfbath": cate_per_row,
            "SALEAMT": df_model["SALEAMT"].values,
        }
    )

    # Average CATE within each ZIP
    zip_effects = (
        df_effects.groupby("ZIP_COAL")["cate_per_plus1_halfbath"]
        .mean()
        .sort_values()
    )

    print("\n=== ZIP-level effect of +1 HALF_BATHS_COAL on SALEAMT ===")
    print("Units: dollars change in SALEAMT for +1 half bath, on average in each ZIP.")
    print("\n--- ZIPs with smallest estimated effect ---")
    print(zip_effects.head(20))
    print("\n--- ZIPs with largest estimated effect ---")
    print(zip_effects.tail(20))

    return est, df_effects, zip_effects


# ============================================================
# 4. Plotting helpers (by ZIP)
# ============================================================

def make_zip_effects_figures(
    zip_effects: pd.Series,
    output_dir: str = "figures",
    top_n: int = 20
) -> None:
    """
    Create bar plots for:
    - Bottom N ZIPs by average CATE
    - Top N ZIPs by average CATE

    Saves PNGs into `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort once
    zip_effects_sorted = zip_effects.sort_values()

    # Bottom N (smallest effect)
    bottom = zip_effects_sorted.head(top_n)
    plt.figure(figsize=(10, 6))
    bottom.plot(kind="barh")
    plt.xlabel("Estimated Δ SALEAMT for +1 half bath (USD)")
    plt.ylabel("ZIP")
    plt.title(f"Bottom {top_n} ZIPs: smallest half-bath effect on SALEAMT")
    plt.tight_layout()
    bottom_path = os.path.join(output_dir, f"zip_effects_bottom_{top_n}.png")
    plt.savefig(bottom_path, dpi=150)
    plt.close()
    print(f"Saved bottom-{top_n} ZIP effects figure to: {bottom_path}")

    # Top N (largest effect)
    top = zip_effects_sorted.tail(top_n)
    plt.figure(figsize=(10, 6))
    top.plot(kind="barh")
    plt.xlabel("Estimated Δ SALEAMT for +1 half bath (USD)")
    plt.ylabel("ZIP")
    plt.title(f"Top {top_n} ZIPs: largest half-bath effect on SALEAMT")
    plt.tight_layout()
    top_path = os.path.join(output_dir, f"zip_effects_top_{top_n}.png")
    plt.savefig(top_path, dpi=150)
    plt.close()
    print(f"Saved top-{top_n} ZIP effects figure to: {top_path}")


def make_zip_effect_histogram(
    zip_effects: pd.Series,
    output_dir: str = "figures"
) -> None:
    """
    Plot a histogram of ZIP-level effects to show distribution.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(zip_effects.values, bins=30)
    plt.xlabel("Estimated Δ SALEAMT for +1 half bath (USD)")
    plt.ylabel("Number of ZIPs")
    plt.title("Distribution of ZIP-level half-bath effects on SALEAMT")
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "zip_effects_histogram.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved ZIP effects histogram to: {hist_path}")

import os
import matplotlib.pyplot as plt
import seaborn as sns  # optional, for prettier distribution


def make_zip_effects_figures(
    zip_effects: pd.Series,
    output_dir: str = "figures",
    top_n: int = 20
) -> None:
    """
    Create bar plots for:
    - Bottom N ZIPs by average CATE
    - Top N ZIPs by average CATE

    Saves PNGs into `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort once
    zip_effects_sorted = zip_effects.sort_values()

    # Bottom N (smallest effect)
    bottom = zip_effects_sorted.head(top_n)
    plt.figure(figsize=(10, 6))
    bottom.plot(kind="barh")
    plt.xlabel("Estimated Δ SALEAMT for +1 half bath (USD)")
    plt.ylabel("ZIP")
    plt.title(f"Bottom {top_n} ZIPs: smallest half-bath effect on SALEAMT")
    plt.tight_layout()
    bottom_path = os.path.join(output_dir, f"zip_effects_bottom_{top_n}.png")
    plt.savefig(bottom_path, dpi=150)
    plt.close()
    print(f"Saved bottom-{top_n} ZIP effects figure to: {bottom_path}")

    # Top N (largest effect)
    top = zip_effects_sorted.tail(top_n)
    plt.figure(figsize=(10, 6))
    top.plot(kind="barh")
    plt.xlabel("Estimated Δ SALEAMT for +1 half bath (USD)")
    plt.ylabel("ZIP")
    plt.title(f"Top {top_n} ZIPs: largest half-bath effect on SALEAMT")
    plt.tight_layout()
    top_path = os.path.join(output_dir, f"zip_effects_top_{top_n}.png")
    plt.savefig(top_path, dpi=150)
    plt.close()
    print(f"Saved top-{top_n} ZIP effects figure to: {top_path}")


def make_zip_effect_distribution(
    zip_effects: pd.Series,
    output_dir: str = "figures"
) -> None:
    """
    Plot the distribution of ZIP-level effects:
    - Histogram of ZIP-level CATEs
    - Optional KDE curve overlaid

    Each point is a ZIP; x-axis is $ effect for +1 half bath.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    # Histogram + KDE (seaborn handles both nicely)
    sns.histplot(zip_effects.values, bins=30, kde=True)
    plt.xlabel("Estimated Δ SALEAMT for +1 half bath (USD)")
    plt.ylabel("Number of ZIP codes")
    plt.title("Distribution of ZIP-level half-bath effects on SALEAMT")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "zip_effects_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ZIP effects distribution figure to: {out_path}")


# ============================================================
# 5. Main
# ============================================================

def main():
    # 1) Load
    df = load_data()

    # 2) Clean / prepare
    df_model = build_clean_dataframe(df)
    print(f"Using {len(df_model)} rows after cleaning.")

    # Optional: subsample if runtime is too slow
    # N_SUB = 100_000
    # if len(df_model) > N_SUB:
    #     df_model = df_model.sample(N_SUB, random_state=42)
    #     print(f"Subsampled to {len(df_model)} rows: {len(df_model)}")

    # 3) Fit EconML and get ZIP-level effects
    est, df_effects, zip_effects = run_econml_zip_cates(df_model)

    # 4) Save ZIP-level effects to CSV (optional)
    zip_effects.to_csv("zip_halfbath_effects.csv", header=["cate_per_plus1_halfbath"])
    print("Saved ZIP-level effects to zip_halfbath_effects.csv")

    # 5) Create figures
    make_zip_effects_figures(zip_effects, output_dir="../figures", top_n=20)
    make_zip_effect_histogram(zip_effects, output_dir="../figures")

    print("\nDone. Figures are in the 'figures/' directory.")
    make_zip_effect_distribution(zip_effects, output_dir="../figures")


if __name__ == "__main__":
    main()
