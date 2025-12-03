#!/usr/bin/env python
"""
- You define:
    * outcome column (Y_COL)
    * treatment columns (TREATMENT_COLS)
    * control columns (CONTROL_COLS)
    * ZIP column (ZIP_COL)
    * heterogeneity column(s) X_COLS (e.g., ZIP_INT)
    * SCENARIOS: dict of {scenario_name: {treatment_col: delta}}

- Script:
    * Loads data
    * Cleans / prepares
    * Fits a single LinearDML model
    * Computes CATEs for each scenario (joint changes in treatments)
    * Aggregates CATEs by ZIP
    * Writes CSV + figures for each scenario
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from econml.dml import LinearDML
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# 0. CONFIG
# ============================================================

# TODO: update this path to your parquet location
DATA_PATH = "/cate_scenario_analyses/data/inference_df.parquet"

Y_COL = "SALEAMT"

# Treatments: you can add/remove as needed
TREATMENT_COLS = [
    "HALF_BATHS_COAL",
    "BEDROOMS_MLS",
]

# Controls / confounders
CONTROL_COLS = [
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

ZIP_COL = "ZIP_COAL"     # original ZIP field
X_COLS = ["ZIP_INT"]     # heterogeneity features (ZIP encoded as integer)

# Scenarios:
#   key = scenario name
#   value = dict { treatment_column: delta }
#   Any treatment not mentioned in a scenario gets delta 0 (held fixed).
SCENARIOS: Dict[str, Dict[str, float]] = {
    # +1 half bath only
    "half_plus1": {
        "HALF_BATHS_COAL": 1.0,
        "BEDROOMS_MLS": 0.0,
    },
    # +1 bedroom only
    "bed_plus1": {
        "HALF_BATHS_COAL": 0.0,
        "BEDROOMS_MLS": 1.0,
    },
    # +1 half bath AND +1 bedroom
    "joint_half1_bed1": {
        "HALF_BATHS_COAL": 1.0,
        "BEDROOMS_MLS": 1.0,
    },
    # example with non-integer increments
    "joint_half1_5_bed0_5": {
        "HALF_BATHS_COAL": 1.5,
        "BEDROOMS_MLS": 0.5,
    },
}


# ============================================================
# 1. Data loading
# ============================================================

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

    required_cols = (
        [Y_COL, ZIP_COL]
        + TREATMENT_COLS
        + CONTROL_COLS
    )
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    df_model = df[required_cols].copy()
    before = len(df_model)
    df_model = df_model.dropna(subset=required_cols)
    after = len(df_model)
    print(f"Dropped {before - after} rows with missing values; remaining: {after}")

    # Ensure ZIP is string, then encode to integer codes
    df_model[ZIP_COL] = df_model[ZIP_COL].astype(str)
    df_model["ZIP_INT"] = df_model[ZIP_COL].astype("category").cat.codes

    return df_model


# ============================================================
# 3. EconML helper: normalize const_marginal_effect to 2D
# ============================================================

def _theta_2d_from_const_marginal_effect(est: LinearDML, X: np.ndarray) -> np.ndarray:
    """
    Normalize est.const_marginal_effect(X) to shape (n_samples, d_treatments).

    EconML often returns shape (n, d_y, d_t). We assume scalar outcome (d_y=1)
    and squeeze that dimension.
    """
    theta = est.const_marginal_effect(X)

    if theta.ndim == 2:
        # (n, d_t)
        return theta

    if theta.ndim == 3:
        # (n, d_y, d_t) -> expect d_y == 1
        if theta.shape[1] != 1:
            raise ValueError(
                f"Expected single-outcome (d_y=1), but const_marginal_effect "
                f"returned shape {theta.shape}. Multi-outcome not supported in this wrapper."
            )
        return theta[:, 0, :]  # -> (n, d_t)

    raise ValueError(
        f"Unexpected const_marginal_effect shape {theta.shape}. "
        "Expected (n, d_t) or (n, 1, d_t)."
    )


# ============================================================
# 4. EconML wrapper: arbitrary treatments & deltas
# ============================================================

def fit_linear_dml(
    df_model: pd.DataFrame,
    y_col: str,
    t_cols: List[str],
    w_cols: List[str],
    x_cols: List[str],
) -> Tuple[LinearDML, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a LinearDML with continuous multi-dimensional treatment.

    Returns:
      - fitted estimator
      - Y, T, X arrays (for reuse in scenario effects)
    """
    Y = df_model[y_col].values
    T = df_model[t_cols].values        # shape (n, d_treatments)
    W = df_model[w_cols].values
    X = df_model[x_cols].values

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
        discrete_treatment=False,
        cv=2,
        random_state=42,
    )

    print("Fitting LinearDML with treatments:", t_cols)
    est.fit(Y, T, X=X, W=W)

    # Just log what EconML thinks the marginal effect shape is,
    # but don't enforce equality here anymore.
    theta_sample = _theta_2d_from_const_marginal_effect(est, X[:5])
    print(f"Sample const_marginal_effect shape after normalization: {theta_sample.shape}")

    return est, Y, T, X


def compute_scenario_effects_per_row(
    est: LinearDML,
    df_model: pd.DataFrame,
    t_cols: List[str],
    X: np.ndarray,
    scenarios: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    For each scenario (name -> {treatment: delta}),
    compute the per-row CATE for the joint change in treatments.

    Uses const_marginal_effect(X) (normalized to 2D) @ delta_vec.

    Returns:
      df_effects: columns [ZIP, ZIP_INT, per-scenario CATEs, SALEAMT, ...]
    """

    n = len(df_model)

    # Base info
    df_effects = pd.DataFrame(
        {
            ZIP_COL: df_model[ZIP_COL].values,
            "ZIP_INT": df_model["ZIP_INT"].values,
            Y_COL: df_model[Y_COL].values,
        }
    )

    # Constant marginal CATE wrt each treatment dimension
    theta_full = _theta_2d_from_const_marginal_effect(est, X)
    n_samples, d_t_model = theta_full.shape
    print(f"Full const_marginal_effect 2D shape: {theta_full.shape}")

    if d_t_model < len(t_cols):
        raise ValueError(
            f"const_marginal_effect has only {d_t_model} treatment dims, "
            f"but TREATMENT_COLS has {len(t_cols)}: {t_cols}. "
            "We can't map fewer dimensions to more treatments."
        )

    # If EconML has extra treatment dims, just use the first len(t_cols)
    if d_t_model > len(t_cols):
        print(
            f"Warning: const_marginal_effect has {d_t_model} dims, "
            f"but TREATMENT_COLS has {len(t_cols)}. "
            f"Using only the first {len(t_cols)} columns."
        )
    theta_2d = theta_full[:, : len(t_cols)]  # shape (n, len(t_cols))

    for scenario_name, deltas in scenarios.items():
        # Build delta vector aligned with t_cols
        delta_vec = np.array([deltas.get(col, 0.0) for col in t_cols])
        print(f"Computing effects for scenario '{scenario_name}' with ΔT={delta_vec} ...")

        # effects_i = theta_i @ delta_vec
        # theta_2d: (n, d_t_used), delta_vec: (d_t_used,)
        effects = theta_2d @ delta_vec  # shape (n,)

        col_name = f"cate_{scenario_name}"
        df_effects[col_name] = effects

    return df_effects


def aggregate_zip_effects(
    df_effects: pd.DataFrame,
    scenarios: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Aggregate per-row CATEs to ZIP-level means for each scenario.
    """

    scenario_cols = [f"cate_{name}" for name in scenarios.keys()]
    group = df_effects.groupby(ZIP_COL)[scenario_cols]

    zip_effects_df = group.mean()

    # For printing, just sort by the first scenario
    zip_effects_df = zip_effects_df.sort_values(by=scenario_cols[0])

    print("\n=== ZIP-level CATEs (mean per ZIP) ===")
    print(zip_effects_df.head(10))
    print("...")
    print(zip_effects_df.tail(10))

    return zip_effects_df


# ============================================================
# 5. Plotting helpers
# ============================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_zip_bar_top_bottom(
    zip_effects_df: pd.DataFrame,
    column: str,
    output_dir: str,
    top_n: int = 20,
    title_prefix: str = ""
) -> None:
    """
    Generic helper to plot bottom/top N ZIPs for a given effect column.
    """
    _ensure_dir(output_dir)
    series = zip_effects_df[column].sort_values()

    # Bottom N
    bottom = series.head(top_n)
    plt.figure(figsize=(10, 6))
    bottom.plot(kind="barh")
    plt.xlabel("Estimated effect on SALEAMT (USD)")
    plt.ylabel("ZIP")
    plt.title(f"{title_prefix} - Bottom {top_n} ZIPs")
    plt.tight_layout()
    out_path_bottom = os.path.join(output_dir, f"{column}_bottom_{top_n}.png")
    plt.savefig(out_path_bottom, dpi=150)
    plt.close()
    print(f"Saved: {out_path_bottom}")

    # Top N
    top = series.tail(top_n)
    plt.figure(figsize=(10, 6))
    top.plot(kind="barh")
    plt.xlabel("Estimated effect on SALEAMT (USD)")
    plt.ylabel("ZIP")
    plt.title(f"{title_prefix} - Top {top_n} ZIPs")
    plt.tight_layout()
    out_path_top = os.path.join(output_dir, f"{column}_top_{top_n}.png")
    plt.savefig(out_path_top, dpi=150)
    plt.close()
    print(f"Saved: {out_path_top}")


def plot_zip_effect_distributions(
    zip_effects_df: pd.DataFrame,
    scenarios: Dict[str, Dict[str, float]],
    output_dir: str = "figures"
) -> None:
    """
    Plot distributions of ZIP-level effects for each scenario.
    """
    _ensure_dir(output_dir)

    for scenario_name in scenarios.keys():
        col = f"cate_{scenario_name}"
        plt.figure(figsize=(9, 5))
        sns.histplot(zip_effects_df[col].values, bins=30, kde=True)
        plt.xlabel("Estimated effect on SALEAMT (USD)")
        plt.ylabel("Number of ZIP codes")
        plt.title(f"Distribution of ZIP-level CATE: {scenario_name}")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{col}_distribution.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


def make_all_zip_figures(
    zip_effects_df: pd.DataFrame,
    scenarios: Dict[str, Dict[str, float]],
    output_dir: str = "figures"
) -> None:
    """
    Make:
      - top/bottom ZIP bar charts for each scenario
      - distributions for each scenario
    """
    _ensure_dir(output_dir)

    for scenario_name, deltas in scenarios.items():
        col = f"cate_{scenario_name}"
        title_prefix = f"Scenario '{scenario_name}' (ΔT={deltas})"
        plot_zip_bar_top_bottom(
            zip_effects_df,
            column=col,
            output_dir=output_dir,
            top_n=20,
            title_prefix=title_prefix,
        )

    plot_zip_effect_distributions(zip_effects_df, scenarios, output_dir=output_dir)


# ============================================================
# 6. Main wrapper
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

    # 3) Fit a single LinearDML model for the treatments
    est, Y, T, X = fit_linear_dml(
        df_model=df_model,
        y_col=Y_COL,
        t_cols=TREATMENT_COLS,
        w_cols=CONTROL_COLS,
        x_cols=X_COLS,
    )

    # 4) Compute per-row CATEs for each scenario
    df_effects = compute_scenario_effects_per_row(
        est=est,
        df_model=df_model,
        t_cols=TREATMENT_COLS,
        X=X,
        scenarios=SCENARIOS,
    )

    # 5) Aggregate to ZIP-level
    zip_effects_df = aggregate_zip_effects(df_effects, SCENARIOS)

    # 6) Save ZIP-level effects to CSV
    out_csv = "zip_effects_scenarios.csv"
    zip_effects_df.to_csv(out_csv)
    print(f"Saved ZIP-level effects to {out_csv}")

    # 7) Create figures
    make_all_zip_figures(zip_effects_df, SCENARIOS, output_dir="figures")

    print("\nDone. Figures are in the 'figures/' directory.")


if __name__ == "__main__":
    main()
