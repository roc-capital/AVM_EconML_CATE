"""
ECONML CAUSAL ANALYSIS: RENOVATION EFFECTS BY GEOGRAPHIC AREA
Estimates heterogeneous treatment effects (HTE) for property renovations

Business Questions Answered:
1. What's the causal effect of adding a half bathroom in Cluster 44 vs Cluster 21?
2. What's the value increase from adding a bedroom in different neighborhoods?
3. Which renovation (bath, bedroom, garage) has highest impact in MY area?

Uses Double ML (DML) with causal forests to discover heterogeneous effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')

# EconML imports
try:
    from econml.dml import CausalForestDML, LinearDML
    from econml.dr import DRLearner

    ECONML_AVAILABLE = True
except ImportError:
    print("⚠️ EconML not installed. Install with: pip install econml")
    ECONML_AVAILABLE = False

# Add this import at the top with other imports
try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

# Add this configuration variable with other config
RUN_DOWHY_VALIDATION = True

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "/Users/jenny.lin/BASIS_AVM_Onboarding/cate_scenario_analyses/data/inference_df_cleaned_deduplicated.parquet"

IS_PANEL_DATA = True
PROPERTYID_COL = "PROPERTYID"
DECAY_FACTOR = 0.9
Y_COL = "SALEAMT"

# TREATMENTS: What renovations are we analyzing?
TREATMENTS = {
    'HALF_BATHS_COAL': {
        'name': 'Half Bathroom',
        'unit': 'bathroom',
        'typical_cost': 5000,  # Typical renovation cost (for reference only)
    },
    'BEDROOMS_MLS': {
        'name': 'Bedroom',
        'unit': 'bedroom',
        'typical_cost': 15000,
    },
    'GARAGE_SPACES_COAL': {
        'name': 'Garage Space',
        'unit': 'space',
        'typical_cost': 12000,
    },
    'FULL_BATHS_COAL': {
        'name': 'Full Bathroom',
        'unit': 'bathroom',
        'typical_cost': 10000,
    }
}

# CONFOUNDERS: What affects both treatment and outcome?
CONFOUNDERS = [
    'LIVINGAREASQFT_COAL',  # Size affects both # of rooms and price
    'LOTSIZESQFT_COAL',
    'YEARBUILT_COAL',
    'EFFECTIVEYEARBUILT_COAL',
    'FIREPLACE_COUNT_MLS',
    'YEAR',
]

# EFFECT MODIFIERS: What makes treatment effects vary?
EFFECT_MODIFIERS = [
    'GEO_CLUSTER',  # Key: effects vary by location!
    'LIVINGAREASQFT_COAL',  # Effects vary by house size
    'YEARBUILT_COAL',  # Effects vary by house age
    'PRICE_LEVEL',  # Effects vary by price tier
]

# Geographic clustering
N_GEO_CLUSTERS = 45
TOP_CLUSTERS_TO_ANALYZE = 10  # Analyze top N clusters by frequency
PRICE_WEIGHT = 0.3  # Weight for price in clustering (0=pure location, 1=pure price, 0.3=recommended)

# EconML settings
ECONML_METHOD = 'causal_forest'  # 'causal_forest', 'linear_dml', or 'dr_learner'
N_BOOTSTRAP = 100
CONFIDENCE_LEVEL = 0.95


# ============================================================================
# DATA PREPARATION
# ============================================================================

def collapse_to_property_level(df, decay=0.9):
    """Collapse panel data - same as before"""
    if PROPERTYID_COL not in df.columns:
        return df

    print(f"\n{'=' * 80}")
    print("COLLAPSING PANEL DATA")
    print(f"{'=' * 80}")

    last_cols = [c for c in [
        "YEAR", "BEDROOMS_MLS", "FULL_BATHS_COAL", "HALF_BATHS_COAL",
        "LIVINGAREASQFT_COAL", "LOTSIZESQFT_COAL",
        "YEARBUILT_COAL", "EFFECTIVEYEARBUILT_COAL",
        "GARAGE_SPACES_COAL", "FIREPLACE_COUNT_MLS",
        "LATITUDE", "LONGITUDE"
    ] if c in df.columns]

    df_sorted = df.sort_values([PROPERTYID_COL, "YEAR"])
    df_last = df_sorted.groupby(PROPERTYID_COL, as_index=False)[last_cols].last()

    def discounted_saleamt(group):
        max_year = group["YEAR"].max()
        weights = decay ** (max_year - group["YEAR"])
        return np.average(group["SALEAMT"], weights=weights)

    df_saleamt = (df_sorted.groupby(PROPERTYID_COL)
                  .apply(discounted_saleamt)
                  .rename("SALEAMT").reset_index())

    df_property = df_last.merge(df_saleamt, on=PROPERTYID_COL, how="left")
    print(f"✓ Collapsed to {len(df_property):,} properties")
    return df_property


def prepare_econml_data(df, treatment_col):
    """
    Prepare data for EconML causal estimation

    Returns: Y, T, X, W
    """
    print(f"\n{'=' * 80}")
    print(f"PREPARING DATA FOR CAUSAL ANALYSIS: {TREATMENTS[treatment_col]['name']}")
    print(f"{'=' * 80}")

    # Create price level categories
    df['PRICE_LEVEL'] = pd.qcut(df[Y_COL], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
    df['PRICE_LEVEL'] = df['PRICE_LEVEL'].cat.codes

    # Center year variables
    for col in ['YEAR', 'YEARBUILT_COAL', 'EFFECTIVEYEARBUILT_COAL']:
        if col in df.columns:
            df[f'{col}_CENTERED'] = df[col] - df[col].mean()

    # Prepare confounders (exclude treatment)
    confounder_cols = [c for c in CONFOUNDERS if c in df.columns and c != treatment_col]
    confounder_cols = [f'{c}_CENTERED' if f'{c}_CENTERED' in df.columns else c
                       for c in confounder_cols]

    # Prepare effect modifiers
    modifier_cols = [c for c in EFFECT_MODIFIERS if c in df.columns]
    modifier_cols = [f'{c}_CENTERED' if f'{c}_CENTERED' in df.columns else c
                     for c in modifier_cols]

    # Select data
    all_cols = [Y_COL, treatment_col] + confounder_cols + modifier_cols
    df_analysis = df[all_cols].dropna()

    # Remove outliers
    q01, q99 = df_analysis[Y_COL].quantile([0.01, 0.99])
    df_analysis = df_analysis[(df_analysis[Y_COL] >= q01) & (df_analysis[Y_COL] <= q99)]

    Y = df_analysis[Y_COL].values
    T = df_analysis[treatment_col].values
    W = df_analysis[confounder_cols].values if confounder_cols else None
    X = df_analysis[modifier_cols].values

    print(f"\n📊 Data Summary:")
    print(f"  Outcome (Y): {Y_COL}")
    print(f"  Treatment (T): {treatment_col}")
    print(f"  Confounders (W): {len(confounder_cols)} variables")
    print(f"  Effect Modifiers (X): {len(modifier_cols)} variables")
    print(f"  Sample size: {len(Y):,}")
    print(f"  Treatment range: {T.min():.0f} - {T.max():.0f}")
    print(f"  Treatment mean: {T.mean():.2f}")

    return {
        'Y': Y,
        'T': T.reshape(-1, 1),  # EconML expects 2D
        'W': W,
        'X': X,
        'df': df_analysis,
        'confounder_names': confounder_cols,
        'modifier_names': modifier_cols,
        'treatment_name': treatment_col
    }


# ============================================================================
# CAUSAL ESTIMATION WITH ECONML
# ============================================================================

def estimate_heterogeneous_effects(data, method='causal_forest'):
    """
    Estimate heterogeneous treatment effects using EconML
    """
    if not ECONML_AVAILABLE:
        print("❌ EconML not available")
        return None, None

    print(f"\n{'=' * 80}")
    print(f"ESTIMATING CAUSAL EFFECTS: {method.upper()}")
    print(f"{'=' * 80}")

    Y, T, X, W = data['Y'], data['T'], data['X'], data['W']

    # Choose estimator
    if method == 'causal_forest':
        print("Using Causal Forest DML (discovers heterogeneity automatically)")
        model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            n_estimators=100,
            min_samples_leaf=50,
            max_depth=8,
            random_state=42,
            verbose=0
        )
    elif method == 'linear_dml':
        print("Using Linear DML (assumes linear heterogeneity)")
        model = LinearDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            random_state=42
        )
    elif method == 'dr_learner':
        print("Using Doubly Robust Learner")
        model = DRLearner(
            model_regression=GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            model_propensity=GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            random_state=42
        )

    # Fit model
    print("\n🔄 Fitting causal model (this may take 2-3 minutes)...")
    if W is not None:
        model.fit(Y, T, X=X, W=W)
    else:
        model.fit(Y, T, X=X)

    print("✓ Model fitted successfully!")

    # Estimate effects
    print("\n📊 Estimating treatment effects...")
    effects = model.effect(X)  # Point estimates

    # Get confidence intervals
    print("📊 Computing confidence intervals...")
    effects_lower, effects_upper = model.effect_interval(X, alpha=1 - CONFIDENCE_LEVEL)

    print(f"✓ Effects estimated for {len(effects):,} properties")

    return model, {
        'effects': effects.flatten(),
        'effects_lower': effects_lower.flatten(),
        'effects_upper': effects_upper.flatten(),
        'X': X,
        'df': data['df']
    }


# ============================================================================
# ANALYZE EFFECTS BY CLUSTER
# ============================================================================

def analyze_effects_by_cluster(effects_dict, data, treatment_info):
    """
    Analyze how treatment effects vary by geographic cluster
    """
    print(f"\n{'=' * 80}")
    print(f"HETEROGENEOUS EFFECTS BY CLUSTER: {treatment_info['name']}")
    print(f"{'=' * 80}")

    df = data['df'].copy()
    df['treatment_effect'] = effects_dict['effects']
    df['effect_lower'] = effects_dict['effects_lower']
    df['effect_upper'] = effects_dict['effects_upper']

    # Get top clusters by frequency
    top_clusters = df['GEO_CLUSTER'].value_counts().head(TOP_CLUSTERS_TO_ANALYZE).index

    cluster_effects = []
    for cluster_id in top_clusters:
        cluster_data = df[df['GEO_CLUSTER'] == cluster_id]

        avg_effect = cluster_data['treatment_effect'].mean()
        median_effect = cluster_data['treatment_effect'].median()
        std_effect = cluster_data['treatment_effect'].std()
        ci_lower = cluster_data['effect_lower'].mean()
        ci_upper = cluster_data['effect_upper'].mean()
        n_properties = len(cluster_data)
        avg_price = cluster_data[Y_COL].mean()

        cluster_effects.append({
            'cluster': int(cluster_id),
            'n_properties': n_properties,
            'avg_price': avg_price,
            'avg_effect': avg_effect,
            'median_effect': median_effect,
            'std_effect': std_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        })

    cluster_df = pd.DataFrame(cluster_effects).sort_values('avg_effect', ascending=False)

    print(f"\nTop {TOP_CLUSTERS_TO_ANALYZE} Clusters by Sample Size:")
    print(f"{'Cluster':<10} {'N':>8} {'Avg Price':>12} {'Effect':>12} {'95% CI':>25}")
    print("-" * 77)

    for _, row in cluster_df.iterrows():
        ci_str = f"[${row['ci_lower']:>6,.0f}, ${row['ci_upper']:>6,.0f}]"
        print(f"Cluster {row['cluster']:<3} {row['n_properties']:>8,} "
              f"${row['avg_price']:>11,.0f} ${row['avg_effect']:>11,.0f} "
              f"{ci_str:>25}")

    # Interpretation
    print(f"\n💡 Interpretation:")
    best_cluster = cluster_df.iloc[0]
    worst_cluster = cluster_df.iloc[-1]

    print(f"\n🏆 Highest Effect:")
    print(
        f"  Cluster {int(best_cluster['cluster'])}: Adding one {treatment_info['unit']} increases value by ${best_cluster['avg_effect']:,.0f}")

    print(f"\n📉 Lowest Effect:")
    print(
        f"  Cluster {int(worst_cluster['cluster'])}: Adding one {treatment_info['unit']} increases value by ${worst_cluster['avg_effect']:,.0f}")

    effect_range = best_cluster['avg_effect'] - worst_cluster['avg_effect']
    print(f"\n📊 Effect Heterogeneity:")
    print(f"  Range of effects across clusters: ${effect_range:,.0f}")
    print(
        f"  This {treatment_info['unit']} is worth ${effect_range:,.0f} more in Cluster {int(best_cluster['cluster'])} vs Cluster {int(worst_cluster['cluster'])}")

    return cluster_df


def compare_renovation_options(all_results):
    """
    Compare treatment effects across different renovation types
    """
    print(f"\n{'=' * 80}")
    print("RENOVATION COMPARISON: WHICH HAS HIGHEST VALUE IMPACT?")
    print(f"{'=' * 80}")

    comparison = []
    for treatment_col, results in all_results.items():
        cluster_df = results['cluster_effects']
        treatment_info = TREATMENTS[treatment_col]

        # Overall averages
        avg_effect = cluster_df['avg_effect'].mean()
        best_effect = cluster_df['avg_effect'].max()
        worst_effect = cluster_df['avg_effect'].min()

        comparison.append({
            'renovation': treatment_info['name'],
            'avg_effect': avg_effect,
            'best_cluster_effect': best_effect,
            'worst_cluster_effect': worst_effect,
            'effect_range': best_effect - worst_effect
        })

    comp_df = pd.DataFrame(comparison).sort_values('avg_effect', ascending=False)

    print(f"\n{'Renovation':<20} {'Avg Effect':>12} {'Best Cluster':>15} {'Worst Cluster':>15} {'Range':>12}")
    print("-" * 80)

    for _, row in comp_df.iterrows():
        print(f"{row['renovation']:<20} ${row['avg_effect']:>11,.0f} "
              f"${row['best_cluster_effect']:>14,.0f} "
              f"${row['worst_cluster_effect']:>14,.0f} "
              f"${row['effect_range']:>11,.0f}")

    print(f"\n🎯 Key Finding:")
    best_effect = comp_df.iloc[0]
    print(f"  Highest average effect: {best_effect['renovation']}")
    print(f"  Average value increase: ${best_effect['avg_effect']:,.0f}")

    print(f"\n📊 Geographic Sensitivity:")
    most_heterogeneous = comp_df.sort_values('effect_range', ascending=False).iloc[0]
    print(f"  Most location-dependent: {most_heterogeneous['renovation']}")
    print(f"  Effect varies by ${most_heterogeneous['effect_range']:,.0f} across clusters")
    print(f"  → This renovation is highly sensitive to neighborhood!")

    return comp_df


def visualize_heterogeneous_effects(results, treatment_info):
    """
    Visualize treatment effect heterogeneity
    """
    cluster_df = results['cluster_effects']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Treatment effects by cluster
    ax1 = axes[0, 0]
    cluster_df_sorted = cluster_df.sort_values('avg_effect')
    colors = ['#3498db'] * len(cluster_df_sorted)

    bars = ax1.barh(range(len(cluster_df_sorted)), cluster_df_sorted['avg_effect'] / 1000, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(cluster_df_sorted)))
    ax1.set_yticklabels([f"C{int(c)}" for c in cluster_df_sorted['cluster']], fontsize=9)
    ax1.set_xlabel('Treatment Effect ($1000s)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Effect of Adding {treatment_info["name"]} by Cluster', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # 2. Effect with confidence intervals
    ax2 = axes[0, 1]
    cluster_df_sorted_ci = cluster_df.sort_values('avg_effect')
    y_pos = range(len(cluster_df_sorted_ci))

    ax2.barh(y_pos, cluster_df_sorted_ci['avg_effect'] / 1000, color='#3498db', alpha=0.7)
    ax2.errorbar(cluster_df_sorted_ci['avg_effect'] / 1000, y_pos,
                 xerr=[(cluster_df_sorted_ci['avg_effect'] - cluster_df_sorted_ci['ci_lower']) / 1000,
                       (cluster_df_sorted_ci['ci_upper'] - cluster_df_sorted_ci['avg_effect']) / 1000],
                 fmt='none', ecolor='black', elinewidth=1, capsize=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"C{int(c)}" for c in cluster_df_sorted_ci['cluster']], fontsize=9)
    ax2.set_xlabel('Treatment Effect ($1000s)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Effect with 95% Confidence Intervals', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # 3. Effect vs Price Level
    ax3 = axes[1, 0]
    ax3.scatter(cluster_df['avg_price'] / 1000, cluster_df['avg_effect'] / 1000, s=100, alpha=0.6, color='#3498db')
    ax3.set_xlabel('Average Property Price ($1000s)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Treatment Effect ($1000s)', fontsize=11, fontweight='bold')
    ax3.set_title('Effect vs Property Price Level', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(cluster_df['avg_price'], cluster_df['avg_effect'], 1)
    p = np.poly1d(z)
    ax3.plot(cluster_df['avg_price'] / 1000, p(cluster_df['avg_price']) / 1000, "r--", alpha=0.8, linewidth=2)

    # 4. Distribution of treatment effects
    ax4 = axes[1, 1]
    effects_data = results['effects_dict']['effects']
    ax4.hist(effects_data / 1000, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax4.axvline(x=effects_data.mean() / 1000, color='red', linestyle='--', linewidth=2,
                label=f'Mean: ${effects_data.mean() / 1000:.1f}K')
    ax4.set_xlabel('Treatment Effect ($1000s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title(f'Distribution of Treatment Effects', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    filename = f'causal_effects_{treatment_info["name"].lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {filename}")


def create_summary_report(all_results):
    """Create executive summary report"""
    with open('causal_analysis_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CAUSAL ANALYSIS SUMMARY: RENOVATION EFFECTS BY GEOGRAPHIC AREA\n")
        f.write("=" * 80 + "\n\n")

        for treatment_col, results in all_results.items():
            treatment_info = results['treatment_info']
            cluster_df = results['cluster_effects']

            f.write(f"\n{treatment_info['name'].upper()}\n")
            f.write("-" * 80 + "\n")

            best = cluster_df.iloc[0]
            worst = cluster_df.iloc[-1]

            f.write(f"\nHighest Effect:\n")
            f.write(f"  Cluster {int(best['cluster'])}: ${best['avg_effect']:,.0f} value increase\n")

            f.write(f"\nLowest Effect:\n")
            f.write(f"  Cluster {int(worst['cluster'])}: ${worst['avg_effect']:,.0f} value increase\n")

            f.write(f"\nHeterogeneity:\n")
            f.write(f"  Effect range: ${best['avg_effect'] - worst['avg_effect']:,.0f}\n")
            f.write(f"  Standard deviation: ${cluster_df['avg_effect'].std():,.0f}\n")

            f.write("\n")

    print("✓ Saved: causal_analysis_summary.txt")

def create_geo_clusters(df, n_clusters=N_GEO_CLUSTERS,
                        price_weight=0.3,
                        sqft_weight=0.2,
                        sqft_col='LIVINGAREASQFT_COAL'):
    """
    Create geographic clusters using location, price, and square footage

    Parameters:
    -----------
    df : DataFrame
        Data with LATITUDE, LONGITUDE, price, and sqft columns
    n_clusters : int
        Number of clusters to create
    price_weight : float (0 to 1)
        Weight for price in clustering
    sqft_weight : float (0 to 1)
        Weight for square footage in clustering
    sqft_col : str
        Column name for square footage (default: 'LIVINGAREASQFT_COAL')

    Note: Remaining weight goes to location (lat/lon)
    Example: price=0.3, sqft=0.2 → location gets 0.5 (50%)
    """
    # Calculate location weight
    location_weight = 1.0 - price_weight - sqft_weight

    if location_weight < 0:
        raise ValueError(
            f"Weights sum to more than 1.0! price_weight({price_weight}) + sqft_weight({sqft_weight}) = {price_weight + sqft_weight}")

    print(f"\n{'=' * 80}")
    print(f"CREATING {n_clusters} GEO-PRICE-SQFT CLUSTERS")
    print(f"{'=' * 80}")
    print(f"  Location weight: {location_weight:.1%}")
    print(f"  Price weight: {price_weight:.1%}")
    print(f"  Sqft weight: {sqft_weight:.1%}")

    # Prepare features
    required_cols = ['LATITUDE', 'LONGITUDE', Y_COL, sqft_col]
    X_data = df[required_cols].dropna()

    print(f"\n  Properties with complete data: {len(X_data):,} ({len(X_data) / len(df) * 100:.1f}%)")

    # Separate features
    X_geo = X_data[['LATITUDE', 'LONGITUDE']].values
    X_price = X_data[[Y_COL]].values
    X_sqft = X_data[[sqft_col]].values

    # Standardize separately to control weighting
    scaler_geo = StandardScaler()
    scaler_price = StandardScaler()
    scaler_sqft = StandardScaler()

    X_geo_scaled = scaler_geo.fit_transform(X_geo)
    X_price_scaled = scaler_price.fit_transform(X_price)
    X_sqft_scaled = scaler_sqft.fit_transform(X_sqft)

    # Apply weights
    X_geo_weighted = X_geo_scaled * location_weight
    X_price_weighted = X_price_scaled * price_weight
    X_sqft_weighted = X_sqft_scaled * sqft_weight

    # Concatenate all features
    X_combined = np.hstack([X_geo_weighted, X_price_weighted, X_sqft_weighted])

    # Fit K-means
    print(f"\n  Fitting K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[X_data.index, 'GEO_CLUSTER'] = kmeans.fit_predict(X_combined)
    df['GEO_CLUSTER'] = df['GEO_CLUSTER'].fillna(df['GEO_CLUSTER'].mode()[0]).astype(int)

    # Analyze clusters
    cluster_stats = df.groupby('GEO_CLUSTER').agg({
        Y_COL: ['mean', 'std', 'min', 'max', 'count'],
        sqft_col: ['mean', 'std', 'min', 'max'],
        'LATITUDE': ['mean', 'std'],
        'LONGITUDE': ['mean', 'std']
    }).round(2)

    print(f"\n✓ Created {n_clusters} clusters")
    print(f"\nCluster Statistics:")
    print(f"  Price range: ${cluster_stats[(Y_COL, 'mean')].min():,.0f} - ${cluster_stats[(Y_COL, 'mean')].max():,.0f}")
    print(
        f"  Sqft range: {cluster_stats[(sqft_col, 'mean')].min():,.0f} - {cluster_stats[(sqft_col, 'mean')].max():,.0f} sqft")
    print(f"  Avg within-cluster price std: ${cluster_stats[(Y_COL, 'std')].mean():,.0f}")
    print(f"  Avg within-cluster sqft std: {cluster_stats[(sqft_col, 'std')].mean():,.0f} sqft")
    print(f"  Avg properties per cluster: {cluster_stats[(Y_COL, 'count')].mean():.0f}")

    # Show example clusters
    print(f"\nExample Clusters (Top 5 by size):")
    print(
        f"{'Cluster':<10} {'N':>8} {'Avg Price':>12} {'Avg Sqft':>10} {'Price Std':>11} {'Sqft Std':>10} {'Lat':>8} {'Lon':>8}")
    print("-" * 95)

    top_clusters = cluster_stats.nlargest(5, (Y_COL, 'count'))
    for idx in top_clusters.index:
        row = cluster_stats.loc[idx]
        print(f"Cluster {idx:<3} {int(row[(Y_COL, 'count')]):>8,} "
              f"${row[(Y_COL, 'mean')]:>11,.0f} {row[(sqft_col, 'mean')]:>9,.0f} "
              f"${row[(Y_COL, 'std')]:>10,.0f} {row[(sqft_col, 'std')]:>9,.0f} "
              f"{row[('LATITUDE', 'mean')]:>7.3f} {row[('LONGITUDE', 'mean')]:>7.3f}")

    return df


def validate_with_dowhy(data, treatment_col, treatment_info):
    """
    Validate causal estimates using DoWhy framework

    Parameters:
    -----------
    data : dict
        Output from prepare_econml_data()
    treatment_col : str
        Name of treatment variable
    treatment_info : dict
        Treatment metadata from TREATMENTS dict

    Returns:
    --------
    dict with DoWhy results, or None if validation fails/skipped
    """
    if not DOWHY_AVAILABLE:
        print("⚠️ DoWhy not available - skipping validation")
        return None

    print(f"\n{'=' * 80}")
    print(f"DOWHY VALIDATION: {treatment_info['name']}")
    print(f"{'=' * 80}")

    try:
        # Prepare data
        df_dowhy = data['df'].copy()
        df_dowhy[treatment_col] = data['T'].flatten()
        df_dowhy[Y_COL] = data['Y']

        # Build causal graph
        confounder_names = data['confounder_names']

        print(f"\n📐 Building Causal DAG...")
        print(f"   Treatment: {treatment_col}")
        print(f"   Outcome: {Y_COL}")
        print(f"   Confounders: {len(confounder_names)}")

        # Create graph edges
        graph_edges = []
        for conf in confounder_names:
            graph_edges.append(f'        {conf} -> {treatment_col};')
            graph_edges.append(f'        {conf} -> {Y_COL};')
        graph_edges.append(f'        {treatment_col} -> {Y_COL};')

        graph = "digraph {\n" + "\n".join(graph_edges) + "\n    }"

        # Create causal model
        model = CausalModel(
            data=df_dowhy,
            treatment=treatment_col,
            outcome=Y_COL,
            graph=graph
        )

        # Identify effect
        print("\n[STEP 1] Identifying causal effect...")
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Estimate with linear regression (fast baseline)
        print("\n[STEP 2] Estimating with linear regression...")
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=True
        )

        print(f"✓ Estimate: ${estimate.value:,.0f}")

        # Refutation tests
        print("\n[STEP 3] Running refutation tests...")
        refutations = {}

        # Placebo test
        print("  → Placebo treatment test...")
        try:
            refute_placebo = model.refute_estimate(
                identified_estimand, estimate,
                method_name="placebo_treatment_refuter",
                num_simulations=20  # Reduced for speed
            )
            refutations['placebo'] = {
                'new_effect': refute_placebo.new_effect,
                'p_value': refute_placebo.refutation_result.get('p_value', None)
            }
            print(f"     Placebo effect: ${refute_placebo.new_effect:,.0f}")
        except Exception as e:
            print(f"     Failed: {e}")
            refutations['placebo'] = None

        # Random common cause
        print("  → Random common cause test...")
        try:
            refute_random = model.refute_estimate(
                identified_estimand, estimate,
                method_name="random_common_cause",
                num_simulations=20
            )
            refutations['random_common_cause'] = {
                'new_effect': refute_random.new_effect,
                'p_value': refute_random.refutation_result.get('p_value', None)
            }
            pct_change = abs((refute_random.new_effect - estimate.value) / estimate.value) * 100
            print(f"     Changed by: {pct_change:.1f}%")
        except Exception as e:
            print(f"     Failed: {e}")
            refutations['random_common_cause'] = None

        # Data subset
        print("  → Data subset test...")
        try:
            refute_subset = model.refute_estimate(
                identified_estimand, estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.9,
                num_simulations=20
            )
            refutations['data_subset'] = {
                'new_effect': refute_subset.new_effect,
                'p_value': refute_subset.refutation_result.get('p_value', None)
            }
            pct_change = abs((refute_subset.new_effect - estimate.value) / estimate.value) * 100
            print(f"     Changed by: {pct_change:.1f}%")
        except Exception as e:
            print(f"     Failed: {e}")
            refutations['data_subset'] = None

        print("\n✓ DoWhy validation complete")

        return {
            'model': model,
            'identified_estimand': identified_estimand,
            'estimate': estimate.value,
            'refutations': refutations
        }

    except Exception as e:
        print(f"\n❌ DoWhy validation failed: {e}")
        return None

def save_dowhy_summary(all_results):
    """Save DoWhy validation results to file"""
    if not any(r.get('dowhy_validation') for r in all_results.values()):
        print("  (No DoWhy validations to save)")
        return

    with open('dowhy_validation_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DOWHY VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for treatment_col, results in all_results.items():
            dowhy = results.get('dowhy_validation')
            if not dowhy:
                continue

            treatment_info = results['treatment_info']
            f.write(f"\n{treatment_info['name'].upper()}\n")
            f.write("-" * 80 + "\n")

            f.write(f"\nLinear Regression Estimate: ${dowhy['estimate']:,.0f}\n")

            f.write(f"\nRefutation Tests:\n")
            for test_name, test_result in dowhy['refutations'].items():
                if test_result:
                    f.write(f"  {test_name}: ${test_result['new_effect']:,.0f}")
                    if test_result.get('p_value'):
                        f.write(f" (p={test_result['p_value']:.4f})")
                    f.write("\n")

            f.write("\n")

    print("✓ Saved: dowhy_validation_summary.txt")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution: Analyze causal effects for multiple renovation types
    """
    print("\n" + "=" * 80)
    print(" " * 10 + "ECONML CAUSAL ANALYSIS: RENOVATION EFFECTS BY AREA")
    print("=" * 80)

    # Load and prepare data
    print("\n[1/6] Loading and preparing data...")
    df = pd.read_parquet(DATA_PATH)

    if IS_PANEL_DATA:
        df = collapse_to_property_level(df, DECAY_FACTOR)

    df = create_geo_clusters(df, n_clusters=N_GEO_CLUSTERS, price_weight=PRICE_WEIGHT)

    # Analyze each treatment
    all_results = {}

    print(f"\n[2/6] Analyzing {len(TREATMENTS)} renovation types...")

    for treatment_col, treatment_info in TREATMENTS.items():
        if treatment_col not in df.columns:
            print(f"\n⚠️ Skipping {treatment_info['name']} - column not in data")
            continue

        print(f"\n{'=' * 80}")
        print(f"ANALYZING: {treatment_info['name']}")
        print(f"{'=' * 80}")

        # Prepare data
        data = prepare_econml_data(df, treatment_col)

        # Estimate effects
        model, effects_dict = estimate_heterogeneous_effects(data, ECONML_METHOD)

        if model is None:
            continue

        # Analyze by cluster
        cluster_df = analyze_effects_by_cluster(effects_dict, data, treatment_info)

        # **NEW: DoWhy validation**
        dowhy_results = None
        if RUN_DOWHY_VALIDATION:
            dowhy_results = validate_with_dowhy(data, treatment_col, treatment_info)

        # Visualize
        visualize_heterogeneous_effects(
            {'cluster_effects': cluster_df, 'effects_dict': effects_dict},
            treatment_info
        )

        all_results[treatment_col] = {
            'model': model,
            'effects_dict': effects_dict,
            'cluster_effects': cluster_df,
            'treatment_info': treatment_info,
            'dowhy_validation': dowhy_results  # **NEW: Store DoWhy results**
        }

    # Compare renovations
    if len(all_results) > 1:
        print(f"\n[3/6] Comparing renovation options...")
        comparison_df = compare_renovation_options(all_results)

        # Save comparison
        comparison_df.to_csv('renovation_comparison.csv', index=False)
        print("\n✓ Saved: renovation_comparison.csv")

    # Save all cluster effects
    print(f"\n[4/6] Saving detailed results...")
    for treatment_col, results in all_results.items():
        filename = f"cluster_effects_{treatment_col.lower()}.csv"
        results['cluster_effects'].to_csv(filename, index=False)
        print(f"✓ Saved: {filename}")

    # **NEW: Save DoWhy validation summary**
    print(f"\n[5/6] Saving validation summary...")
    save_dowhy_summary(all_results)

    print(f"\n[6/6] Creating summary report...")
    create_summary_report(all_results)

    print(f"\n{'=' * 80}")
    print("✅ CAUSAL ANALYSIS COMPLETE!")
    print(f"{'=' * 80}")

    return all_results

if __name__ == "__main__":
    if not ECONML_AVAILABLE:
        print("\n❌ ERROR: EconML not installed")
        print("Install with: pip install econml")
        print("Then run this script again")
    else:
        results = main()