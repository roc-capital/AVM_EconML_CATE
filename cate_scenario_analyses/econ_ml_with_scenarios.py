"""
ECONML MULTI-TREATMENT CAUSAL ANALYSIS: RENOVATION EFFECTS BY GEOGRAPHIC AREA
Estimates heterogeneous treatment effects (HTE) for property renovations

Business Questions Answered:
1. What's the causal effect of adding a half bathroom in Cluster 44 vs Cluster 21?
2. What's the value increase from adding a bedroom in different neighborhoods?
3. Which renovation (bath, bedroom, garage) has highest impact in MY area?
4. What's the combined effect of multiple renovations together?
5. Do renovations have synergies (worth more together than separate)?

Uses Double ML (DML) with causal forests to discover heterogeneous effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from econml.dml import CausalForestDML, LinearDML

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

# DoWhy import
try:
    from dowhy import CausalModel

    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

# Configuration
RUN_DOWHY_VALIDATION = True

# ============================================================================
# CONFIGURATION
# ============================================================================

IS_PANEL_DATA = True
PROPERTYID_COL = "propertyid"
DECAY_FACTOR = 0.9
Y_COL = "saleamt"

# TREATMENTS: What renovations are we analyzing?
TREATMENTS = {
    'half_baths_coal': {
        'name': 'Half Bathroom',
        'unit': 'bathroom',
        'typical_cost': 5000,
    },
    'bedrooms_mls': {
        'name': 'Bedroom',
        'unit': 'bedroom',
        'typical_cost': 15000,
    },
    'garage_spaces_coal': {
        'name': 'Garage Space',
        'unit': 'space',
        'typical_cost': 12000,
    },
    'full_baths_coal': {
        'name': 'Full Bathroom',
        'unit': 'bathroom',
        'typical_cost': 10000,
    }
}

# CONFOUNDERS: What affects both treatment and outcome?
CONFOUNDERS = [
    'livingareasqft_coal',
    'lotsizesqft_coal',
    'yearbuilt_coal',
    'effectiveyearbuilt_coal',
    'fireplace_count_mls',
    'year',
    'total_population_25plus',
    'male_high_school_graduate',
    'female_high_school_graduate',
    'male_bachelors_degree',
    'female_bachelors_degree'
]

# EFFECT MODIFIERS: What makes treatment effects vary?
EFFECT_MODIFIERS = [
    'geo_cluster',
    'livingareasqft_coal',
    'yearbuilt_coal',
    'price_level',
]

# Geographic clustering
N_GEO_CLUSTERS = 12
TOP_CLUSTERS_TO_ANALYZE = 12
PRICE_WEIGHT = 0.3

# EconML settings
ECONML_METHOD = 'causal_forest'
N_BOOTSTRAP = 100
CONFIDENCE_LEVEL = 0.95

# Multi-treatment settings
RUN_MULTI_TREATMENT_ANALYSIS = True
DETAILED_CLUSTERS_TO_ANALYZE = 3


# ============================================================================
# DATA PREPARATION
# ============================================================================

def collapse_to_property_level(df, decay=0.9):
    """Collapse panel data"""
    if PROPERTYID_COL not in df.columns:
        return df

    print(f"\n{'=' * 80}")
    print("COLLAPSING PANEL DATA")
    print(f"{'=' * 80}")

    last_cols = [c for c in [
        "year", "bedrooms_mls", "full_baths_coal", "half_baths_coal",
        "livingareasqft_coal", "lotsizesqft_coal",
        "yearbuilt_coal", "effectiveyearbuilt_coal",
        "garage_spaces_coal", "fireplace_count_mls",
        "latitude", "longitude",
        "total_population_25plus", "male_high_school_graduate",
        "female_high_school_graduate", "male_bachelors_degree",
        "female_bachelors_degree"
    ] if c in df.columns]

    df_sorted = df.sort_values([PROPERTYID_COL, "year"])
    df_last = df_sorted.groupby(PROPERTYID_COL, as_index=False)[last_cols].last()

    def discounted_saleamt(group):
        max_year = group["year"].max()
        weights = decay ** (max_year - group["year"])
        return np.average(group["saleamt"], weights=weights)

    df_saleamt = (df_sorted.groupby(PROPERTYID_COL)
                  .apply(discounted_saleamt)
                  .rename("saleamt").reset_index())

    df_property = df_last.merge(df_saleamt, on=PROPERTYID_COL, how="left")
    print(f"✓ Collapsed to {len(df_property):,} properties")
    return df_property


def prepare_econml_data(df, treatment_col):
    """Prepare data for EconML causal estimation"""
    print(f"\n{'=' * 80}")
    print(f"PREPARING DATA FOR CAUSAL ANALYSIS: {TREATMENTS[treatment_col]['name']}")
    print(f"{'=' * 80}")

    # Create price level categories
    df['price_level'] = pd.qcut(df[Y_COL], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
    df['price_level'] = df['price_level'].cat.codes

    # Center year variables
    for col in ['year', 'yearbuilt_coal', 'effectiveyearbuilt_coal']:
        if col in df.columns:
            df[f'{col}_centered'] = df[col] - df[col].mean()

    # Prepare confounders (exclude treatment)
    confounder_cols = [c for c in CONFOUNDERS if c in df.columns and c != treatment_col]
    confounder_cols = [f'{c}_centered' if f'{c}_centered' in df.columns else c
                       for c in confounder_cols]

    # Prepare effect modifiers
    modifier_cols = [c for c in EFFECT_MODIFIERS if c in df.columns]
    modifier_cols = [f'{c}_centered' if f'{c}_centered' in df.columns else c
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
        'T': T.reshape(-1, 1),
        'W': W,
        'X': X,
        'df': df_analysis,
        'confounder_names': confounder_cols,
        'modifier_names': modifier_cols,
        'treatment_name': treatment_col
    }


def create_geo_clusters_with_politics_income(df, n_clusters=12,
                                             price_weight=0.18,
                                             sqft_weight=0.12,
                                             education_weight=0.12,
                                             politics_weight=0.14,
                                             income_weight=0.12,
                                             sqft_col='livingareasqft_coal',
                                             income_col='median_earnings_total',
                                             Y_COL='saleamt'):
    """Create geographic clusters including political affiliation and median income"""

    location_weight = 1.0 - price_weight - sqft_weight - education_weight - politics_weight - income_weight

    if location_weight < 0:
        raise ValueError(
            f"Weights sum to more than 1.0! Current sum: {price_weight + sqft_weight + education_weight + politics_weight + income_weight:.2f}")

    print(f"\n{'=' * 80}")
    print(f"CREATING {n_clusters} GEO-PRICE-SQFT-EDUCATION-POLITICS-INCOME CLUSTERS")
    print(f"{'=' * 80}")
    print(f"  Location weight:  {location_weight:.1%}")
    print(f"  Price weight:     {price_weight:.1%}")
    print(f"  Sqft weight:      {sqft_weight:.1%}")
    print(f"  Education weight: {education_weight:.1%}")
    print(f"  Politics weight:  {politics_weight:.1%}")
    print(f"  Income weight:    {income_weight:.1%}")
    print(
        f"  Total:            {location_weight + price_weight + sqft_weight + education_weight + politics_weight + income_weight:.1%}")

    # Calculate education percentage
    df['pct_bachelors_plus'] = (
                                       (df['male_bachelors_degree'] + df['female_bachelors_degree']) /
                                       df['total_population_25plus']
                               ).fillna(0) * 100

    # Define required columns
    required_cols = ['latitude', 'longitude', Y_COL, sqft_col,
                     'pct_bachelors_plus', 'dem_lift', income_col]

    X_data = df[required_cols].dropna()

    print(f"\n  Properties with complete data: {len(X_data):,} ({len(X_data) / len(df) * 100:.1f}%)")

    # Prepare features
    X_geo = X_data[['latitude', 'longitude']].values
    X_price = X_data[[Y_COL]].values
    X_sqft = X_data[[sqft_col]].values
    X_education = X_data[['pct_bachelors_plus']].values
    X_politics = X_data[['dem_lift']].values
    X_income = X_data[[income_col]].values

    # Scale all features
    scaler_geo = StandardScaler()
    scaler_price = StandardScaler()
    scaler_sqft = StandardScaler()
    scaler_education = StandardScaler()
    scaler_politics = StandardScaler()
    scaler_income = StandardScaler()

    X_geo_scaled = scaler_geo.fit_transform(X_geo)
    X_price_scaled = scaler_price.fit_transform(X_price)
    X_sqft_scaled = scaler_sqft.fit_transform(X_sqft)
    X_education_scaled = scaler_education.fit_transform(X_education)
    X_politics_scaled = scaler_politics.fit_transform(X_politics)
    X_income_scaled = scaler_income.fit_transform(X_income)

    # Apply weights
    X_geo_weighted = X_geo_scaled * location_weight
    X_price_weighted = X_price_scaled * price_weight
    X_sqft_weighted = X_sqft_scaled * sqft_weight
    X_education_weighted = X_education_scaled * education_weight
    X_politics_weighted = X_politics_scaled * politics_weight
    X_income_weighted = X_income_scaled * income_weight

    # Combine features
    X_combined = np.hstack([
        X_geo_weighted, X_price_weighted, X_sqft_weighted,
        X_education_weighted, X_politics_weighted, X_income_weighted
    ])

    print(f"\n  Fitting K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[X_data.index, 'geo_cluster'] = kmeans.fit_predict(X_combined)
    df['geo_cluster'] = df['geo_cluster'].fillna(df['geo_cluster'].mode()[0]).astype(int)

    print(f"\n✓ Created {n_clusters} clusters")

    # Print cluster statistics
    print(f"\n  Cluster demographics summary:")
    print(f"  {'Cluster':<8} {'N':>8} {'Price':>12} {'Bach%':>6} {'Dem+':>6} {'Income':>10}")
    print("  " + "-" * 60)

    for cluster_id in sorted(df['geo_cluster'].unique()):
        cluster_data = df[df['geo_cluster'] == cluster_id]
        n = len(cluster_data)
        avg_price = cluster_data[Y_COL].mean()
        avg_edu = cluster_data['pct_bachelors_plus'].mean()
        avg_dem = cluster_data['dem_lift'].mean()
        avg_income = cluster_data[income_col].mean()

        print(f"  {cluster_id:<8} {n:>8,} ${avg_price:>10,.0f} "
              f"{avg_edu:>5.1f}% {avg_dem:>5.1f}% ${avg_income:>9,.0f}")

    # Print income range by cluster
    print(f"\n  Income range by cluster:")
    print(f"  {'Cluster':<8} {'Min Income':>12} {'Median':>12} {'Max Income':>12}")
    print("  " + "-" * 50)

    for cluster_id in sorted(df['geo_cluster'].unique()):
        cluster_data = df[df['geo_cluster'] == cluster_id]
        min_income = cluster_data[income_col].min()
        med_income = cluster_data[income_col].median()
        max_income = cluster_data[income_col].max()

        print(f"  {cluster_id:<8} ${min_income:>11,.0f} ${med_income:>11,.0f} ${max_income:>11,.0f}")

    return df


# ============================================================================
# INDIVIDUAL TREATMENT CAUSAL ESTIMATION
# ============================================================================

def estimate_heterogeneous_effects(data, method='causal_forest'):
    """Estimate heterogeneous treatment effects using EconML"""
    if not ECONML_AVAILABLE:
        print("❌ EconML not available")
        return None, None

    print(f"\n{'=' * 80}")
    print(f"ESTIMATING CAUSAL EFFECTS: {method.upper()}")
    print(f"{'=' * 80}")

    Y, T, X, W = data['Y'], data['T'], data['X'], data['W']

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

    print("\n🔄 Fitting causal model (this may take 2-3 minutes)...")
    if W is not None:
        model.fit(Y, T, X=X, W=W)
    else:
        model.fit(Y, T, X=X)

    print("✓ Model fitted successfully!")

    print("\n📊 Estimating treatment effects...")
    effects = model.effect(X)

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


def analyze_effects_by_cluster(effects_dict, data, treatment_info):
    """Analyze how treatment effects vary by geographic cluster"""
    print(f"\n{'=' * 80}")
    print(f"HETEROGENEOUS EFFECTS BY CLUSTER: {treatment_info['name']}")
    print(f"{'=' * 80}")

    df = data['df'].copy()
    df['treatment_effect'] = effects_dict['effects']
    df['effect_lower'] = effects_dict['effects_lower']
    df['effect_upper'] = effects_dict['effects_upper']

    top_clusters = df['geo_cluster'].value_counts().head(TOP_CLUSTERS_TO_ANALYZE).index

    cluster_effects = []
    for cluster_id in top_clusters:
        cluster_data = df[df['geo_cluster'] == cluster_id]

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

    return cluster_df, cluster_data


def compare_renovation_options(all_results):
    """Compare treatment effects across different renovation types"""
    print(f"\n{'=' * 80}")
    print("RENOVATION COMPARISON: WHICH HAS HIGHEST VALUE IMPACT?")
    print(f"{'=' * 80}")

    comparison = []
    for treatment_col, results in all_results.items():
        cluster_df = results['cluster_effects']
        treatment_info = TREATMENTS[treatment_col]

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

    return comp_df


def visualize_heterogeneous_effects(results, treatment_info):
    """Visualize treatment effect heterogeneity"""
    cluster_df = results['cluster_effects']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Treatment effects by cluster
    ax1 = axes[0, 0]
    cluster_df_sorted = cluster_df.sort_values('avg_effect')
    colors = ['#3498db'] * len(cluster_df_sorted)

    ax1.barh(range(len(cluster_df_sorted)), cluster_df_sorted['avg_effect'] / 1000, color=colors, alpha=0.7)
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


# ============================================================================
# MULTI-TREATMENT SCENARIO ANALYSIS
# ============================================================================

def estimate_multi_treatment_effects(df, treatments_list):
    """
    Estimate effects of multiple treatments simultaneously
    Captures interaction effects between treatments

    **FIX: Uses MultiOutputRegressor to handle multi-dimensional treatments**
    """
    if not ECONML_AVAILABLE:
        print("❌ EconML not available")
        return None, None, None

    print(f"\n{'=' * 80}")
    print(f"MULTI-TREATMENT ANALYSIS: {len(treatments_list)} TREATMENTS")
    print(f"{'=' * 80}")
    print(f"Treatments: {', '.join([TREATMENTS[t]['name'] for t in treatments_list if t in TREATMENTS])}")

    # Create price level
    df['price_level'] = pd.qcut(df[Y_COL], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
    df['price_level'] = df['price_level'].cat.codes

    # Center year variables
    for col in ['year', 'yearbuilt_coal', 'effectiveyearbuilt_coal']:
        if col in df.columns:
            df[f'{col}_centered'] = df[col] - df[col].mean()

    # Prepare confounders (exclude ALL treatments)
    confounder_cols = [c for c in CONFOUNDERS
                       if c in df.columns and c not in treatments_list]
    confounder_cols = [f'{c}_centered' if f'{c}_centered' in df.columns else c
                       for c in confounder_cols]

    # Prepare effect modifiers
    modifier_cols = [c for c in EFFECT_MODIFIERS if c in df.columns]
    modifier_cols = [f'{c}_centered' if f'{c}_centered' in df.columns else c
                     for c in modifier_cols]

    # Select complete cases
    all_cols = [Y_COL] + treatments_list + confounder_cols + modifier_cols
    df_analysis = df[all_cols].dropna()

    # Remove outliers
    q01, q99 = df_analysis[Y_COL].quantile([0.01, 0.99])
    df_analysis = df_analysis[
        (df_analysis[Y_COL] >= q01) & (df_analysis[Y_COL] <= q99)
        ]

    Y = df_analysis[Y_COL].values
    T = df_analysis[treatments_list].values  # Multi-dimensional treatment
    W = df_analysis[confounder_cols].values if confounder_cols else None
    X = df_analysis[modifier_cols].values

    print(f"\n📊 Data Summary:")
    print(f"  Sample size: {len(Y):,}")
    print(f"  Confounders (W): {len(confounder_cols)} variables")
    print(f"  Effect Modifiers (X): {len(modifier_cols)} variables")
    for i, t in enumerate(treatments_list):
        print(f"  Treatment {i + 1} ({t}): mean={T[:, i].mean():.2f}, range=[{T[:, i].min():.0f}, {T[:, i].max():.0f}]")

    # **FIX: Use LinearDML with MultiOutputRegressor for multi-dimensional treatments**
    print("\n🔄 Fitting multi-treatment linear DML model (this may take 3-5 minutes)...")
    print("   Using LinearDML (better for multi-dimensional treatments)")

    # Wrap base learners in MultiOutputRegressor to handle multiple treatments
    model = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
        model_t=MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)),
        discrete_treatment=False,
        random_state=42
    )

    if W is not None:
        model.fit(Y, T, X=X, W=W)
    else:
        model.fit(Y, T, X=X)

    print("✓ Multi-treatment model fitted successfully!")

    data_dict = {
        'df': df_analysis,
        'modifier_cols': modifier_cols,
        'treatments_list': treatments_list,
        'X': X,
        'T': T,
        'Y': Y
    }

    return model, df_analysis, data_dict


def predict_scenario(model, data_dict, cluster_id, treatment_values, property_size='median'):
    """
    Predict effect for a specific renovation scenario
    """
    df_analysis = data_dict['df']
    modifier_cols = data_dict['modifier_cols']
    treatments_list = data_dict['treatments_list']

    # Get properties in this cluster
    cluster_data = df_analysis[df_analysis['geo_cluster'] == cluster_id]

    if len(cluster_data) == 0:
        print(f"❌ No data for cluster {cluster_id}")
        return None

    # Get typical property characteristics
    if property_size == 'median':
        X_baseline = cluster_data[modifier_cols].median().values.reshape(1, -1)
    elif property_size == 'mean':
        X_baseline = cluster_data[modifier_cols].mean().values.reshape(1, -1)
    elif property_size == 'small':
        X_baseline = cluster_data[modifier_cols].quantile(0.25).values.reshape(1, -1)
    elif property_size == 'large':
        X_baseline = cluster_data[modifier_cols].quantile(0.75).values.reshape(1, -1)
    else:
        X_baseline = cluster_data[modifier_cols].median().values.reshape(1, -1)

    # Baseline: no changes (zeros)
    T_baseline = np.zeros((1, len(treatments_list)))

    # Scenario: apply treatments
    T_scenario = T_baseline.copy()
    for i, treatment in enumerate(treatments_list):
        if treatment in treatment_values:
            T_scenario[0, i] = treatment_values[treatment]

    # Predict marginal effect
    # For LinearDML with multi-dimensional treatment, effect() returns total effect
    effect_result = model.effect(X_baseline, T0=T_baseline, T1=T_scenario)

    # Handle different output shapes
    if effect_result.ndim > 1:
        marginal_effect = effect_result[0, 0] if effect_result.shape[1] > 0 else effect_result[0]
    else:
        marginal_effect = effect_result[0]

    return float(marginal_effect)


def analyze_scenarios_by_cluster(model, data_dict, scenarios, top_n_clusters=10):
    """
    Analyze multiple scenarios across top clusters
    """
    print(f"\n{'=' * 80}")
    print(f"SCENARIO ANALYSIS ACROSS CLUSTERS")
    print(f"{'=' * 80}")

    df_analysis = data_dict['df']
    treatments_list = data_dict['treatments_list']

    # Get top clusters by frequency
    top_clusters = df_analysis['geo_cluster'].value_counts().head(top_n_clusters).index

    results = []

    for cluster_id in top_clusters:
        cluster_data = df_analysis[df_analysis['geo_cluster'] == cluster_id]
        n_properties = len(cluster_data)
        avg_price = cluster_data[Y_COL].mean()

        for scenario_name, treatment_values in scenarios.items():
            effect = predict_scenario(model, data_dict, cluster_id, treatment_values)

            if effect is not None:
                results.append({
                    'cluster': int(cluster_id),
                    'scenario': scenario_name,
                    'effect': effect,
                    'n_properties': n_properties,
                    'avg_price': avg_price,
                    **treatment_values  # Add individual treatment quantities
                })

    results_df = pd.DataFrame(results)

    # Print summary
    print(f"\nAnalyzed {len(scenarios)} scenarios across {len(top_clusters)} clusters")
    print(f"Total predictions: {len(results_df)}")

    # Show best scenario per cluster
    print(f"\n{'=' * 80}")
    print("BEST SCENARIO PER CLUSTER")
    print(f"{'=' * 80}\n")
    print(f"{'Cluster':<10} {'N':>8} {'Best Scenario':<40} {'Effect':>12}")
    print("-" * 80)

    best_per_cluster = results_df.loc[results_df.groupby('cluster')['effect'].idxmax()]
    best_per_cluster = best_per_cluster.sort_values('effect', ascending=False)

    for _, row in best_per_cluster.iterrows():
        print(f"Cluster {row['cluster']:<3} {int(row['n_properties']):>8,} "
              f"{row['scenario']:<40} ${row['effect']:>11,.0f}")

    return results_df


def create_geo_clusters_with_politics_income(df, n_clusters=12,
                                             price_weight=0.18,
                                             sqft_weight=0.12,
                                             education_weight=0.12,
                                             politics_weight=0.14,
                                             income_weight=0.12,
                                             sqft_col='livingareasqft_coal',
                                             income_col='median_earnings_total',
                                             Y_COL='saleamt'):
    """Create geographic clusters including political affiliation and median income"""

    location_weight = 1.0 - price_weight - sqft_weight - education_weight - politics_weight - income_weight

    if location_weight < 0:
        raise ValueError(
            f"Weights sum to more than 1.0! Current sum: {price_weight + sqft_weight + education_weight + politics_weight + income_weight:.2f}")

    print(f"\n{'=' * 80}")
    print(f"CREATING {n_clusters} GEO-PRICE-SQFT-EDUCATION-POLITICS-INCOME CLUSTERS")
    print(f"{'=' * 80}")
    print(f"  Location weight:  {location_weight:.1%}")
    print(f"  Price weight:     {price_weight:.1%}")
    print(f"  Sqft weight:      {sqft_weight:.1%}")
    print(f"  Education weight: {education_weight:.1%}")
    print(f"  Politics weight:  {politics_weight:.1%}")
    print(f"  Income weight:    {income_weight:.1%}")
    print(
        f"  Total:            {location_weight + price_weight + sqft_weight + education_weight + politics_weight + income_weight:.1%}")

    # Calculate education percentage
    df['pct_bachelors_plus'] = (
                                       (df['male_bachelors_degree'] + df['female_bachelors_degree']) /
                                       df['total_population_25plus']
                               ).fillna(0) * 100

    # Define required columns
    required_cols = ['latitude', 'longitude', Y_COL, sqft_col,
                     'pct_bachelors_plus', 'dem_lift', income_col]

    X_data = df[required_cols].dropna()

    print(f"\n  Properties with complete data: {len(X_data):,} ({len(X_data) / len(df) * 100:.1f}%)")

    # Prepare features
    X_geo = X_data[['latitude', 'longitude']].values
    X_price = X_data[[Y_COL]].values
    X_sqft = X_data[[sqft_col]].values
    X_education = X_data[['pct_bachelors_plus']].values
    X_politics = X_data[['dem_lift']].values
    X_income = X_data[[income_col]].values

    # Scale all features
    scaler_geo = StandardScaler()
    scaler_price = StandardScaler()
    scaler_sqft = StandardScaler()
    scaler_education = StandardScaler()
    scaler_politics = StandardScaler()
    scaler_income = StandardScaler()

    X_geo_scaled = scaler_geo.fit_transform(X_geo)
    X_price_scaled = scaler_price.fit_transform(X_price)
    X_sqft_scaled = scaler_sqft.fit_transform(X_sqft)
    X_education_scaled = scaler_education.fit_transform(X_education)
    X_politics_scaled = scaler_politics.fit_transform(X_politics)
    X_income_scaled = scaler_income.fit_transform(X_income)

    # Apply weights
    X_geo_weighted = X_geo_scaled * location_weight
    X_price_weighted = X_price_scaled * price_weight
    X_sqft_weighted = X_sqft_scaled * sqft_weight
    X_education_weighted = X_education_scaled * education_weight
    X_politics_weighted = X_politics_scaled * politics_weight
    X_income_weighted = X_income_scaled * income_weight

    # Combine features
    X_combined = np.hstack([
        X_geo_weighted, X_price_weighted, X_sqft_weighted,
        X_education_weighted, X_politics_weighted, X_income_weighted
    ])

    print(f"\n  Fitting K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[X_data.index, 'geo_cluster'] = kmeans.fit_predict(X_combined)
    df['geo_cluster'] = df['geo_cluster'].fillna(df['geo_cluster'].mode()[0]).astype(int)

    print(f"\n✓ Created {n_clusters} clusters")

    # Print cluster statistics
    print(f"\n  Cluster demographics summary:")
    print(f"  {'Cluster':<8} {'N':>8} {'Price':>12} {'Bach%':>6} {'Dem+':>6} {'Income':>10}")
    print("  " + "-" * 60)

    for cluster_id in sorted(df['geo_cluster'].unique()):
        cluster_data = df[df['geo_cluster'] == cluster_id]
        n = len(cluster_data)
        avg_price = cluster_data[Y_COL].mean()
        avg_edu = cluster_data['pct_bachelors_plus'].mean()
        avg_dem = cluster_data['dem_lift'].mean()
        avg_income = cluster_data[income_col].mean()

        print(f"  {cluster_id:<8} {n:>8,} ${avg_price:>10,.0f} "
              f"{avg_edu:>5.1f}% {avg_dem:>5.1f}% ${avg_income:>9,.0f}")

    # Print income range by cluster
    print(f"\n  Income range by cluster:")
    print(f"  {'Cluster':<8} {'Min Income':>12} {'Median':>12} {'Max Income':>12}")
    print("  " + "-" * 50)

    for cluster_id in sorted(df['geo_cluster'].unique()):
        cluster_data = df[df['geo_cluster'] == cluster_id]
        min_income = cluster_data[income_col].min()
        med_income = cluster_data[income_col].median()
        max_income = cluster_data[income_col].max()

        print(f"  {cluster_id:<8} ${min_income:>11,.0f} ${med_income:>11,.0f} ${max_income:>11,.0f}")

    return df


def compare_scenarios_single_cluster(model, data_dict, cluster_id, scenarios):
    """
    Compare multiple scenarios for a single cluster with detailed output
    """
    print(f"\n{'=' * 80}")
    print(f"DETAILED SCENARIO COMPARISON: CLUSTER {cluster_id}")
    print(f"{'=' * 80}")

    df_analysis = data_dict['df']
    treatments_list = data_dict['treatments_list']

    # Get cluster info
    cluster_data = df_analysis[df_analysis['geo_cluster'] == cluster_id]

    if len(cluster_data) == 0:
        print(f"❌ No data for cluster {cluster_id}")
        return None

    n_properties = len(cluster_data)
    avg_price = cluster_data[Y_COL].mean()

    print(f"\nCluster Statistics:")
    print(f"  Properties: {n_properties:,}")
    print(f"  Avg Price: ${avg_price:,.0f}")

    results = []

    for scenario_name, treatment_values in scenarios.items():
        effect = predict_scenario(model, data_dict, cluster_id, treatment_values)

        if effect is not None:
            # Get treatment descriptions
            treatment_desc = []
            for t, qty in treatment_values.items():
                if t in TREATMENTS:
                    name = TREATMENTS[t]['name']
                    treatment_desc.append(f"+{qty} {name}")

            results.append({
                'scenario': scenario_name,
                'effect': effect,
                'treatments': ', '.join(treatment_desc) if treatment_desc else 'Baseline',
                **treatment_values
            })

    results_df = pd.DataFrame(results).sort_values('effect', ascending=False)

    # Print ranking
    print(f"\n{'=' * 80}")
    print("SCENARIO RANKING (Best to Worst)")
    print(f"{'=' * 80}\n")
    print(f"{'Rank':<6} {'Scenario':<40} {'Treatments':<50} {'Effect':>12}")
    print("-" * 115)

    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i:<6} {row['scenario']:<40} {row['treatments']:<50} ${row['effect']:>11,.0f}")

    # Highlight best
    best = results_df.iloc[0]
    print(f"\n🏆 WINNER: {best['scenario']}")
    print(f"   Value increase: ${best['effect']:,.0f}")
    print(f"   Renovations: {best['treatments']}")

    # Show interaction insight
    if len(results_df) > 1:
        print(f"\n📊 Insights:")
        print(f"   Range: ${results_df['effect'].min():,.0f} to ${results_df['effect'].max():,.0f}")
        print(f"   Difference: ${results_df['effect'].max() - results_df['effect'].min():,.0f}")

    return results_df


def visualize_scenario_comparison(results_df, save_filename='scenario_comparison.png'):
    """
    Visualize scenario comparison across clusters
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 1. Heatmap: Scenarios vs Clusters
    ax1 = axes[0]
    pivot = results_df.pivot(index='scenario', columns='cluster', values='effect')
    pivot = pivot / 1000  # Convert to thousands

    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax1,
                cbar_kws={'label': 'Effect ($1000s)'}, vmin=pivot.min().min(), vmax=pivot.max().max())
    ax1.set_title('Renovation Effects by Cluster & Scenario', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Scenario', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_yticklabels(), fontsize=9)

    # 2. Bar chart: Average effect per scenario
    ax2 = axes[1]
    scenario_avg = results_df.groupby('scenario')['effect'].mean().sort_values(ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scenario_avg)))

    ax2.barh(range(len(scenario_avg)), scenario_avg / 1000, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(scenario_avg)))
    ax2.set_yticklabels(scenario_avg.index, fontsize=9)
    ax2.set_xlabel('Average Effect ($1000s)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Effect Across All Clusters', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved: {save_filename}")


def create_scenario_summary_report(results_df, detailed_results, save_filename='scenario_summary.txt'):
    """
    Create comprehensive scenario summary report
    """
    with open(save_filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-TREATMENT SCENARIO ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Overall best scenarios
        f.write("TOP 10 SCENARIOS OVERALL (Average Across All Clusters)\n")
        f.write("-" * 80 + "\n")
        scenario_avg = results_df.groupby('scenario')['effect'].mean().sort_values(ascending=False).head(10)
        for i, (scenario, effect) in enumerate(scenario_avg.items(), 1):
            f.write(f"{i:2d}. {scenario:<50} ${effect:>10,.0f}\n")

        # Best per cluster
        f.write("\n\nBEST SCENARIO BY CLUSTER\n")
        f.write("-" * 80 + "\n")
        best_per_cluster = results_df.loc[results_df.groupby('cluster')['effect'].idxmax()]
        best_per_cluster = best_per_cluster.sort_values('cluster')

        for _, row in best_per_cluster.iterrows():
            f.write(f"\nCluster {int(row['cluster'])}: {row['scenario']}\n")
            f.write(f"  Effect: ${row['effect']:,.0f}\n")
            f.write(f"  Avg Property Price: ${row['avg_price']:,.0f}\n")

        # Detailed cluster analyses
        if detailed_results:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("DETAILED CLUSTER ANALYSES\n")
            f.write("=" * 80 + "\n")

            for cluster_id, cluster_df in detailed_results.items():
                f.write(f"\n\nCLUSTER {cluster_id} - TOP 10 SCENARIOS\n")
                f.write("-" * 80 + "\n")
                top_10 = cluster_df.head(10)
                for i, (_, row) in enumerate(top_10.iterrows(), 1):
                    f.write(f"{i:2d}. {row['scenario']:<50} ${row['effect']:>10,.0f}\n")

    print(f"✓ Saved: {save_filename}")


# ============================================================================
# DOWHY VALIDATION
# ============================================================================

def validate_with_dowhy(data, treatment_col, treatment_info):
    """Validate causal estimates using DoWhy framework"""
    if not DOWHY_AVAILABLE:
        print("⚠️ DoWhy not available - skipping validation")
        return None

    print(f"\n{'=' * 80}")
    print(f"DOWHY VALIDATION: {treatment_info['name']}")
    print(f"{'=' * 80}")

    try:
        df_dowhy = data['df'].copy()
        df_dowhy[treatment_col] = data['T'].flatten()
        df_dowhy[Y_COL] = data['Y']

        confounder_names = data['confounder_names']

        print(f"\n📐 Building Causal DAG...")
        print(f"   Treatment: {treatment_col}")
        print(f"   Outcome: {Y_COL}")
        print(f"   Confounders: {len(confounder_names)}")

        graph_edges = []
        for conf in confounder_names:
            graph_edges.append(f'        {conf} -> {treatment_col};')
            graph_edges.append(f'        {conf} -> {Y_COL};')
        graph_edges.append(f'        {treatment_col} -> {Y_COL};')

        graph = "digraph {\n" + "\n".join(graph_edges) + "\n    }"

        model = CausalModel(
            data=df_dowhy,
            treatment=treatment_col,
            outcome=Y_COL,
            graph=graph
        )

        print("\n[STEP 1] Identifying causal effect...")
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        print("\n[STEP 2] Estimating with linear regression...")
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=True
        )

        print(f"✓ Estimate: ${estimate.value:,.0f}")
        print("\n✓ DoWhy validation complete")

        return {
            'model': model,
            'identified_estimand': identified_estimand,
            'estimate': estimate.value,
            'refutations': {}
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
            f.write("\n")

    print("✓ Saved: dowhy_validation_summary.txt")


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


def build_price_model(df):
    """
    Build model to predict current property price (saleamt)
    """
    print(f"\n=== BUILDING PRICE PREDICTION MODEL ===")

    # Features for price prediction
    price_features = [
        'livingareasqft_coal',
        'lotsizesqft_coal',
        'yearbuilt_coal',
        'effectiveyearbuilt_coal',
        'fireplace_count_mls',
        'half_baths_coal',
        'full_baths_coal',
        'bedrooms_mls',
        'garage_spaces_coal',
        'total_population_25plus',
        'male_bachelors_degree',
        'female_bachelors_degree',
        'pct_bachelors_plus',
        'pct_white',
        'dem_lift',
        'geo_cluster'
    ]

    # Get available features
    available_features = [f for f in price_features if f in df.columns]

    # Clean data
    df_price = df[available_features + [Y_COL]].dropna()

    # Remove outliers
    q01, q99 = df_price[Y_COL].quantile([0.01, 0.99])
    df_price = df_price[(df_price[Y_COL] >= q01) & (df_price[Y_COL] <= q99)]

    print(f"Sample size: {len(df_price):,}")
    print(f"Features: {len(available_features)}")

    X = df_price[available_features].values
    y = df_price[Y_COL].values

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\nModel Performance:")
    print(f"  Train MAE: ${train_mae:,.0f} | R²: {train_r2:.3f}")
    print(f"  Test MAE:  ${test_mae:,.0f} | R²: {test_r2:.3f}")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.3f}")

    # Save model predictions
    df_price['predicted_price'] = model.predict(X)
    df_price['price_error'] = df_price[Y_COL] - df_price['predicted_price']
    df_price['price_error_pct'] = (df_price['price_error'] / df_price[Y_COL]) * 100

    output_cols = available_features + [Y_COL, 'predicted_price', 'price_error', 'price_error_pct']
    df_price[output_cols].to_csv('price_predictions.csv', index=False)
    print(f"\n✓ Saved: price_predictions.csv")

    return model, importance_df


def build_price_model_svr(df):
    """
    Build SVR model to predict current property price (saleamt)
    """
    print(f"\n=== BUILDING PRICE PREDICTION MODEL (SVR) ===")

    # Features for price prediction
    price_features = [
        'livingareasqft_coal',
        'lotsizesqft_coal',
        'yearbuilt_coal',
        'effectiveyearbuilt_coal',
        'fireplace_count_mls',
        'half_baths_coal',
        'full_baths_coal',
        'bedrooms_mls',
        'garage_spaces_coal',
        'total_population_25plus',
        'male_bachelors_degree',
        'female_bachelors_degree',
        'pct_bachelors_plus',
        'pct_white',
        'dem_lift',
        'geo_cluster'
    ]

    # Get available features
    available_features = [f for f in price_features if f in df.columns]

    # Clean data
    df_price = df[available_features + [Y_COL]].dropna()

    # Remove outliers
    q01, q99 = df_price[Y_COL].quantile([0.01, 0.99])
    df_price = df_price[(df_price[Y_COL] >= q01) & (df_price[Y_COL] <= q99)]

    print(f"Sample size: {len(df_price):,}")
    print(f"Features: {len(available_features)}")

    X = df_price[available_features].values
    y = df_price[Y_COL].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # CRITICAL: Scale features for SVR
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    X_scaled = scaler_X.transform(X)

    # Scale y for better SVR performance
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    print("\n→ Scaling features (required for SVR)...")

    # Train SVR model
    print("→ Training SVR model...")
    model = SVR(
        kernel='rbf',  # Radial basis function kernel
        C=100,  # Regularization parameter
        epsilon=0.1,  # Epsilon in epsilon-SVR model
        gamma='scale',  # Kernel coefficient
        cache_size=1000,  # Cache size in MB
        verbose=False
    )
    model.fit(X_train_scaled, y_train_scaled)

    # Make predictions (need to inverse transform)
    y_pred_train_scaled = model.predict(X_train_scaled)
    y_pred_test_scaled = model.predict(X_test_scaled)

    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

    # Evaluate
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\nModel Performance:")
    print(f"  Train MAE:  ${train_mae:,.0f} | R²: {train_r2:.3f} | RMSE: ${train_rmse:,.0f}")
    print(f"  Test MAE:   ${test_mae:,.0f} | R²: {test_r2:.3f} | RMSE: ${test_rmse:,.0f}")

    # Feature importance (using permutation importance since SVR doesn't have feature_importances_)
    print("\n→ Calculating feature importance (this may take a moment)...")
    perm_importance = permutation_importance(
        model, X_test_scaled,
        scaler_y.transform(y_test.reshape(-1, 1)).ravel(),
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features (Permutation Importance):")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.3f} ± {row['std']:.3f}")

    # Save model predictions on full dataset
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    df_price['predicted_price'] = y_pred
    df_price['price_error'] = df_price[Y_COL] - df_price['predicted_price']
    df_price['price_error_pct'] = (df_price['price_error'] / df_price[Y_COL]) * 100

    output_cols = available_features + [Y_COL, 'predicted_price', 'price_error', 'price_error_pct']
    df_price[output_cols].to_csv('price_predictions_svr.csv', index=False)
    print(f"\n✓ Saved: price_predictions_svr.csv")

    # Return model and scalers (both needed for predictions)
    return {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'importance_df': importance_df,
        'features': available_features
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution: Analyze causal effects for multiple renovation types
    """
    print("\n" + "=" * 80)
    print(" " * 5 + "ECONML MULTI-TREATMENT CAUSAL ANALYSIS: RENOVATION EFFECTS")
    print("=" * 80)

    DATA_PATH = "/Users/jenny.lin/BASIS_AVM_Onboarding/cate_scenario_analyses/data/inference_df_with_politics.parquet"
    # Load and prepare data
    print("\n[1/8] Loading and preparing data...")
    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.lower()
    print(df.columns)

    build_price_model_svr(df)

    df = create_geo_clusters_with_politics_income(df, n_clusters=12,
                                     price_weight=0.18,
                                     sqft_weight=0.12,
                                     education_weight=0.12,
                                     race_weight=0.10,
                                     politics_weight=0.15,
                                     sqft_col='livingareasqft_coal',
                                     Y_COL='saleamt')

    # ADD THIS: Save property-to-cluster mapping
    print("\n[BONUS] Saving property-to-cluster mapping...")
    mapping_cols = ['propertyid', 'geo_cluster', 'latitude', 'longitude',
                    'saleamt', 'livingareasqft_coal']
    mapping_df = df[mapping_cols].copy()
    mapping_df.to_csv('property_cluster_mapping.csv', index=False)
    print(f"✓ Saved: property_cluster_mapping.csv ({len(mapping_df):,} properties)")

    # ========================================================================
    # PART A: INDIVIDUAL TREATMENT ANALYSIS
    # ========================================================================
    all_results = {}

    print(f"\n[2/8] Analyzing {len(TREATMENTS)} renovation types individually...")

    for treatment_col, treatment_info in TREATMENTS.items():
        if treatment_col not in df.columns:
            print(f"\n⚠️ Skipping {treatment_info['name']} - column not in data")
            continue

        print(f"\n{'=' * 80}")
        print(f"ANALYZING: {treatment_info['name']}")
        print(f"{'=' * 80}")

        data = prepare_econml_data(df, treatment_col)
        model, effects_dict = estimate_heterogeneous_effects(data, ECONML_METHOD)

        if model is None:
            continue

        cluster_df, cluster_data = analyze_effects_by_cluster(effects_dict, data, treatment_info)

        cluster_data.to_csv('all_results_cluster_data.csv')

        dowhy_results = None
        if RUN_DOWHY_VALIDATION:
            dowhy_results = validate_with_dowhy(data, treatment_col, treatment_info)

        visualize_heterogeneous_effects(
            {'cluster_effects': cluster_df, 'effects_dict': effects_dict},
            treatment_info
        )

        all_results[treatment_col] = {
            'model': model,
            'effects_dict': effects_dict,
            'cluster_effects': cluster_df,
            'cluster_data': cluster_data,
            'treatment_info': treatment_info,
            'dowhy_validation': dowhy_results
        }

    # Compare individual renovations
    if len(all_results) > 1:
        print(f"\n[3/8] Comparing individual renovation options...")
        comparison_df = compare_renovation_options(all_results)
        comparison_df.to_csv('renovation_comparison.csv', index=False)
        print("\n✓ Saved: renovation_comparison.csv")

    # Save individual results
    print(f"\n[4/8] Saving individual treatment results...")
    for treatment_col, results in all_results.items():
        filename = f"cluster_effects_{treatment_col.lower()}.csv"
        results['cluster_effects'].to_csv(filename, index=False)
        print(f"✓ Saved: {filename}")

    save_dowhy_summary(all_results)
    create_summary_report(all_results)

    # ========================================================================
    # PART B: MULTI-TREATMENT SCENARIO ANALYSIS
    # ========================================================================

    if not RUN_MULTI_TREATMENT_ANALYSIS:
        print(f"\n⚠️ Multi-treatment analysis disabled")
        print(f"\n{'=' * 80}")
        print("✅ ANALYSIS COMPLETE!")
        print(f"{'=' * 80}")
        return {'individual_results': all_results}

    print(f"\n[5/8] Starting multi-treatment scenario analysis...")

    # Define which treatments to analyze together
    multi_treatments = ['half_baths_coal', 'bedrooms_mls', 'full_baths_coal', 'garage_spaces_coal']
    multi_treatments = [t for t in multi_treatments if t in df.columns]

    if len(multi_treatments) < 2:
        print(f"\n⚠️ Need at least 2 treatments for multi-treatment analysis")
        return {'individual_results': all_results}

    # Estimate multi-treatment model
    model_multi, df_multi, data_multi = estimate_multi_treatment_effects(df, multi_treatments)

    if model_multi is None:
        return {'individual_results': all_results}

    print('Cluster Data Printed')

    # ========================================================================
    # DEFINE COMPREHENSIVE SCENARIOS
    # ========================================================================

    print(f"\n[6/8] Defining renovation scenarios...")

    scenarios = {
        # ===== BASELINE =====
        'Baseline (No Changes)': {},

        # ===== SINGLE RENOVATIONS =====
        '1 Half Bath': {'half_baths_coal': 1},
        '1 Full Bath': {'full_baths_coal': 1},
        '1 Bedroom': {'bedrooms_mls': 1},
        '1 Garage Space': {'garage_spaces_coal': 1},
        '2 Bedrooms': {'bedrooms_mls': 2},

        # ===== BATHROOM COMBINATIONS =====
        '1 Half Bath + 1 Full Bath': {
            'half_baths_coal': 1,
            'full_baths_coal': 1
        },
        '2 Half Baths': {'half_baths_coal': 2},
        '2 Full Baths': {'full_baths_coal': 2},

        # ===== BEDROOM + BATHROOM COMBOS =====
        '1 Bedroom + 1 Half Bath': {
            'bedrooms_mls': 1,
            'half_baths_coal': 1
        },
        '1 Bedroom + 1 Full Bath': {
            'bedrooms_mls': 1,
            'full_baths_coal': 1
        },
        '2 Bedrooms + 1 Full Bath': {
            'bedrooms_mls': 2,
            'full_baths_coal': 1
        },
        '2 Bedrooms + 2 Full Baths': {
            'bedrooms_mls': 2,
            'full_baths_coal': 2
        },

        # ===== BEDROOM + GARAGE COMBOS =====
        '1 Bedroom + 1 Garage': {
            'bedrooms_mls': 1,
            'garage_spaces_coal': 1
        },
        '2 Bedrooms + 1 Garage': {
            'bedrooms_mls': 2,
            'garage_spaces_coal': 1
        },

        # ===== BATHROOM + GARAGE COMBOS =====
        '1 Full Bath + 1 Garage': {
            'full_baths_coal': 1,
            'garage_spaces_coal': 1
        },
        '1 Half Bath + 1 Garage': {
            'half_baths_coal': 1,
            'garage_spaces_coal': 1
        },

        # ===== MODERATE RENOVATIONS (3 items) =====
        'Moderate: 1 Bed + 1 Half + 1 Garage': {
            'bedrooms_mls': 1,
            'half_baths_coal': 1,
            'garage_spaces_coal': 1
        },
        'Moderate: 1 Bed + 1 Full + 1 Garage': {
            'bedrooms_mls': 1,
            'full_baths_coal': 1,
            'garage_spaces_coal': 1
        },
        'Moderate: 2 Beds + 1 Half Bath': {
            'bedrooms_mls': 2,
            'half_baths_coal': 1
        },

        # ===== MAJOR RENOVATIONS (4+ items) =====
        'Major: 2 Beds + 1 Full + 1 Garage': {
            'bedrooms_mls': 2,
            'full_baths_coal': 1,
            'garage_spaces_coal': 1
        },
        'Major: 2 Beds + 2 Full Baths + 1 Garage': {
            'bedrooms_mls': 2,
            'full_baths_coal': 2,
            'garage_spaces_coal': 1
        },
        'Major: 2 Beds + 1 Half + 1 Full + 1 Garage': {
            'bedrooms_mls': 2,
            'half_baths_coal': 1,
            'full_baths_coal': 1,
            'garage_spaces_coal': 1
        },

        # ===== LUXURY RENOVATIONS =====
        'Luxury: 3 Bedrooms + 2 Full Baths': {
            'bedrooms_mls': 3,
            'full_baths_coal': 2
        },
        'Luxury: 3 Bedrooms + 2 Full + 1 Garage': {
            'bedrooms_mls': 3,
            'full_baths_coal': 2,
            'garage_spaces_coal': 1
        },
        'Luxury: 3 Beds + 2 Full + 1 Half + 2 Garage': {
            'bedrooms_mls': 3,
            'full_baths_coal': 2,
            'half_baths_coal': 1,
            'garage_spaces_coal': 2
        },

        # ===== GARAGE-FOCUSED =====
        '2 Garage Spaces': {'garage_spaces_coal': 2},
        '1 Bedroom + 2 Garage Spaces': {
            'bedrooms_mls': 1,
            'garage_spaces_coal': 2
        },

        # ===== BATHROOM-FOCUSED =====
        'Bathroom Upgrade: 1 Half + 2 Full': {
            'half_baths_coal': 1,
            'full_baths_coal': 2
        },

        # ===== BEDROOM-FOCUSED =====
        '3 Bedrooms Only': {'bedrooms_mls': 3},
        '4 Bedrooms': {'bedrooms_mls': 4},
    }

    print(f"✓ Defined {len(scenarios)} renovation scenarios")

#     # ========================================================================
#     # ANALYZE SCENARIOS ACROSS ALL CLUSTERS
#     # ========================================================================

    print(f"\n[7/8] Analyzing scenarios across clusters...")
    results_all = analyze_scenarios_by_cluster(
        model_multi, data_multi, scenarios, top_n_clusters=TOP_CLUSTERS_TO_ANALYZE
    )
    results_all.to_csv('scenario_analysis_all_clusters.csv', index=False)
    print("✓ Saved: scenario_analysis_all_clusters.csv")

#     # ========================================================================
#     # DETAILED ANALYSIS FOR TOP CLUSTERS
#     # ========================================================================

    print(f"\n[8/8] Detailed analysis for top {DETAILED_CLUSTERS_TO_ANALYZE} clusters...")
    top_clusters = df_multi['geo_cluster'].value_counts().head(DETAILED_CLUSTERS_TO_ANALYZE).index

    detailed_results = {}
    for cluster_id in top_clusters:
        results_cluster = compare_scenarios_single_cluster(
            model_multi, data_multi, cluster_id, scenarios
        )

        if results_cluster is not None:
            filename = f'scenario_analysis_cluster_{cluster_id}.csv'
            results_cluster.to_csv(filename, index=False)
            print(f"✓ Saved: {filename}")
            detailed_results[cluster_id] = results_cluster

#     # ========================================================================
#     # VISUALIZATIONS AND SUMMARY
#     # ========================================================================

    print(f"\n📊 Creating visualizations...")
    visualize_scenario_comparison(results_all)

    print(f"\n📄 Creating comprehensive summary report...")
    create_scenario_summary_report(results_all, detailed_results)

    print(f"\n{'=' * 80}")
    print("✅ MULTI-TREATMENT SCENARIO ANALYSIS COMPLETE!")
    print(f"{'=' * 80}")

    print(f"\n📁 Output Files Generated:")
    print(f"   • renovation_comparison.csv - Individual treatment comparison")
    print(f"   • scenario_analysis_all_clusters.csv - All scenarios across clusters")
    print(f"   • scenario_analysis_cluster_X.csv - Detailed per-cluster analysis")
    print(f"   • scenario_comparison.png - Heatmap visualization")
    print(f"   • scenario_summary.txt - Comprehensive text report")
    print(f"   • causal_analysis_summary.txt - Individual treatment summary")
    print(f"   • cluster_effects_*.csv - Individual treatment effects by cluster")

    print(f"\n{'=' * 80}")
    print("✅ ALL ANALYSES COMPLETE!")
    print(f"{'=' * 80}")

    return {
        'individual_results': all_results,
        'multi_treatment_model': model_multi,
        'multi_treatment_data': data_multi,
        'scenario_results': results_all,
        'detailed_results': detailed_results
    }

#
# # ============================================================================
# # CUSTOM SCENARIO RUNNER (OPTIONAL)
# # ============================================================================
#
def run_custom_scenario(results, cluster_id, custom_renovations):
    """
    Run a custom scenario not in the predefined list

    Example usage:
        run_custom_scenario(results, cluster_id=44,
            custom_renovations={
                'bedrooms_mls': 1,
                'half_baths_coal': 2,
                'garage_spaces_coal': 1
            })
    """
    if results['multi_treatment_model'] is None:
        print("❌ Multi-treatment model not available")
        return None

    print(f"\n{'=' * 80}")
    print(f"CUSTOM SCENARIO ANALYSIS: CLUSTER {cluster_id}")
    print(f"{'=' * 80}")

    # Describe the scenario
    renovation_desc = []
    for treatment, qty in custom_renovations.items():
        if treatment in TREATMENTS:
            name = TREATMENTS[treatment]['name']
            renovation_desc.append(f"+{qty} {name}(s)")

    print(f"\nRenovations: {', '.join(renovation_desc)}")

    # Predict effect
    effect = predict_scenario(
        results['multi_treatment_model'],
        results['multi_treatment_data'],
        cluster_id,
        custom_renovations
    )

    if effect:
        print(f"\n💰 Predicted Value Increase: ${effect:,.0f}")

        # Compare to baseline scenarios
        if 'scenario_results' in results:
            cluster_scenarios = results['scenario_results'][
                results['scenario_results']['cluster'] == cluster_id
                ]

            if len(cluster_scenarios) > 0:
                rank_position = (cluster_scenarios['effect'] > effect).sum() + 1
                total_scenarios = len(cluster_scenarios)

                print(f"\n📊 Comparison to Predefined Scenarios:")
                print(f"   This scenario ranks #{rank_position} out of {total_scenarios}")

                # Show nearby scenarios
                nearby = cluster_scenarios.iloc[(cluster_scenarios['effect'] - effect).abs().argsort()[:3]]
                print(f"\n   Similar scenarios:")
                for _, row in nearby.iterrows():
                    print(f"   • {row['scenario']:<50} ${row['effect']:>10,.0f}")

    return effect


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

if __name__ == "__main__":
    if not ECONML_AVAILABLE:
        print("\n❌ ERROR: EconML not installed")
        print("Install with: pip install econml")
        print("Then run this script again")
    else:
        results = main()

        # ====================================================================
        # OPTIONAL: Run custom scenarios after main analysis
        # ====================================================================

        print(f"\n{'=' * 80}")
        print("CUSTOM SCENARIO EXAMPLES")
        print(f"{'=' * 80}")

        if results.get('multi_treatment_model') is not None:
            # Example 1: Specific renovation for a cluster
            print(f"\n📋 Example 1: What if I add 1 half bath + 2 bedrooms in top cluster?")
            top_cluster = results['multi_treatment_data']['df']['geo_cluster'].value_counts().index[0]
            run_custom_scenario(results, cluster_id=top_cluster,
                                custom_renovations={
                                    'half_baths_coal': 1,
                                    'bedrooms_mls': 2
                                })

            print(f"\n💡 TIP: Use run_custom_scenario() to test any renovation combination!")
            print(f"   Example: run_custom_scenario(results, cluster_id=X, custom_renovations={{...}})")