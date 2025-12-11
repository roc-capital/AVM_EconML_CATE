import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

Y_COL = 'saleamt'
PROPERTYID_COL = 'propertyid'
DECAY_FACTOR = 0.9
N_GEO_CLUSTERS = 12
MIN_PRICE_THRESHOLD = 100000
TEST_SIZE = 0.2
RANDOM_STATE = 42

BASE_FEATURES = [
    'livingareasqft_coal', 'lotsizesqft_coal', 'yearbuilt_coal',
    'effectiveyearbuilt_coal', 'fireplace_count_mls', 'half_baths_coal',
    'full_baths_coal', 'bedrooms_mls', 'garage_spaces_coal',
    'total_population_25plus', 'male_bachelors_degree', 'female_bachelors_degree',
    'pct_bachelors_plus', 'geo_cluster'
]

ENGINEERED_FEATURES = [
    'sqft_per_bedroom', 'bedrooms_per_1000sqft', 'sqft_per_bath', 'bath_to_bedroom_ratio',
    'lot_to_living_ratio', 'property_age', 'age_squared', 'is_new', 'is_vintage',
    'has_renovation', 'has_garage', 'has_fireplace', 'luxury_score', 'log_sqft', 'log_lotsize'
]

CLUSTER_FEATURES = ['cluster_avg_price', 'cluster_med_price', 'cluster_price_std']

CATEGORICAL_FEATURES = ['geo_cluster', 'is_new', 'is_vintage', 'has_renovation',
                        'has_garage', 'has_fireplace']


def safe_div(num, denom, fill=0, clip=1e6):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denom != 0, num / denom, fill)
        return np.clip(result, -clip, clip)


def collapse_to_property_level(df):
    if PROPERTYID_COL not in df.columns:
        return df
    df_sorted = df.sort_values([PROPERTYID_COL, "year"])
    last_cols = [c for c in df.columns if c not in [Y_COL, PROPERTYID_COL]]
    df_last = df_sorted.groupby(PROPERTYID_COL, as_index=False)[last_cols].last()

    def discounted_saleamt(group):
        weights = DECAY_FACTOR ** (group["year"].max() - group["year"])
        return np.average(group[Y_COL], weights=weights)

    df_saleamt = df_sorted.groupby(PROPERTYID_COL).apply(
        discounted_saleamt, include_groups=False
    ).rename(Y_COL).reset_index()
    return df_last.merge(df_saleamt, on=PROPERTYID_COL, how="left")


def create_features(df):
    df['pct_bachelors_plus'] = ((df['male_bachelors_degree'] + df['female_bachelors_degree']) /
                                df['total_population_25plus']).fillna(0) * 100

    df['sqft_per_bedroom'] = safe_div(df['livingareasqft_coal'], df['bedrooms_mls'] + 1)
    df['bedrooms_per_1000sqft'] = safe_div(df['bedrooms_mls'] * 1000, df['livingareasqft_coal'])

    total_baths = df['full_baths_coal'] + df['half_baths_coal'] * 0.5
    df['sqft_per_bath'] = safe_div(df['livingareasqft_coal'], total_baths + 1)
    df['bath_to_bedroom_ratio'] = safe_div(total_baths, df['bedrooms_mls'] + 1)
    df['lot_to_living_ratio'] = safe_div(df['lotsizesqft_coal'], df['livingareasqft_coal'] + 1, clip=100)

    current_year = df['year'].median() if 'year' in df.columns else 2024
    df['property_age'] = np.clip(current_year - df['yearbuilt_coal'], 0, 200)
    df['age_squared'] = df['property_age'] ** 2
    df['is_new'] = (df['property_age'] <= 5).astype(int)
    df['is_vintage'] = (df['property_age'] >= 50).astype(int)
    df['has_renovation'] = (df['effectiveyearbuilt_coal'] > df['yearbuilt_coal']).astype(int)

    df['has_garage'] = (df['garage_spaces_coal'] > 0).astype(int)
    df['has_fireplace'] = (df['fireplace_count_mls'] > 0).astype(int)
    df['luxury_score'] = df['garage_spaces_coal'] + df['fireplace_count_mls']

    df['log_sqft'] = np.log1p(df['livingareasqft_coal'])
    df['log_lotsize'] = np.log1p(df['lotsizesqft_coal'])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


def create_geo_clusters(df_train, df_test):
    weights = np.array([0.50, 0.50, 0.20, 0.20, 0.10])
    required_cols = ['latitude', 'longitude', 'livingareasqft_coal', 'pct_bachelors_plus', 'yearbuilt_coal']

    X_train = df_train[required_cols].dropna()
    current_year = df_train['year'].median() if 'year' in df_train.columns else 2024
    X_train['property_age'] = current_year - X_train['yearbuilt_coal']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[['latitude', 'longitude', 'livingareasqft_coal',
                                                   'pct_bachelors_plus', 'property_age']]) * weights

    kmeans = KMeans(n_clusters=N_GEO_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df_train.loc[X_train.index, 'geo_cluster'] = kmeans.fit_predict(X_train_scaled)
    df_train['geo_cluster'].fillna(df_train['geo_cluster'].mode()[0], inplace=True)
    df_train['geo_cluster'] = df_train['geo_cluster'].astype(int)

    X_test = df_test[required_cols].dropna()
    X_test['property_age'] = current_year - X_test['yearbuilt_coal']
    X_test_scaled = scaler.transform(X_test[['latitude', 'longitude', 'livingareasqft_coal',
                                             'pct_bachelors_plus', 'property_age']]) * weights

    df_test.loc[X_test.index, 'geo_cluster'] = kmeans.predict(X_test_scaled)
    df_test['geo_cluster'].fillna(df_train['geo_cluster'].mode()[0], inplace=True)
    df_test['geo_cluster'] = df_test['geo_cluster'].astype(int)

    return df_train, df_test, {'kmeans': kmeans, 'scaler': scaler, 'current_year': current_year,
                               'weights': weights, 'default_cluster': df_train['geo_cluster'].mode()[0]}


def add_cluster_features(df_train, df_test):
    stats = df_train.groupby('geo_cluster')[Y_COL].agg(['mean', 'median', 'std']).reset_index()
    stats.columns = ['geo_cluster'] + CLUSTER_FEATURES
    stats['cluster_price_std'].fillna(stats['cluster_price_std'].median(), inplace=True)

    df_train = df_train.merge(stats, on='geo_cluster', how='left')
    df_test = df_test.merge(stats, on='geo_cluster', how='left')

    for col in CLUSTER_FEATURES:
        median_val = df_train[col].median()
        df_train[col].fillna(median_val, inplace=True)
        df_test[col].fillna(median_val, inplace=True)

    return df_train, df_test, stats


def build_model(df_train, df_test):
    df_train = df_train[df_train[Y_COL] >= MIN_PRICE_THRESHOLD]
    df_test = df_test[df_test[Y_COL] >= MIN_PRICE_THRESHOLD]

    df_train = create_features(df_train)
    df_test = create_features(df_test)

    df_train = df_train[df_train['lot_to_living_ratio'] <= 3.5]

    df_train, df_test, cluster_objs = create_geo_clusters(df_train, df_test)
    df_train, df_test, cluster_stats = add_cluster_features(df_train, df_test)

    all_features = [f for f in BASE_FEATURES + ENGINEERED_FEATURES + CLUSTER_FEATURES
                    if f in df_train.columns and f in df_test.columns]

    df_train = df_train[all_features + [Y_COL, PROPERTYID_COL]].dropna()
    df_test = df_test[all_features + [Y_COL, PROPERTYID_COL]].dropna()

    q01, q99 = df_train[Y_COL].quantile([0.01, 0.99])
    df_train = df_train[(df_train[Y_COL] >= q01) & (df_train[Y_COL] <= q99)]

    X_train, y_train = df_train[all_features], np.log1p(df_train[Y_COL].values)
    X_test, y_test = df_test[all_features], np.log1p(df_test[Y_COL].values)

    cat_features_idx = [i for i, f in enumerate(all_features) if f in CATEGORICAL_FEATURES]

    train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_features_idx)

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        subsample=0.8,
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
        task_type='CPU'
    )

    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50, verbose=False)

    y_pred_train, y_pred_test = np.expm1(model.predict(X_train)), np.expm1(model.predict(X_test))
    y_train_orig, y_test_orig = np.expm1(y_train), np.expm1(y_test)

    train_r2 = r2_score(y_train_orig, y_pred_train)
    test_r2 = r2_score(y_test_orig, y_pred_test)
    test_mae = mean_absolute_error(y_test_orig, y_pred_test)
    test_mape = np.mean(np.abs((y_test_orig - y_pred_test) / y_test_orig) * 100)

    print(f"TRAIN R²: {train_r2:.4f} | TEST R²: {test_r2:.4f}, MAE: ${test_mae:,.0f}, MAPE: {test_mape:.2f}%")

    train_results = pd.DataFrame({
        PROPERTYID_COL: df_train[PROPERTYID_COL].values,
        'actual': y_train_orig, 'predicted': y_pred_train,
        'error_pct': ((y_train_orig - y_pred_train) / y_train_orig) * 100
    })

    test_results = pd.DataFrame({
        PROPERTYID_COL: df_test[PROPERTYID_COL].values,
        'actual': y_test_orig, 'predicted': y_pred_test,
        'error_pct': ((y_test_orig - y_pred_test) / y_test_orig) * 100
    })

    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'cluster_objs': cluster_objs,
        'cluster_stats': cluster_stats,
        'features': all_features,
        'cat_features_idx': cat_features_idx,
        'train_results': train_results,
        'test_results': test_results,
        'feature_importance': feature_importance,
        'metrics': {'train_r2': train_r2, 'test_r2': test_r2, 'test_mae': test_mae, 'test_mape': test_mape}
    }


def predict(model, cluster_objs, cluster_stats, features, df, property_ids, cat_features_idx):
    df = df[df[PROPERTYID_COL].isin(property_ids)]
    if len(df) == 0:
        return pd.DataFrame()

    df = create_features(df)

    required_cols = ['latitude', 'longitude', 'livingareasqft_coal', 'pct_bachelors_plus', 'yearbuilt_coal']
    X = df[required_cols].dropna()
    X['property_age'] = cluster_objs['current_year'] - X['yearbuilt_coal']
    X_scaled = cluster_objs['scaler'].transform(
        X[['latitude', 'longitude', 'livingareasqft_coal', 'pct_bachelors_plus', 'property_age']]
    ) * cluster_objs['weights']

    df.loc[X.index, 'geo_cluster'] = cluster_objs['kmeans'].predict(X_scaled)
    df['geo_cluster'].fillna(cluster_objs['default_cluster'], inplace=True)
    df['geo_cluster'] = df['geo_cluster'].astype(int)

    df = df.merge(cluster_stats, on='geo_cluster', how='left')
    for col in CLUSTER_FEATURES:
        df[col].fillna(df[col].median(), inplace=True)

    df = df[features + [PROPERTYID_COL]].dropna()
    X = df[features]

    return pd.DataFrame({
        PROPERTYID_COL: df[PROPERTYID_COL].values,
        'predicted_price': np.expm1(model.predict(X))
    })


if __name__ == "__main__":
    DATA_PATH = "/Users/jenny.lin/BASIS_AVM_Onboarding/cate_scenario_analyses/data/inference_df_with_politics.parquet"

    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.lower()
    df = collapse_to_property_level(df)

    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    results = build_model(df_train, df_test)

    results['train_results'].to_csv('train_predictions_catboost.csv', index=False)
    results['test_results'].to_csv('test_predictions_catboost.csv', index=False)
    results['feature_importance'].to_csv('feature_importance_catboost.csv', index=False)

    print(f"\nTOP 10 FEATURES:")
    for _, row in results['feature_importance'].head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")