import pandas as pd
import numpy as np
from xgboost import XGBRegressor
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

def filter_extreme_properties(df, lot_to_living_threshold=3.5):
    if 'lot_to_living_ratio' not in df.columns:
        df['lot_to_living_ratio'] = df['lotsizesqft_coal'] / (df['livingareasqft_coal'] + 1)
    return df[df['lot_to_living_ratio'] <= lot_to_living_threshold].copy()

def collapse_to_property_level(df, decay=0.9):
    if PROPERTYID_COL not in df.columns:
        return df
    last_cols = [c for c in df.columns if c not in [Y_COL, PROPERTYID_COL]]
    df_sorted = df.sort_values([PROPERTYID_COL, "year"])
    df_last = df_sorted.groupby(PROPERTYID_COL, as_index=False)[last_cols].last()
    def discounted_saleamt(group):
        max_year = group["year"].max()
        weights = decay ** (max_year - group["year"])
        return np.average(group[Y_COL], weights=weights)
    df_saleamt = df_sorted.groupby(PROPERTYID_COL).apply(
        discounted_saleamt, include_groups=False
    ).rename(Y_COL).reset_index()
    return df_last.merge(df_saleamt, on=PROPERTYID_COL, how="left")

def create_geo_clusters_no_leakage(df_train, df_test, n_clusters=N_GEO_CLUSTERS):
    location_weight = 0.50
    sqft_weight = 0.20
    education_weight = 0.20
    age_weight = 0.10

    df_train = df_train.copy()
    df_train['pct_bachelors_plus'] = (
        (df_train['male_bachelors_degree'] + df_train['female_bachelors_degree']) /
        df_train['total_population_25plus']
    ).fillna(0) * 100

    if 'pct_white' not in df_train.columns and 'non_hispanic_white_population' in df_train.columns:
        df_train['pct_white'] = (
            df_train['non_hispanic_white_population'] / df_train['total_population']
        ).fillna(0) * 100

    required_cols = ['latitude', 'longitude', 'livingareasqft_coal', 'pct_bachelors_plus', 'yearbuilt_coal']
    X_train_data = df_train[required_cols].dropna()
    current_year = df_train['year'].median() if 'year' in df_train.columns else 2024
    X_train_data['property_age'] = current_year - X_train_data['yearbuilt_coal']

    scaler_location = StandardScaler().fit(X_train_data[['latitude', 'longitude']])
    scaler_sqft = StandardScaler().fit(X_train_data[['livingareasqft_coal']])
    scaler_edu = StandardScaler().fit(X_train_data[['pct_bachelors_plus']])
    scaler_age = StandardScaler().fit(X_train_data[['property_age']])

    features_train = [
        scaler_location.transform(X_train_data[['latitude', 'longitude']]) * location_weight,
        scaler_sqft.transform(X_train_data[['livingareasqft_coal']]) * sqft_weight,
        scaler_edu.transform(X_train_data[['pct_bachelors_plus']]) * education_weight,
        scaler_age.transform(X_train_data[['property_age']]) * age_weight
    ]
    X_combined_train = np.hstack(features_train)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_combined_train)
    df_train.loc[X_train_data.index, 'geo_cluster'] = kmeans.predict(X_combined_train)
    df_train['geo_cluster'] = df_train['geo_cluster'].fillna(df_train['geo_cluster'].mode()[0]).astype(int)

    df_test = df_test.copy()
    df_test['pct_bachelors_plus'] = (
        (df_test['male_bachelors_degree'] + df_test['female_bachelors_degree']) /
        df_test['total_population_25plus']
    ).fillna(0) * 100

    if 'pct_white' not in df_test.columns and 'non_hispanic_white_population' in df_test.columns:
        df_test['pct_white'] = (
            df_test['non_hispanic_white_population'] / df_test['total_population']
        ).fillna(0) * 100

    X_test_data = df_test[required_cols].dropna()
    X_test_data['property_age'] = current_year - X_test_data['yearbuilt_coal']

    features_test = [
        scaler_location.transform(X_test_data[['latitude', 'longitude']]) * location_weight,
        scaler_sqft.transform(X_test_data[['livingareasqft_coal']]) * sqft_weight,
        scaler_edu.transform(X_test_data[['pct_bachelors_plus']]) * education_weight,
        scaler_age.transform(X_test_data[['property_age']]) * age_weight
    ]
    X_combined_test = np.hstack(features_test)

    df_test.loc[X_test_data.index, 'geo_cluster'] = kmeans.predict(X_combined_test)
    df_test['geo_cluster'] = df_test['geo_cluster'].fillna(df_test['geo_cluster'].mode()[0]).astype(int)

    clustering_objects = {
        'kmeans': kmeans,
        'scaler_location': scaler_location,
        'scaler_sqft': scaler_sqft,
        'scaler_edu': scaler_edu,
        'scaler_age': scaler_age,
        'current_year': current_year,
        'location_weight': location_weight,
        'sqft_weight': sqft_weight,
        'education_weight': education_weight,
        'age_weight': age_weight,
        'default_cluster': df_train['geo_cluster'].mode()[0]
    }

    return df_train, df_test, clustering_objects

def create_features(df):
    df = df.copy()

    def safe_div(num, denom, fill=0, clip=1e6):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = num / denom
            result = np.where(np.isfinite(result), result, fill)
            return np.clip(result, -clip, clip)

    if 'livingareasqft_coal' in df.columns and 'bedrooms_mls' in df.columns:
        df['sqft_per_bedroom'] = safe_div(df['livingareasqft_coal'], df['bedrooms_mls'] + 1)
        df['bedrooms_per_1000sqft'] = safe_div(df['bedrooms_mls'] * 1000, df['livingareasqft_coal'])

    if 'full_baths_coal' in df.columns:
        total_baths = df['full_baths_coal'] + df['half_baths_coal'] * 0.5
        df['sqft_per_bath'] = safe_div(df['livingareasqft_coal'], total_baths + 1)
        df['bath_to_bedroom_ratio'] = safe_div(total_baths, df['bedrooms_mls'] + 1)

    if 'lotsizesqft_coal' in df.columns:
        df['lot_to_living_ratio'] = safe_div(df['lotsizesqft_coal'], df['livingareasqft_coal'] + 1, clip=100)

    if 'yearbuilt_coal' in df.columns and 'year' in df.columns:
        df['property_age'] = np.clip(df['year'] - df['yearbuilt_coal'], 0, 200)
        df['age_squared'] = df['property_age'] ** 2
        df['is_new'] = (df['property_age'] <= 5).astype(int)
        df['is_vintage'] = (df['property_age'] >= 50).astype(int)

    if 'effectiveyearbuilt_coal' in df.columns:
        df['has_renovation'] = (df['effectiveyearbuilt_coal'] > df['yearbuilt_coal']).astype(int)

    luxury_score = 0
    if 'garage_spaces_coal' in df.columns:
        df['has_garage'] = (df['garage_spaces_coal'] > 0).astype(int)
        luxury_score += df['garage_spaces_coal']
    if 'fireplace_count_mls' in df.columns:
        df['has_fireplace'] = (df['fireplace_count_mls'] > 0).astype(int)
        luxury_score += df['fireplace_count_mls']
    df['luxury_score'] = luxury_score

    if 'livingareasqft_coal' in df.columns:
        df['log_sqft'] = np.log1p(df['livingareasqft_coal'])
    if 'lotsizesqft_coal' in df.columns:
        df['log_lotsize'] = np.log1p(df['lotsizesqft_coal'])

    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df

def create_cluster_features_no_leakage(df_train, df_test):
    train_stats = df_train.groupby('geo_cluster')[Y_COL].agg(['mean', 'median', 'std']).reset_index()
    train_stats.columns = ['geo_cluster', 'cluster_avg_price', 'cluster_med_price', 'cluster_price_std']
    train_stats['cluster_price_std'] = train_stats['cluster_price_std'].fillna(train_stats['cluster_price_std'].median())

    df_train = df_train.merge(train_stats, on='geo_cluster', how='left')
    df_test = df_test.merge(train_stats, on='geo_cluster', how='left')

    for col in ['cluster_avg_price', 'cluster_med_price', 'cluster_price_std']:
        global_median = df_train[col].median()
        df_train[col] = df_train[col].fillna(global_median)
        df_test[col] = df_test[col].fillna(global_median)

    return df_train, df_test

def build_single_model_no_leakage(df_train, df_test, use_cluster_features=True):
    df_train = df_train[df_train[Y_COL] >= MIN_PRICE_THRESHOLD].copy()
    df_test = df_test[df_test[Y_COL] >= MIN_PRICE_THRESHOLD].copy()

    df_train = create_features(df_train)
    df_test = create_features(df_test)

    base_features = [
        'livingareasqft_coal', 'lotsizesqft_coal', 'yearbuilt_coal',
        'effectiveyearbuilt_coal', 'fireplace_count_mls', 'half_baths_coal',
        'full_baths_coal', 'bedrooms_mls', 'garage_spaces_coal',
        'total_population_25plus', 'male_bachelors_degree', 'female_bachelors_degree',
        'pct_bachelors_plus', 'geo_cluster'
    ]

    engineered_features = [
        'sqft_per_bedroom', 'bedrooms_per_1000sqft', 'sqft_per_bath', 'bath_to_bedroom_ratio',
        'lot_to_living_ratio', 'property_age', 'age_squared',
        'is_new', 'is_vintage', 'has_renovation',
        'has_garage', 'has_fireplace', 'luxury_score',
        'log_sqft', 'log_lotsize'
    ]

    available_features = [f for f in base_features + engineered_features
                          if f in df_train.columns and f in df_test.columns]

    if use_cluster_features:
        df_train, df_test = create_cluster_features_no_leakage(df_train, df_test)
        cluster_features = ['cluster_avg_price', 'cluster_med_price', 'cluster_price_std']
        available_features.extend([f for f in cluster_features if f in df_train.columns])

    df_train_clean = df_train[available_features + [Y_COL, PROPERTYID_COL]].copy().dropna()
    df_test_clean = df_test[available_features + [Y_COL, PROPERTYID_COL]].copy().dropna()

    q01, q99 = df_train_clean[Y_COL].quantile([0.01, 0.99])
    df_train_clean = df_train_clean[(df_train_clean[Y_COL] >= q01) & (df_train_clean[Y_COL] <= q99)]

    X_train = df_train_clean[available_features].values
    y_train = np.log1p(df_train_clean[Y_COL].values)
    X_test = df_test_clean[available_features].values
    y_test = np.log1p(df_test_clean[Y_COL].values)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
    y_train = np.nan_to_num(y_train, nan=np.median(y_train[np.isfinite(y_train)]))
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
    y_test = np.nan_to_num(y_test, nan=np.median(y_test[np.isfinite(y_test)]))

    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist',
        verbosity=0
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_train = np.expm1(model.predict(X_train))
    y_pred_test = np.expm1(model.predict(X_test))

    train_mae = mean_absolute_error(y_train_orig, y_pred_train)
    train_r2 = r2_score(y_train_orig, y_pred_train)
    train_ape = np.abs((y_train_orig - y_pred_train) / y_train_orig) * 100
    train_mape = np.mean(train_ape)

    test_mae = mean_absolute_error(y_test_orig, y_pred_test)
    test_r2 = r2_score(y_test_orig, y_pred_test)
    test_ape = np.abs((y_test_orig - y_pred_test) / y_test_orig) * 100
    test_mape = np.mean(test_ape)
    test_medape = np.median(test_ape)

    print(f"\nTRAIN - R²: {train_r2:.4f}, MAE: ${train_mae:,.0f}, MAPE: {train_mape:.2f}%")
    print(f"TEST  - R²: {test_r2:.4f}, MAE: ${test_mae:,.0f}, MAPE: {test_mape:.2f}%, MedAPE: {test_medape:.2f}%")

    train_results = pd.DataFrame({
        PROPERTYID_COL: df_train_clean[PROPERTYID_COL].values,
        'actual_price': y_train_orig,
        'predicted_price': y_pred_train,
        'price_error': y_train_orig - y_pred_train,
        'price_error_pct': ((y_train_orig - y_pred_train) / y_train_orig) * 100,
        'abs_pct_error': train_ape,
        'dataset': 'train'
    })

    test_results = pd.DataFrame({
        PROPERTYID_COL: df_test_clean[PROPERTYID_COL].values,
        'actual_price': y_test_orig,
        'predicted_price': y_pred_test,
        'price_error': y_test_orig - y_pred_test,
        'price_error_pct': ((y_test_orig - y_pred_test) / y_test_orig) * 100,
        'abs_pct_error': test_ape,
        'dataset': 'test'
    })

    train_results = pd.concat([train_results.reset_index(drop=True), df_train_clean.reset_index(drop=True)], axis=1)
    test_results = pd.concat([test_results.reset_index(drop=True), df_test_clean.reset_index(drop=True)], axis=1)

    train_results = train_results.loc[:, ~train_results.columns.duplicated()]
    test_results = test_results.loc[:, ~test_results.columns.duplicated()]

    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    train_results.to_csv('train_predictions_no_leakage.csv', index=False)
    test_results.to_csv('test_predictions_no_leakage.csv', index=False)
    feature_importance.to_csv('feature_importance_no_leakage.csv', index=False)

    worst_predictions = pd.concat([
        test_results.nlargest(50, 'price_error_pct'),
        test_results.nsmallest(50, 'price_error_pct')
    ])
    worst_predictions.to_csv('worst_predictions_no_leakage.csv', index=False)

    return {
        'model': model,
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_mape': train_mape,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_medape': test_medape,
        'train_results': train_results,
        'test_results': test_results,
        'feature_importance': feature_importance
    }

def predict_on_property_ids(model, clustering_objects, train_cluster_stats, df_original,
                            property_ids, available_features, current_year=None):
    df_pred = df_original[df_original[PROPERTYID_COL].isin(property_ids)].copy()
    if len(df_pred) == 0:
        return pd.DataFrame()

    df_pred = create_features(df_pred)

    df_pred['pct_bachelors_plus'] = (
        (df_pred['male_bachelors_degree'] + df_pred['female_bachelors_degree']) /
        df_pred['total_population_25plus']
    ).fillna(0) * 100

    if 'pct_white' not in df_pred.columns and 'non_hispanic_white_population' in df_pred.columns:
        df_pred['pct_white'] = (
            df_pred['non_hispanic_white_population'] / df_pred['total_population']
        ).fillna(0) * 100

    required_cols = ['latitude', 'longitude', 'livingareasqft_coal', 'pct_bachelors_plus', 'yearbuilt_coal']
    X_pred_data = df_pred[required_cols].dropna()

    if current_year is None:
        current_year = clustering_objects.get('current_year',
                                             df_pred['year'].median() if 'year' in df_pred.columns else 2024)

    X_pred_data['property_age'] = current_year - X_pred_data['yearbuilt_coal']

    features_pred = [
        clustering_objects['scaler_location'].transform(X_pred_data[['latitude', 'longitude']]) * clustering_objects['location_weight'],
        clustering_objects['scaler_sqft'].transform(X_pred_data[['livingareasqft_coal']]) * clustering_objects['sqft_weight'],
        clustering_objects['scaler_edu'].transform(X_pred_data[['pct_bachelors_plus']]) * clustering_objects['education_weight'],
        clustering_objects['scaler_age'].transform(X_pred_data[['property_age']]) * clustering_objects['age_weight']
    ]
    X_combined_pred = np.hstack(features_pred)

    df_pred.loc[X_pred_data.index, 'geo_cluster'] = clustering_objects['kmeans'].predict(X_combined_pred)
    df_pred['geo_cluster'] = df_pred['geo_cluster'].fillna(clustering_objects['default_cluster']).astype(int)

    df_pred = df_pred.merge(train_cluster_stats, on='geo_cluster', how='left')

    for col in ['cluster_avg_price', 'cluster_med_price', 'cluster_price_std']:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].fillna(df_pred[col].median())

    df_pred_clean = df_pred[available_features + [PROPERTYID_COL]].copy()

    missing_features = set(available_features) - set(df_pred_clean.columns)
    for feat in missing_features:
        df_pred_clean[feat] = 0

    df_pred_clean = df_pred_clean.dropna()

    X_pred = df_pred_clean[available_features].values
    X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=1e10, neginf=-1e10)

    y_pred = np.expm1(model.predict(X_pred))

    results = pd.DataFrame({
        PROPERTYID_COL: df_pred_clean[PROPERTYID_COL].values,
        'predicted_price': y_pred,
        'geo_cluster': df_pred_clean['geo_cluster'].values if 'geo_cluster' in df_pred_clean.columns else None
    })

    results = results.merge(
        df_pred[[PROPERTYID_COL, 'livingareasqft_coal', 'lotsizesqft_coal',
                'yearbuilt_coal', 'bedrooms_mls', 'full_baths_coal', 'latitude', 'longitude']],
        on=PROPERTYID_COL, how='left'
    )

    if Y_COL in df_pred.columns:
        actual_prices = df_pred_clean[PROPERTYID_COL].map(df_pred.set_index(PROPERTYID_COL)[Y_COL])
        results['actual_price'] = actual_prices
        results['price_error'] = actual_prices - results['predicted_price']
        results['price_error_pct'] = (results['price_error'] / actual_prices) * 100
        results['abs_pct_error'] = np.abs(results['price_error_pct'])

    return results

if __name__ == "__main__":
    DATA_PATH = "/Users/jenny.lin/BASIS_AVM_Onboarding/cate_scenario_analyses/data/inference_df_with_politics.parquet"

    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.lower()

    df = filter_extreme_properties(df, lot_to_living_threshold=3.5)
    df = collapse_to_property_level(df, decay=DECAY_FACTOR)

    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    df_train, df_test, kmeans = create_geo_clusters_no_leakage(df_train, df_test)

    results = build_single_model_no_leakage(df_train.copy(), df_test.copy(), use_cluster_features=True)

    print(f"\nTOP 15 FEATURES:")
    for idx, row in results['feature_importance'].head(15).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")