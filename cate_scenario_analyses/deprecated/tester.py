import pandas as pd

file = r'/cate_scenario_analyses/data/inference_df.parquet'
df = pd.read_parquet(file)

print()
print(df.head())
