import pandas as pd

df = pd.read_csv("basin_metrics_log_trn.csv")

nse_mean = df["nse"].mean()
rmse_mean = df["rmse"].mean()
mae_mean = df["mae"].mean()

print(f"Number of basins: {len(df)}")
print(f"NSE:  {nse_mean:.4f}")
print(f"RMSE: {rmse_mean:.4f}")
print(f"MAE:  {mae_mean:.4f}")
