import pandas as pd

df = pd.read_parquet("data/futures/um/klines/BTCUSDT_15m_mark.parquet")
df["date"] = pd.to_datetime(df["open_time"], unit="ms")
df["month"] = df["date"].dt.to_period("M")
print("Current data range:")
print(f"  Start: {df['date'].min()}")
print(f"  End:   {df['date'].max()}")
print(f"  Months: {df['month'].nunique()}")
print()
months = sorted(df["month"].unique())
print("All months:")
for m in months[:5]:
    print(f"  {m}")
print("  ...")
for m in months[-5:]:
    print(f"  {m}")
