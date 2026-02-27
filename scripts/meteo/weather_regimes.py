# %%
import pandas as pd
import numpy as np

df = pd.read_csv(
    "/srv/data/mtitz/WR_LCattribution.txt", skiprows=list(np.arange(0, 100)), sep=r"\s+"
)
# %%
cols = [
    "time in h since 19790101_00",
    "YYYYMMDD_HH",
    "EOF attribution",
    "max WR index",
    "lifecycle WR index",
]
df.columns = cols

origin = pd.Timestamp("1979-01-01 00:00:00")
df["datetime"] = origin + pd.to_timedelta(
    pd.to_numeric(df["time in h since 19790101_00"], errors="coerce"), unit="h"
)

df.head()
# %%
weather_regimes = df["lifecycle WR index"]
weather_regimes.name = "weather_regime"
weather_regimes.index = df["datetime"]
weather_regimes = weather_regimes[weather_regimes.index.year == 2013]
# %%
weather_regimes.to_csv("/srv/data/mtitz/weather_regimes_2013.csv")
# %%
weather_regimes.hist()
