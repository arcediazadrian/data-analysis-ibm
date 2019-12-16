import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, header=None)

df.columns=headers

# Format data before binning
# df = df.replace("?", np.nan)
df.replace("?", np.nan, inplace=True)
df.dropna(subset=["price"], axis=0, inplace=True)
df["price"] = df["price"].astype(int)
print(df["price"].tail(4))

# Create bins array of 4 equaly spaced numbers
bins = np.linspace(min(df["price"]), max(df["price"]), 4)

# Create bin names array
group_names = ["Low", "Medium", "High"]

# Use cut to segment and sort values into bins
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names, include_lowest=True)
print(df[["price", "price-binned"]].tail(5))

df.to_csv("./data-wrangling/binning.csv")