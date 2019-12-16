import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, header=None)

df.columns=headers

# Calculation to an entire column
df["city-mpg"] = 235/df["city-mpg"]
df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True)

# Check types
print(df.dtypes)
print(df[["price"]].describe())

# Change types
# First step to get rid of NaN values
df = df.replace("?", np.nan)
df.dropna(subset=["price"], axis=0, inplace=True)
df["price"] = df["price"].astype(int)

df.to_csv("./data-wrangling/data_formating.csv")