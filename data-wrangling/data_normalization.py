import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, header=None)

df.columns=headers

# # Simple feature scaling
# df["length"] = df["length"]/df["length"].max()
# print(df["length"].head(5))

# # Min-max
# df_range = df["length"].max()-df["length"].min()
# df["length"] = (df["length"]-df["length"].min())/df_range
# print(df["length"].head(5))

# Z-score
df["length"] = (df["length"]-df["length"].mean())/df["length"].std()
print(df["length"].head(5))

df.to_csv("./data-wrangling/data_normalization.csv")