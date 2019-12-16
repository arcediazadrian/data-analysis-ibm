import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, header=None)

df.columns=headers

# Drop NaN values in certian rows, axis = 0->row, 1->column, inplace=True to replace the value of the df
df = df.replace("?", np.nan)
df.dropna(subset=["price"], axis=0, inplace=True)

# Replace NaN with the mean
df['normalized-losses'] = df['normalized-losses'].astype(float)
mean = df['normalized-losses'].mean()
df["normalized-losses"].replace(np.nan, mean, inplace=True)

df.to_csv("./data-wrangling/missing_data.csv")