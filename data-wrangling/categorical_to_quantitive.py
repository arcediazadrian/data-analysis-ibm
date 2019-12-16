import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, header=None)

df.columns=headers

# One hot encoding
fuel = pd.get_dummies(df["fuel-type"])
print(fuel)
df["diesel"] = fuel["diesel"]
df["gas"] = fuel["gas"]

df.to_csv("./data-wrangling/categorical_to_quantitive.csv")