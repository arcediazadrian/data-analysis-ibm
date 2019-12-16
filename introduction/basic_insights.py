import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, header=None)

df.columns=headers

print(df.dtypes)

print(df.describe()) # Describes only numerical values
print(df.describe(include="all")) # Describes all columns even objects(string)
print(df[['length','compression-ratio']].describe())

print(df.info) # Grabs top 30 rows and last 30 rows

