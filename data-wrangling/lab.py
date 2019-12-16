import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(url, header=None)

df.columns=headers



## Deal with missing data



# Convert ? to NaN
df.replace("?", np.nan, inplace=True)

# Identifying missing values
missing_data = df.isnull()
print(missing_data.head(5))

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# Replace with mean
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized losses: ", avg_norm_loss)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

avg_bore = df["bore"].astype("float").mean(axis=0)
df["bore"].replace(np.nan, avg_bore, inplace=True)

avg_stroke = df["stroke"].astype("float").mean(axis=0)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)

avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)
df["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace=True)

# Replace with frecuency
num_of_doors_most_common_value = df["num-of-doors"].value_counts().idxmax() # get the most common type
df["num-of-doors"].replace(np.nan, num_of_doors_most_common_value, inplace=True)

# Drop values
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True) # reset index because we droped values



## Data Format



# Check format
types = df.dtypes
print(types)

# Convert to wanted format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
newtypes = df.dtypes
print(newtypes)



## Data standardization



# Converting to metric system
df["city-mpg"] = 235/df["city-mpg"]
df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True)

df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns={"highway-mpg": "highway-L/100km"}, inplace=True)

# Data normalization
# By Simple feature scaling
df["length"] = df["length"]/df["length"].max()  
df["width"] = df["width"]/df["width"].max()  
df["height"] = df["height"]/df["height"].max()

# Binning
df["horsepower"] = df["horsepower"].astype(int, copy=True)

# # plot to see
# plt.pyplot.hist(df["horsepower"])
# plt.pyplot.xlabel("horsepower")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("horsepower bins")
# plt.pyplot.show() # or plt.pyplot.savefig("horsepowergraph.png")

# back to binnig
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ["Low", "Medium", "High"]
df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)

# # plot binned distribution
# pyplot.bar(group_names, df["horsepower-binned"].value_counts())
# plt.pyplot.xlabel("horsepower")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("horsepower bins")
# plt.pyplot.show()

# # plot with bins directly
# plt.pyplot.hist(df["horsepower"], bins=3)
# plt.pyplot.xlabel("horsepower")
# plt.pyplot.ylabel("count")
# plt.pyplot.title("horsepower bins")
# plt.pyplot.show()

# Indicator variables (dummy variables)

fuel_indicator_var = pd.get_dummies(df["fuel-type"])
fuel_indicator_var.rename(columns={"gas": "fuel-type-gas", "diesel": "fuel-type-diesel"}, inplace=True)
df = pd.concat([df, fuel_indicator_var], axis=1)
df.drop("fuel-type", axis=1, inplace=True)

aspiration_indicator_var = pd.get_dummies(df["aspiration"])
aspiration_indicator_var.rename(columns={"std": "aspiration-std", "turbo": "aspiration-turbo"}, inplace=True)
df = pd.concat([df, aspiration_indicator_var], axis=1)
df.drop("aspiration", axis=1, inplace=True)

df.to_csv("./data-wrangling/lab_clean_data.csv")


