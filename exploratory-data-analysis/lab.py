import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("./clean_data.csv")


## Analizing Individual Feature Patterns using Visualization


# correlation
cor = df[["bore", "stroke", "horsepower", "compression-ratio"]].corr()
cor.to_csv("./exploratory-data-analysis/correlation.csv")

# Continuous variables

# positive linear relationship
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
plt.savefig("./exploratory-data-analysis/positive-linear-regression.png")

print(df[["engine-size", "price"]].corr())

plt.clf()

# negative linear relationship
sns.regplot(x="highway-L/100km", y="price", data=df)
plt.ylim(0,)
plt.savefig("./exploratory-data-analysis/negative-linear-regression.png")

print(df[["highway-L/100km", "price"]].corr())

plt.clf()

# weak linear relationship
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.savefig("./exploratory-data-analysis/weak-linear-regression.png")

print(df[["peak-rpm", "price"]].corr())

plt.clf()

# Catergorical variables


sns.boxenplot(x="body-style", y="price", data=df)
plt.savefig("./exploratory-data-analysis/bad-boxplot.png")

plt.clf()

sns.boxenplot(x="engine-location", y="price", data=df)
plt.savefig("./exploratory-data-analysis/good-boxplot.png")

plt.clf()

sns.boxenplot(x="drive-wheels", y="price", data=df)
plt.savefig("./exploratory-data-analysis/good-boxplot2.png")

plt.clf()



# Descriptive Statistical Analysis


# Describe
df.describe().to_csv("./exploratory-data-analysis/description_of_data.csv")
print(df.describe(include=["object"]))

# Value Counts
drive_wheels_counts = df["drive-wheels"].value_counts().to_frame()
drive_wheels_counts.rename(columns={"drive-wheels": "value_counts"}, inplace=True)
drive_wheels_counts.index.name = "drive-wheels"
print(drive_wheels_counts)

engine_loc_counts = df["engine-location"].value_counts().to_frame()
engine_loc_counts.rename(columns={"engine-location": "value_counts"}, inplace=True)
engine_loc_counts.index.name = "engine-location"
print(engine_loc_counts)

# Grouping
df["drive-wheels"].unique()

group_one = df[["drive-wheels", "body-style", "price"]]
group_one.groupby(["drive-wheels"], as_index=False).mean()
print(group_one)

group_two = df[["drive-wheels", "body-style", "price"]]
group_two = group_two.groupby(["drive-wheels", "body-style"], as_index=False).mean()
print(group_two)

group_pivot = group_two.pivot(index="drive-wheels", columns="body-style")
group_pivot = group_pivot.fillna(0) # fill missing values with 0
print(group_pivot)

# Simple heat map
plt.pcolor(group_pivot, cmap="RdBu")
plt.colorbar()
plt.savefig("./exploratory-data-analysis/simple-heatmap.png")

plt.clf()

# Verbose heatmap
fig, ax = plt.subplots()
im = ax.pcolor(group_pivot, cmap="RdBu")

row_labels = group_pivot.columns.levels[1]
col_labels = group_pivot.index

ax.set_xticks(np.arange(group_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group_pivot.shape[0]) + 0.5, minor=False)

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

plt.xticks(rotation=90)

fig.colorbar(im)
plt.savefig("./exploratory-data-analysis/custom-heatmap.png")

plt.clf()



# Correlation and Causation


# 1 = positive linear correlation | 0 = No correlation | -1 = negative linear correlation
# P-value = probability that the correlation bewtween two var is statistically significant | p < 0.05 = 95% sure that is significant
df.corr().to_csv("./exploratory-data-analysis/df-correlation.csv")

pearson_coef, p_value = stats.pearsonr(df["wheel-base"], df["price"])
print("The Pearson Correlation Coefficient of wheel-base is ", pearson_coef, " with a P-value of P=", p_value)

pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])
print("The Pearson Correlation Coefficient of horsepower is ", pearson_coef, " with a P-value of P=", p_value)

pearson_coef, p_value = stats.pearsonr(df["length"], df["price"])
print("The Pearson Correlation Coefficient of length is ", pearson_coef, " with a P-value of P=", p_value)

pearson_coef, p_value = stats.pearsonr(df["width"], df["price"])
print("The Pearson Correlation Coefficient of width is ", pearson_coef, " with a P-value of P=", p_value)

pearson_coef, p_value = stats.pearsonr(df["curb-weight"], df["price"])
print("The Pearson Correlation Coefficient of curb-weight is ", pearson_coef, " with a P-value of P=", p_value)

pearson_coef, p_value = stats.pearsonr(df["engine-size"], df["price"])
print("The Pearson Correlation Coefficient of engine-size is ", pearson_coef, " with a P-value of P=", p_value)

pearson_coef, p_value = stats.pearsonr(df["bore"], df["price"])
print("The Pearson Correlation Coefficient of bore is ", pearson_coef, " with a P-value of P=", p_value)

pearson_coef, p_value = stats.pearsonr(df["city-L/100km"], df["price"])
print("The Pearson Correlation Coefficient of city-L/100km is ", pearson_coef, " with a P-value of P=", p_value)

pearson_coef, p_value = stats.pearsonr(df["highway-L/100km"], df["price"])
print("The Pearson Correlation Coefficient of highway-L/100km is ", pearson_coef, " with a P-value of P=", p_value)



# ANOVA - Analysis of Variance



# F-test score: A larger score means there is a larger difference between the means
# P-value: How statistically significant is our calculated score
anova_group_test = df[["drive-wheels", "price"]].groupby(["drive-wheels"])

f_val, p_val = stats.f_oneway(anova_group_test.get_group("fwd")["price"], anova_group_test.get_group("rwd")["price"], anova_group_test.get_group("4wd")["price"])

f_val, p_val = stats.f_oneway(anova_group_test.get_group("fwd")["price"], anova_group_test.get_group("rwd")["price"])
print("ANOVA fwd+rwd results: F=", f_val, ", P=", p_val)

f_val, p_val = stats.f_oneway(anova_group_test.get_group("4wd")["price"], anova_group_test.get_group("rwd")["price"])
print("ANOVA 4wd+rwd results: F=", f_val, ", P=", p_val)

f_val, p_val = stats.f_oneway(anova_group_test.get_group("4wd")["price"], anova_group_test.get_group("fwd")["price"])
print("ANOVA 4wd+fwd results: F=", f_val, ", P=", p_val)






