import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df = pd.read_csv("./clean_data.csv")

### Linear regression and multiple linear regression

#Single linear regression Yhat = a + bX
lm = LinearRegression()

X = df[['highway-L/100km']]
Y = df['price']

lm.fit(X,Y)

Yhat = lm.predict(X)
print(Yhat[0:5])

#intercept(a)
print(lm.intercept_)
#slope(b)
print(lm.coef_)

lm = LinearRegression()
lm.fit(df[['engine-size']], df['price'])
print(lm.intercept_)
print(lm.coef_)

# Multiple linear regression Yhat = a + b1X1 + b2X2 + b3X3

lm = LinearRegression()
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-L/100km']]

lm.fit(Z, df['price'])

print(lm.intercept_)
print(lm.coef_)

lm = LinearRegression()
lm.fit(df[['normalized-losses', 'highway-L/100km']], df['price'])
print(lm.intercept_)
print(lm.coef_)



## Model evaluation using Visualization



width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x='highway-L/100km', y='price', data=df)
print(plt.ylim(0,))
plt.savefig("./model-development/linear-regression-highway.png")

plt.clf()

plt.figure(figsize=(width, height))
sns.regplot(x='peak-rpm', y='price', data=df)
print(plt.ylim(0,))
plt.savefig("./model-development/linear-regression-peak.png")

plt.clf()

#To verify correlation
c = df[['peak-rpm', 'highway-L/100km', 'price']].corr()
print(c)

#Residual plot
plt.figure(figsize=(width, height))
sns.residplot(df['highway-L/100km'], df['price'])
plt.savefig("./model-development/residual-highway.png")

plt.clf()

#Visualizing multiple linear regression with distribution plot
lm = LinearRegression()
lm.fit(Z, df['price'])
Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['price'], hist=False, color='r', label='Actual value')
sns.distplot(Y_hat, hist=False, label='Fitted values', ax=ax1)

plt.title('Actual vs Fitted values for Price')
plt.xlabel('Price(in dollars)')
plt.ylabel('Proportion of Cars')
plt.savefig("./model-development/distribution.png")

plt.clf()



#Polynomial regression and pipelines



#Polynomial regression Yhat = a + b1X^2 + b2X^2 + b3X^3 + ...
def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15,55,100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.savefig("./model-development/polly_" + Name + ".png")
    plt.clf()


x = df['highway-L/100km']
y = df['price']

#Create and display a polynomial of 3rd order(cubic)
f = np.polyfit(x,y,3)
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'highway')

f = np.polyfit(x,y,11)
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'highway11')

#Multivariate Polynomial function
pr = PolynomialFeatures(degree=2)
Z_pr = pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape)


## Pipelines
Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)

pipe.fit(Z,y)
ypipe=pipe.predict(Z)
print(ypipe[0:4])



## Measures for In-Sample Evaluation



# We want a measure to determine how accurate the model is
# R-sqaured. How close is the data to the fitted regression line
# Mean Squarred Error(MSE). Measures the average of the squares of errors

#R-square, the higher the R-square value the better
lm.fit(X,Y)
print('The R-square is: ', lm.score(X,Y))

#MSE, the smallest the MSE the better
Yhat = lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

#With multiple linear regression
lm.fit(Z, df['price'])
print('The R-square is: ', lm.score(Z,df['price']))

y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], y_predict_multifit))

#With polynomial regression
r_squared = r2_score(y, p(x))
print('The R-square is: ', r_squared)

print('The mean square error of price and predicted value using polynomial is: ', mean_squared_error(df['price'], p(x)))



## Prediction and Decision Making



new_input=np.arange(1,100,1).reshape(-1,1)
lm.fit(X,Y)
yhat = lm.predict(new_input)
print(yhat[0:5])

plt.plot(new_input, yhat)
plt.savefig("./model-development/prediction.png")
plt.clf()



