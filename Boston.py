import pandas as pd
import numpy as np

df = pd.read_csv("housing.csv")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from matplotlib import pyplot as plt
import seaborn as sb

df["medv"] = np.log1p(df["medv"])
x = df.drop(["medv", "chas", "nox", "rm", "ptratio"], axis = 1)
y = df["medv"]
y = y.astype(int)

# Feature Selection
best_features = SelectKBest(score_func = chi2, k = "all")
fit = best_features.fit(x, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x.columns)
features_scores = pd.concat([df_columns, df_scores], axis = 1)
features_scores.columns = ["Attributes", "Score"]
# print(features_scores)

lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42,
                                                    test_size=0.1)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("Linear Regression:", mean_squared_error(y_test, y_pred))

'''
OUTPUT:
Linear Regression: 0.15631827291996633
'''