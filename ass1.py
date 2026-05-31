import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("housing.csv", header=None)

print("Shape =", df.shape)
print(df.head())

df.columns = [
    "ID", "Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt",
    "Garage", "Garden", "Age", "Distance", "CrimeRate", "Price"
]

print(df.head())
print(df.shape)


df.columns = [
    "ID", "Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt",
    "Garage", "Garden", "Age", "Distance", "CrimeRate", "Price"
]

# Display first rows
df.head()

# Dataset information
df.info()
df.describe()
df.isnull().sum()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

sns.histplot(df["Price"], kde=True)
plt.show()

X = df.drop(["ID", "Price"], axis=1)
y = df["Price"]

X = np.array(X)
y = np.array(y).reshape(-1,1)

X = np.c_[np.ones(X.shape[0]), X]

beta = np.linalg.inv(X.T @ X) @ X.T @ y

print(beta)

y_pred = X @ beta

RSS = np.sum((y - y_pred)**2)
MSE = np.mean((y - y_pred)**2)

print("RSS =", RSS)
print("MSE =", MSE)

residuals = y - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.show()

plt.scatter(y, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()