import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("Advertising.csv")

print(df.head())
print(df.isna().sum())
print(df.describe())

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

plt.scatter(x = X["TV"], y = y)
plt.title("TV AD Sales")
plt.xlabel("TV AD")
plt.ylabel("Sales")
plt.show()

plt.scatter(x = X["Radio"], y = y)
plt.title("Radio AD Sales")
plt.xlabel("Radio AD")
plt.ylabel("Sales")
plt.show()

plt.scatter(x = X["Newspaper"], y = y)
plt.title("Newspaper AD Sales")
plt.xlabel("Newspaper AD")
plt.ylabel("Sales")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score :", r2_score(y_test, y_pred))
print("RMSE :", root_mean_squared_error(y_test, y_pred))

print(model.coef_)
print(model.intercept_)