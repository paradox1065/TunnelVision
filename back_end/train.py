# trains model and saves it (model.pkl)
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

model = RandomForestRegressor()

df = pd.read_csv("fake_clean_data.csv")
X = df.drop("cost_estimate", axis = 1)
y = df["cost_estimate"]

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=3)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(scores)
print(scores.mean())

print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)