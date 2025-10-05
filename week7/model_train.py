import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

data_path = "../data/house_prices.csv"
output_dir = "../outputs/week7"
model_dir = "../models"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(data_path)
X = df[['size','bedrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with open(os.path.join(output_dir,"metrics.txt"),"w") as f:
    f.write(f"MSE: {mse}\nR2: {r2}\n")

joblib.dump(model, os.path.join(model_dir,"house_model.pkl"))

print("Week 7 ML model trained! Check outputs/week7/")
