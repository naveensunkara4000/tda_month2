import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import os

data_path = "../data/house_prices.csv"
output_dir = "../outputs/week8"
model_dir = "../models"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)
X = df[['size','bedrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Load previous model
model = joblib.load(os.path.join(model_dir,"house_model.pkl"))

# Grid search
param_grid = {'n_estimators':[50,100], 'max_depth':[None,10]}
gs = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
joblib.dump(best_model, os.path.join(model_dir,"house_model_best.pkl"))

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

with open(os.path.join(output_dir,"tuned_metrics.txt"), "w") as f:
    f.write(f"MSE after tuning: {mse}\n")

print("Week 8 tuning done! Check outputs/week8/")
