import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.svm              import SVR
from sklearn.ensemble         import GradientBoostingRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
data = pd.read_excel('./static/Book1.xlsx')

data = data.dropna()
# Separate the data into input X and output Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

from xgboost import XGBRegressor

# Create and train the XGBoost regression model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Make predictions using the trained model
y_pred = model.predict(X_test)
y_pred_f= model.predict(X)
dc1 = pd.DataFrame(y_pred)
# dc1.to_excel('xgb.xlsx')
# Evaluate the model
r2 = r2_score(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred, squared=False))
mse = mean_squared_error(Y_test, y_pred)

print(r2)
import joblib

joblib.dump(model,'./static/XGB.joblib')
print("dumped")