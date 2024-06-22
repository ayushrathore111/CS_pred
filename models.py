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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

from xgboost import XGBRegressor

# Create and train the XGBoost regression model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Make predictions using the trained model
y_pred = model.predict(X_test)
y_pred_f= model.predict(X)
r2 = r2_score(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred, squared=False))
mse = mean_squared_error(Y_test, y_pred)

RF = RandomForestRegressor(max_depth=2, random_state=0)
RF.fit(X_train, Y_train)
Y_pred_RF = RF.predict(X_test)
r2_rf = r2_score(Y_test, Y_pred_RF)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred, squared=False))
mse = mean_squared_error(Y_test, y_pred)

DT = DecisionTreeRegressor(random_state=0)
DT.fit(X_train, Y_train)
Y_pred_DT = DT.predict(X_test)
r2_dtr = r2_score(Y_test, Y_pred_DT)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred, squared=False))
mse = mean_squared_error(Y_test, y_pred)

GBR =GradientBoostingRegressor(random_state=0)
GBR.fit(X_train, Y_train)
Y_pred_GBR = GBR.predict(X_test)
r2_gbr = r2_score(Y_test, Y_pred_GBR)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred, squared=False))
mse = mean_squared_error(Y_test, y_pred)


from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, Y_train)
y_pred_knn = knn_model.predict(X_test)
r2_knn = r2_score(Y_test, y_pred_knn)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred, squared=False))
mse = mean_squared_error(Y_test, y_pred)

MLP = MLPRegressor(random_state=1, max_iter=50)
MLP.fit(X_train, Y_train)
Y_pred_MLP = MLP.predict(X_test)
r2_mlp = r2_score(Y_test, Y_pred_MLP)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred, squared=False))
mse = mean_squared_error(Y_test, y_pred)


print("XGB: ",r2)
print("RF: ",r2_rf)
print("DTR:",r2_dtr)
print("GBR: ",r2_gbr)
print("KNN: ",r2_knn)
print("MLP: ",r2_mlp)
import joblib

joblib.dump(model,'./static/XGB.joblib')
# joblib.dump(knn_model,'./static/KNN.joblib')
# joblib.dump(MLP,'./static/MLP.joblib')
# joblib.dump(GBR,'./static/GBR.joblib')
# joblib.dump(DT,'./static/DTR.joblib')
# joblib.dump(RF,'./static/rf.joblib')

print("dumped")