import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import joblib

file_path = "static\Book2.xlsx"
df = pd.read_excel(file_path)

# Display the first few rows of the dataframe
print(df.head())


X = df[['OPC', 'NFA', 'CA', 'RSA Content', 'SP', 'Water', 'Curing Days']]
y = df['Flex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(),
    'Huber': HuberRegressor(max_iter=2000),
    'AdaBoost': AdaBoostRegressor(),
    'Extra Trees': ExtraTreesRegressor(),
    'LightGBM': LGBMRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on the test set
    score = r2_score(y_test, y_pred)  # Calculate R² score
    results[name] = score
    print(f"{name} R² score: {score:.4f}")
    joblib.dump(model,f"model_flexural_{name}.joblib")
# Print out the results for each model
print("\nModel Performance:")

for model_name, r2 in results.items():
    print(f"{model_name}: R² = {r2:.4f}")

