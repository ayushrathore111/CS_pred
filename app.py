import numpy as np
from flask import Flask, request, jsonify, render_template,redirect
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
knn_loaded = joblib.load('./static/KNN.joblib')
dtr_loaded = joblib.load('./static/Decision_Tree.joblib')
gbr_loaded = joblib.load('./static/Gradient_Boosting.joblib')
mlp_loaded = joblib.load('./static/MLP.joblib')
rf_loaded = joblib.load('./static/Random_Forest.joblib')
xgb_loaded = joblib.load('./static/XGB.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = []
    for x in request.form.values():
        try:
            int_features.append(int(float(x)))
        except ValueError:
            print(f"Skipping invalid value: {x}")

    print(int_features)
    mo = int_features[0]
    prediction = []
    output = 0
    if mo == 1:
        int_features = int_features[1:]
        final_features = np.array(int_features)
        prediction = knn_loaded.predict([final_features])
    elif mo == 2:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = rf_loaded.predict(final_features)
    elif mo == 3:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbr_loaded.predict(final_features)
    elif mo == 4:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = dtr_loaded.predict(final_features)
    elif mo == 5:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = mlp_loaded.predict(final_features)
        
    else:
        int_features = int_features[1:]
        final_features = np.array(int_features)
        prediction = xgb_loaded.predict(final_features.reshape(1,-1))

    output = round(prediction[0], 2)
    print(output)
    
    return render_template('post.html',prediction_text= "The predicted Compressive Strength of Rice Straw ash Based sample is {:.2f}MPa".format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
