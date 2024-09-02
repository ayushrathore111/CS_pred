import numpy as np
from flask import Flask, request, jsonify, render_template,redirect
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
f_ar = joblib.load(r'static/model_flexural_AdaBoost.joblib')
f_en = joblib.load(r'static/model_flexural_ElasticNet.joblib')
f_etr = joblib.load(r'static/model_flexural_Extra Trees.joblib')
f_hub = joblib.load(r'static/model_flexural_Huber.joblib')
f_lt = joblib.load(r'static/model_flexural_LightGBM.joblib')
f_r = joblib.load(r'static/model_flexural_Ridge.joblib')
s_ar = joblib.load(r'static/model_split_tensile_AdaBoost.joblib')
s_en = joblib.load(r'static/model_split_tensile_ElasticNet.joblib')
s_etr = joblib.load(r'static/model_split_tensile_Extra Trees.joblib')
s_hub = joblib.load(r'static/model_split_tensile_Huber.joblib')
s_lt = joblib.load(r'static/model_split_tensile_LightGBM.joblib')
s_r = joblib.load(r'static/model_split_tensile_Ridge.joblib')



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
        prediction = f_ar.predict([final_features])
    elif mo == 2:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = f_en.predict(final_features)
    elif mo == 3:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = f_etr.predict(final_features)
    elif mo == 4:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = f_hub.predict(final_features)
    elif mo == 5:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = f_lt.predict(final_features)
        
    elif mo==6:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        # prediction = f_r.predict(final_features.reshape(1,-1))
        prediction = f_r.predict(final_features)
    elif mo==7:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = s_ar.predict(final_features)
    elif mo==8:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = s_en.predict(final_features)
    elif mo==9:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = s_etr.predict(final_features)
    elif mo==10:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = s_hub.predict(final_features)
    elif mo==11:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = s_lt.predict(final_features)
    elif mo==12:
        int_features = int_features[1:]
        final_features = [np.array(int_features)]
        prediction = s_r.predict(final_features)

    output = round(prediction[0], 2)
    print(output)
    
    return render_template('post.html',prediction_text= "The predicted Strength of Rice Straw ash Based sample is {:.2f}MPa".format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
