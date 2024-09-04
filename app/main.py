from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

def load_model():
    return joblib.load('D:/playground/jupyter/customer-churn-prediction/model/churn_model.pkl')

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data_df = pd.DataFrame([data])

    prediction = model.predict(data_df)
    result = {'Churn': int(prediction[0])}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
