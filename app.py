from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([features])
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = "High Credit Risk (Loan Rejected)"
    else:
        result = "Low Credit Risk (Loan Approved)"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
