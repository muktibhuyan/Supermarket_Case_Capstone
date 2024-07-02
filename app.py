import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open(r"C:\Users\ASUS\Desktop\CAPSTONE PROJECT\Group2 capstone\ML-MODEL-DEPLOYMENT-USING-FLASK-main - xgb\model.pkl", "rb"))

@flask_app.route("/") 
def Home():
    return render_template("index1.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        return render_template("index1.html", prediction_text = "The predicted engagement response is {}".format(prediction))
   

if __name__ == "__main__":
    flask_app.run(debug=True)