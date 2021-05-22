from flask import Flask, request, redirect, url_for
from flask.templating import render_template
from fastai import learner

app = Flask(__name__)
model = learner.load_learner("export.pkl")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/", methods = ['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save("image")
        pred,pred_idx,probs = model.predict("image")
        return render_template("index.html", prediction_text = f"Prediction: {pred}", probability_text = f"Probability: {probs[pred_idx]:.04f}")
    return render_template("index.html", prediction_text = "There was an error")