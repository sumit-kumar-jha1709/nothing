from predict import preprocess, predictyolo
import os
import numpy as np
from flask import Flask, redirect, url_for, request, render_template,send_from_directory, Response, jsonify, redirect
from flask_cors import CORS
from util import base64_to_pil
import cv2

app = Flask(__name__)
CORS(app)
@app.route('/', methods=['GET'] )
def index():
    #return render_template("index.html")
    return ("Flask is running")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = preprocess(img)
        pred = predictyolo(img)
        return jsonify(result=pred)
    return None

if __name__ == "__main__":
    app.run(debug=True)