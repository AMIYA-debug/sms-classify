from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from stored import cleaning_part, separate, conversion

app = Flask(__name__)

# Try to load the sklearn model (ML-model.pkl) at startup. If it fails we keep the error
model = None
load_error = None
model_path = os.path.join(os.path.dirname(__file__), "ML-model.pkl")
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    load_error = str(e)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    message = message.strip()
    if not message:
        return render_template('home.html', error='Please enter a message to classify.')

    if model is None:
        return render_template('home.html', error=f'Model not loaded: {load_error}')

    try:
        # preprocessing pipeline from stored.py
        first = cleaning_part(message)
        second = separate(first)
        third = conversion(second)

        # conversion returns a list of vectors, take the first vector
        vec = np.array(third[0]).reshape(1, -1)

        # predict
        try:
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(vec)[0]
            pred = model.predict(vec)[0]
        except Exception:
            # fallback if the model expects different input
            pred = model.predict(vec)

        label = 'Spam' if int(pred) == 1 else 'Not Spam'

        spam_proba = None
        if proba is not None:
            classes = getattr(model, 'classes_', None)
            # try to find the index of class '1' (spam). If classes are text, try 'spam'
            idx = None
            if classes is not None:
                try:
                    idx = list(classes).index(1)
                except ValueError:
                    try:
                        idx = list(classes).index('spam')
                    except ValueError:
                        idx = None
            if idx is None:
                # if no classes_ or can't find, show the max probability
                spam_proba = float(max(proba))
            else:
                spam_proba = float(proba[idx])

        return render_template('home.html', message=message, label=label, proba=spam_proba)

    except Exception as e:
        return render_template('home.html', error=str(e))


if __name__ == '__main__':
    # Use 0.0.0.0 so the app is reachable on the network if needed
    app.run(debug=True, host='0.0.0.0', port=5000)
