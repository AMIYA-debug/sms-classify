from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from stored import cleaning_part, separate, conversion

app = Flask(__name__)

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
        
        first = cleaning_part(message)
        second = separate(first)
        third = conversion(second)

        
        vec = np.array(third[0]).reshape(1, -1)

        
        try:
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(vec)[0]
            pred = model.predict(vec)[0]
        except Exception:
            
            pred = model.predict(vec)

        label = 'Spam' if int(pred) == 1 else 'Not Spam'

        spam_proba = None
        if proba is not None:
            classes = getattr(model, 'classes_', None)
            
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
                
                spam_proba = float(max(proba))
            else:
                spam_proba = float(proba[idx])

        return render_template('home.html', message=message, label=label, proba=spam_proba)

    except Exception as e:
        return render_template('home.html', error=str(e))


if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0', port=5000)
