import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('randForest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
        For rendering results on HTML GUI
        '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    result = ''
    if output == 0:
        result = 'not diabetic.'
    else:
        result = 'diabetic.'

    return render_template('index.html', prediction_text='Based on the user input the patient is {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)