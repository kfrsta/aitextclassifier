import string
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from catboost import CatBoostClassifier
from nltk.tokenize import word_tokenize
import nltk

app = Flask(__name__)

nltk.download('punkt')

model = joblib.load('./padlav2.pkl')
glove_embeddings = joblib.load('./glove_embeddings.pkl')


def preprcess_text(x: str):
    x = x.translate(str.maketrans("", "", string.punctuation))
    x = x.lower()
    tokens = word_tokenize(x)
    return tokens


def essay_to_vector(sentence, glove_embeddings):
    essay_vector = np.mean([glove_embeddings[word]
                           for word in sentence if word in glove_embeddings], axis=0)
    return essay_vector


def preprocess_input(input: str):
    input = input.replace('\n', '')
    input = preprcess_text(input)
    vector = essay_to_vector(input, glove_embeddings)
    return vector


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.headers['Content-Type'] == 'application/json':
            data = request.json['text']
        else:
            data = request.form['text']

        preprocessed_text = preprocess_input(data)

        prediction = model.predict_proba(preprocessed_text)

        output = "".join(
            str(np.int32(prediction[1]*100)))+"% AI generated text"

        return render_template('index.html', prediction_result=f'Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_result=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
