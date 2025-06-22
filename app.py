from flask import Flask, request, render_template
import nltk 
from nltk import classify
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, download
from nltk.stem.wordnet import WordNetLemmatizer
import re, string, joblib

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load the model
model = joblib.load('model.pkl')

app = Flask(__name__)

# Preprocessing function
def remove_noise(tweet_tokens):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub(r"http\S+|www\.\S+|@\w+", "", token)  # Remove URLs and mentions
        token = re.sub(r"[^A-Za-z0-9]", "", token)  # Keep only alphanumeric

        if not token:
            continue

        pos = 'n'
        if tag.startswith('VB'):
            pos = 'v'
        elif tag.startswith('JJ'):
            pos = 'a'

        token = lemmatizer.lemmatize(token, pos)

        if token.lower() not in stop_words and token.lower() not in string.punctuation:
            cleaned_tokens.append(token.lower())

    return " ".join(cleaned_tokens)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['message']  # Match the textarea name in HTML
    tokens = word_tokenize(input_text)
    cleaned_text = remove_noise(tokens)
    features = {word: True for word in cleaned_text.split()}
    prediction = model.classify(features)
    return render_template('result.html', prediction=prediction, input_text=input_text)

# Run app
if __name__ == '__main__':
    app.run(debug=True)