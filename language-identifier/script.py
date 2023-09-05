from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the language classification model (e.g., modelensemble.pkl)
with open('modelensemble.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def classify_language():
    if request.method == 'POST':
        # Get the input text from the form
        input_text = request.form['text_input']
        text = re.sub(r'[^a-zA-Z]', ' ', input_text)
        text = text.lower()

    # Tokenization (split text into words)
        words = text.split()

    # Initialize a Porter Stemmer for word stemming
        ps = PorterStemmer()

    # Remove stopwords and apply stemming
        words = [ps.stem(word) for word in words if word not in stopwords.words('english')]

    # Join the processed words back into a single string
        processed_text = ' '.join(words)
        with open('countvectorizer.pkl', 'rb') as file:
            cv = pickle.load(file)

        processed_text=cv.transform([processed_text]).toarray()

        # Perform language classification using the model
        prediction = ensemble_model.predict(processed_text)[0]
        language_labels = ['Arabic', 'Chinese', 'Dutch', 'English', 'Estonian', 'French',
                           'Hindi', 'Indonesian', 'Japanese', 'Korean', 'Latin', 'Persian',
                           'Portuguese', 'Pushto', 'Romanian', 'Russian', 'Spanish', 'Swedish',
                           'Tamil', 'Thai', 'Turkish', 'Urdu']

        predicted_language = language_labels[prediction]

        return render_template('idk.html', input_text=input_text, predicted_language=predicted_language)

    return render_template('idk.html', input_text=None, predicted_language=None)

if __name__ == '__main__':
    app.run(debug=True)
