# app.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request
nltk.download('stopwords')

app = Flask(__name__, static_folder='static')

# Assuming the files are in the same directory as your script
fake_file_path = "Fake.csv"
true_file_path = "True.csv"

# Read CSV files
df_fake = pd.read_csv(fake_file_path)
df_true = pd.read_csv(true_file_path)

# Add labels to the data
df_fake['label'] = 0  # 0 for fake news
df_true['label'] = 1   # 1 for true news

# Concatenate fake and true datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Data preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, _, y_train, _ = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)

# Create and train the model
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input_page')
def input_page():
    return render_template('input.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/classify', methods=['POST'])
def classify():
    user_input = request.form['news_input']
    user_input_processed = preprocess_text(user_input)
    user_input_tfidf = tfidf_vectorizer.transform([user_input_processed])
    prediction = model.predict(user_input_tfidf)[0]

    result = "The news is classified as FAKE." if prediction == 0 else "The news is classified as TRUE."
    
    return render_template('output.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
