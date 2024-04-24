from flask import Flask, render_template, request, flash, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
import bcrypt
from flask_mysqldb import MySQL

from flask import Flask, render_template, request
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
from sklearn.model_selection import KFold
from collections import Counter

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mydatabase'

mysql = MySQL(app)


# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')
def clean_tweets(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def remove_html(text):
    text = text.replace("\n", " ")
    pattern = re.compile('<.*?>')  # all the HTML tags
    return pattern.sub(r'', text)

def remove_email(text):
    text = re.sub(r'[\w.<>]\w+@\w+[\w.<>]', " ", text)
    return text

def remove_all_special_chars(text):
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    return text

def replace_mult_spaces(text):
    text = text.replace("&quot", "")
    pattern = re.compile(' +')
    text = pattern.sub(r' ', text)
    text = text.strip()
    return text

def replace_chars(text, pattern):
    pattern = re.compile(pattern)
    text = pattern.sub(r'', text)
    return text

def _get_unique(elems):
    if type(elems[0]) == list:
        corpus = [item for sublist in elems for item in sublist]
    else:
        corpus = elems
    elems, freqs = zip(*Counter(corpus).most_common())
    return list(elems)

def convert_categorical_label_to_int(labels):
    if type(labels[0]) == list:
        uniq_labels = _get_unique(labels)
    else:
        uniq_labels = _get_unique(labels)

    label_to_id = {}
    if type(labels[0]) == list:
        label_to_id = {w: i+1 for i, w in enumerate(uniq_labels)}
    else:
        label_to_id = {w: i for i, w in enumerate(uniq_labels)}

    new_labels = []
    if type(labels[0]) == list:
        for i in labels:
            new_labels.append([label_to_id[j] for j in i])
    else:
        new_labels = [label_to_id[j] for j in labels]

    return new_labels, label_to_id

def load_and_preprocess_data():
    df = pd.read_csv('Sarcasm data.txt', sep='\t', header=None, usecols=[0, 1])
    df.columns = ['text', 'category']

    # Drop rows with missing values and empty text
    df = df.dropna()
    df = df[df['text'] != '']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(df['text']):
        break

    train_df = df.iloc[train_index]

    train_df['text'] = train_df['text'].apply(clean_tweets)
    train_df['text'] = train_df['text'].apply(remove_html)
    train_df['text'] = train_df['text'].apply(remove_email)
    train_df['text'] = train_df['text'].apply(remove_all_special_chars)
    train_df['text'] = train_df['text'].apply(replace_mult_spaces)
    train_df['text'] = train_df['text'].apply(lambda x: replace_chars(x, '[()!@&;]'))

    train_df['category'], label2idx = convert_categorical_label_to_int(train_df['category'].values)
    return train_df

train_preprocessed = load_and_preprocess_data()
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectorizer.fit(train_preprocessed['text'])
train_tfidf = tfidf_vectorizer.transform(train_preprocessed['text'])

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_preprocessed['text'])

def predict_sarcasm(text):
    max_len = 100  
    text = clean_tweets(text)
    text = remove_html(text)
    text = remove_email(text)
    text = remove_all_special_chars(text)
    text = replace_mult_spaces(text)
    text = replace_chars(text, '[()!@&;]')
    
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(sequence)
    return prediction[0][0]

class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self, field):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users where email=%s", (field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError('Email Already Taken')

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/register', methods=['GET','POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # store data into database
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (name,email,password) VALUES (%s,%s,%s)", (name, email, hashed_password))
        mysql.connection.commit()
        cursor.close()

        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            session['user_id'] = user[0]
            return redirect(url_for('abusiveContentDetection'))
        else:
            flash("Login failed. Please check your email and password")
            return redirect(url_for('login'))

    return render_template('login.html', form=form)



@app.route('/dashboard')
def dashboard():

    if 'user_id' in session:
        user_id = session['user_id']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users where id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            return render_template('dashboard.html', user=user)
            
    return redirect(url_for('login'))

@app.route("/abusiveContentDetection")
def abusiveContentDetection():
    if 'user_id' in session:
        user_id = session['user_id']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            return render_template("abusiveContentDetection.html", user=user)
    return render_template("abusiveContentDetection.html")  # Render template without user info if not logged in

@app.route("/sarcasmContentDetection", methods=['GET', 'POST'])
def sarcasmContentDetection():
    input_sentence = None
    prediction = None

    if 'user_id' in session:
        user_id = session['user_id']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()

        if user and request.method == 'POST':
            input_sentence = request.form['text']
            prediction = predict_sarcasm(input_sentence)
            binary_prediction = 1 if prediction > 0.5 else 0

            if binary_prediction == 1:
                prediction = "Sarcasm"
            else:
                prediction = "Not Sarcasm"
                    
    return render_template("sarcasmContentDetection.html", user=user, input_sentence=input_sentence, prediction=prediction)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))


@app.route('/contact')
def contact():
    if 'user_id' in session:
        user_id = session['user_id']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            return render_template("contact.html", user=user)
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
