import string
#import pickle
import gc

from flask import Flask, flash, request, render_template, redirect
from flask_sitemap import Sitemap

#from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download("wordnet")
nltk.download("stopwords")

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


app = Flask(__name__)

ext = Sitemap(app=app)

#### CONFIGS ####
app.secret_key = '####### super secret key ######'
app.config['SESSION_TYPE'] = 'memcached'
## CONFIGS END ##

df_catalog = pd.read_csv("pg_catalog.csv.zip", delimiter=",", compression="infer")
df_catalog.drop(columns=["Language", "Issued", "Type", "Subjects", "LoCC", "Bookshelves"], inplace=True)

df_preprocessed = pd.read_pickle("preprocessed_data.pkl.zip", compression="infer")
df_preprocessed.drop(columns=["a", "b"], inplace=True)
df_preprocessed.sort_values(by=["similar"], ascending=False, inplace=True)
df_preprocessed.reset_index(inplace=True, drop=True)

d = {}
for text,title,authors in df_catalog.itertuples(name=None,index=False):
    d[str(text)] = (title, authors)
del df_catalog

df_preprocessed["title_a"] = df_preprocessed.apply(
    lambda row: d[row["id_a"]][0], axis=1
)

df_preprocessed["title_b"] = df_preprocessed.apply(
    lambda row: d[row["id_b"]][0], axis=1
)

df_preprocessed["authors_a"] = df_preprocessed.apply(
    lambda row: d[row["id_a"]][1], axis=1
)

df_preprocessed["authors_b"] = df_preprocessed.apply(
    lambda row: d[row["id_b"]][1], axis=1
)
del d

#model = load_model("model.h5")
#model.summary()
#with open ("tokenizer.pkl", "rb") as h:
#    tokenizer = pickle.load(h)


ALLOWED_EXTENSIONS = ("txt")

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

tbl = str.maketrans("","",string.punctuation+string.digits)
lm = nltk.wordnet.WordNetLemmatizer()
stoppers = stopwords.words("english")

def preprocess_text (text: str) -> str:
	text = [ w.lower() for w in text.translate(tbl).split() ]
	return " ".join([ lm.lemmatize(w) for w in text if w not in stoppers and w not in ENGLISH_STOP_WORDS ])

vectorizer = TfidfVectorizer(input="content", use_idf=True)

def get_dfidf_similarity_estimate (book_pair: tuple) -> float:
    X = vectorizer.fit_transform(book_pair)
    X = linear_kernel(X)[0][1]
    return X

def make_sequences (tokenizer: Tokenizer, books: np.ndarray) -> np.ndarray:
	books = tokenizer.texts_to_sequences(books)
	books = pad_sequences(books,
		maxlen=4000,
		padding="post")
	return books

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file1" not in request.files or "file2" not in request.files:
            flash("no files")
            return redirect(request.url)
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        
        if file1.filename == "" or file2.filename == "":
            flash("No two selected files")
            return redirect(request.url)
        if file1 and allowed_file(file1.filename) and file2\
         and allowed_file(file2.filename):
            try:
                a = file1.read().decode("UTF-8")
                b = file2.read().decode("UTF-8")
            except UnicodeDecodeError:
                flash("One or more files not utf-8 decodable!")
                return redirect(request.url)
            a = preprocess_text(a)
            b = preprocess_text(b)
            sim = get_dfidf_similarity_estimate((a, b))
            #a = np.array([a]); b = np.array([b])
            #a = make_sequences(tokenizer=tokenizer, books=a)
            #b = make_sequences(tokenizer=tokenizer, books=b)
            #sim2 = "I think the files are similar."\
            #    if model.predict([a, b], batch_size=8)[0][0] >= 0.50000\
            #    else "I don't think the files are similar..?"
            gc.collect()
            sim2 = "I think the files are similar."\
                if sim >= 0.30000\
                else "I don't think the files are similar..?"
            flash("Files sent succesfully! Similarity is: {:0.2f}".format(sim))
            flash("{}".format(sim2))
        else:
            flash("Only txt files please :)")
        return redirect(request.url)
            
    htmlstuffs = df_preprocessed.to_html(classes="data")
    return render_template("index.html", title="Home", tables=[htmlstuffs], titles=df_preprocessed.columns.values)

@app.route("/details", methods=["GET"])
def details():
    flash("Find the project description below.")
    return render_template("details.html", title="Details")

@ext.register_generator
def index():
    yield "index", {}


