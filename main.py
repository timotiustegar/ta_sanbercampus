import os
import schedule
import time
import threading
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import ngrams
from nltk.corpus import stopwords
from wordcloud import WordCloud
import spacy
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# secrets
dataset_path = st.secrets.path_configuration.dataset_path
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# ------------------------------------------ Variabel -------------------------------------------------
text=""
label=""
bahasa=""
data = pd.DataFrame()
stopword_en = stopwords.words('english')
stopword_id = stopwords.words('indonesian')
button_clicked = False

# ------------------------------------------- Fungsi --------------------------------------------------
def hapus_semua_file():
    # Ambil daftar file dalam folder
    file_list = os.listdir(dataset_path)

    # Hapus setiap file dalam folder
    for file_name in file_list:
        file_path = os.path.join(dataset_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("Semua file telah dihapus.")

def jadwal_hapus_file():
    # Jadwalkan penghapusan semua file setiap pukul 00.00
    schedule.every().day.at("00:00").do(hapus_semua_file)

    # Jalankan scheduler sekali
    while True:
        schedule.run_pending()
        time.sleep(1)

@st.cache_data
def import_data(file):
    with open(dataset_path + file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    result = pd.read_csv(dataset_path + file.name)
    return result

@st.cache_data
def label_count(df, label):
    result = df[label].value_counts().reset_index()
    return result

@st.cache_data
def text_preprocessing(text, lang):
    # Mengubah menjadi lowecase
    text = text.lower()
    # Menghilangkan angka
    text = re.sub(r"\d+", r'', text)
    # Menghilangkan tanda baca
    text = text.translate(str.maketrans("","",string.punctuation))
    # Menghilangkan tautan
    pola_tautan = re.compile(r'https?://\S+|www\.\S+')
    text = pola_tautan.sub(r'', text)
    # Menghilangkan whitespace
    text = text.strip()
    # Tokenize
    word_list = word_tokenize(text)
    # List stopwords
    stopword_en = stopwords.words('english')
    stopword_id = stopwords.words('indonesian')
    # Hapus stopword
    if lang == "Indonesia":
        list_no_stopwords  = [word for word in word_list if word not in stopword_id]
    else:
        list_no_stopwords  = [word for word in word_list if word not in stopword_en]
    text = ' '.join(list_no_stopwords)

    return text

@st.cache_data
def make_wordcloud(df, text, lang):
    temp_cleaned = df.copy()
    temp_cleaned["cleaned"] = df[text].apply(lambda x : text_preprocessing(x, lang))
    tokens = temp_cleaned["cleaned"].apply(lambda x: word_tokenize(str(x)))
    temp_tokenized = [word for sentence in tokens for word in sentence]
    wcloud = WordCloud(width=1600, height=1600, max_font_size=200).generate(' '.join(temp_tokenized))
    fig = plt.figure(figsize=(12, 10))
    # Tampilkan Gambar Wordcloud pada figure
    plt.imshow(wcloud, interpolation="bilinear")
    plt.axis("off")
    return fig
    

@st.cache_resource
def choose_model(chosen_models):
    if button_clicked:
        if "Support Vector Classification" in chosen_models:
            st.text("SVC")
        if "Naive Bayes" in chosen_models:
            st.text("NB")
        if "Decision Tree" in chosen_models:
            st.text("DT")
        if "Random Forest" in chosen_models:
            st.text("RF")

@st.cache_data
def train_model(df, lang, label, text):
    temp_cleaned = df.copy()
    temp_cleaned["cleaned"] = df[text].apply(lambda x : text_preprocessing(x, lang))
    vectorizer = TfidfVectorizer()
    vectorized = vectorizer.fit_transform(temp_cleaned['cleaned']).toarray()
    # Pembagian data menjadi data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(vectorized, temp_cleaned[label], test_size=0.2, random_state=18)
    # SVC
    param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]}
    # Buat grid search dengan parameter
    grid_nb = GridSearchCV(MultinomialNB(), param_grid, cv=5)
    # Fit grid ke model
    grid_nb.fit(X_train,y_train)
    print(grid_nb.best_params_)
    # Predict dengan hasil grid search
    grid_nb_predict = grid_nb.predict(X_test)
    # Cetak akurasi
    return accuracy_score(y_test, grid_nb_predict)

# Start thread untuk menjalankan scheduler
thread = threading.Thread(target=jadwal_hapus_file)
thread.start()

# ---------------------------------------------- APP ----------------------------------------------------
# --- Upload Dataset ---
# Uploader
uploaded_file = st.file_uploader("Unggah dataset anda disini", type=["csv"])
# Simpan file
if uploaded_file is not None:
    # Baca dataset yg di upload
    data=import_data(uploaded_file)

# --- Tabs ---
eda, sentiment, train = st.tabs(["EDA", "Sentiment Analysis", "Predict"])
# Tab EDA
with eda:
    if data.empty:
        st.text("Silahkan import dataset anda")
    else:
        # Judul
        st.header(f"Dataset {uploaded_file.name}")
        st.dataframe(data=data, use_container_width=True)
        n_baris, n_kolom = data.shape
        infobaris_col, infokolom_col = st.columns([2, 10])
        with infobaris_col:
            st.text(f"Banyak Baris : {n_baris}")
        with infokolom_col:
            st.text(f"Banyak Kolom : {n_kolom}")
        st.header("Konfigurasi Dataset")
        text_col, label_col, lang_col = st.columns(3)        
        with text_col:
            pilih = st.selectbox(
                "Pilih field text pada dataset anda",
                data.columns,
                index=None,
                placeholder="Ex. tweet, text, etc"
            )
            text=pilih
            st.write('You selected:', pilih)

        with label_col:
            pilih = st.selectbox(
                "Pilih field label pada dataset anda",
                data.columns,
                index=None,
                placeholder="Ex. label, sentiment, etc",
            )
            label=pilih
            st.write('You selected:', pilih)
        with lang_col:
            pilih = st.selectbox(
                "Pilih bahasa yang digunakan pada dataset anda",
                ["Indonesia", "Inggris"],
                index=None,
            )
            bahasa=pilih
            st.write('You selected:', pilih)
        if label is None or text is None or bahasa is None:
            st.text("Pilih field text dan label dan bahasa pada dataset anda")
        else:
            # EDA Distribusi Label
            data_count = label_count(data, label)
            st.bar_chart(data=data_count, x=label, y="count", height=500)
            # Wordcloud
            fig = make_wordcloud(data, text, bahasa)
            st.pyplot(fig)          
with sentiment:
    if label is None or text is None or bahasa is None or data.empty:
        st.text("Selesaikan proses import dataset dan exploratory data analysis terlebih dahulu")
    else:
        st.header("Pilih model dan latih dataset anda")
        # select_col, train_col = st.columns([11, 1])
        # with select_col:
        #     models = st.multiselect(
        #     'Pilih model yang ingin coba dilatih',
        #     ['Support Vector Classification', 'Naive Bayes', 'Decision Tree', 'Random Forest'])
        if st.button(label="Latih"):
            st.write(train_model(data, bahasa, label, text))
        
            