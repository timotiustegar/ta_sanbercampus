import os
import schedule
import time
import threading
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import ngrams
from nltk.corpus import stopwords
from wordcloud import WordCloud
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
import pickle
import nltk

nltk.download("popular")
# secrets
dataset_path = st.secrets.path_configuration.dataset_path
model_path = st.secrets.path_configuration.model_path
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# ------------------------------------------ Variabel -------------------------------------------------
text=""
label=""
bahasa=""
choose_model=""
data = pd.DataFrame()
stopword_en = stopwords.words('english')
stopword_id = stopwords.words('indonesian')
# ------------------------------------------- Fungsi --------------------------------------------------
def hapus_semua_file():
    # Ambil daftar file dalam folder dataset
    dataset_list = os.listdir(dataset_path)
    # Hapus setiap file dalam folder dataset
    for file_name in dataset_list:
        file_path = os.path.join(dataset_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Ambil daftar file dalam folder model
    model_list = os.listdir(model_path)
    # Hapus setiap file dalam folder model
    for file_name in model_list:
        file_path = os.path.join(model_path, file_name)
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
def text_preprocessing(text, lang, name, isTrain=0):
    # Mengubah menjadi lowecase
    text = text.lower()
    # Menghilangkan karakter selain huruf
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Menghilangkan tag akun
    text = re.sub(r'@\w+', '', text)
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
    # Hapus tergantung karakter
    length_bound = [kata for kata in text.split() if len(kata) > 3 and len(kata) < 10]
    text = ' '.join(length_bound)
    # Tokenize
    word_list = word_tokenize(text)
    if (isTrain):
        list_no_commons  = [word for word in word_list if word not in isTrain]
        text = ' '.join(list_no_commons)
    return text

@st.cache_data
def make_wordcloud(df, text, lang, name):
    temp_cleaned = df.copy()
    temp_cleaned["cleaned"] = df[text].apply(lambda x : text_preprocessing(x, lang, name))
    tokens = temp_cleaned["cleaned"].apply(lambda x: word_tokenize(str(x)))
    temp_tokenized = [word for sentence in tokens for word in sentence]
    wcloud = WordCloud(width=1600, height=1600, max_font_size=200).generate(' '.join(temp_tokenized))
    fig = plt.figure(figsize=(12, 10))
    # Tampilkan Gambar Wordcloud pada figure
    plt.imshow(wcloud, interpolation="bilinear")
    plt.axis("off")
    return fig

@st.cache_data
def word_frequency(data, text, lang, name):
    temp = pd.DataFrame()
    temp["cleaned"] = data[text].apply(lambda x : text_preprocessing(x, lang, name))
    temp['token'] = temp["cleaned"].apply(lambda x: word_tokenize(str(x)))
    tokens = [word for sentence in temp['token'] for word in sentence] #data flatening
    token_freq = FreqDist(tokens)
    most_common_words = token_freq.most_common(20)
    return most_common_words

@st.cache_data
def ngram_dist(data, text, lang, name, n_grams = 2):
    temp = pd.DataFrame()
    temp["cleaned"] = data[text].apply(lambda x : text_preprocessing(x, lang, name))
    tokens = temp["cleaned"].apply(lambda x: word_tokenize(str(x)))
    _ = [word for sentence in tokens for word in sentence]
    result = ngrams(_,n_grams)
    token_freq = FreqDist(result)
    most_ngram = token_freq.most_common(20)
    return most_ngram

@st.cache_data
def train_model(df, lang, label, text, name):
    temp = df.copy()
    temp_sampled = pd.DataFrame()
    # ----------------------------- Sampling ---------------------------------------
    label_class = []
    for i in temp[label].unique():
        label_class.append(temp[temp[label]==i].shape[0])
    if min(label_class) >= 2500:
        for i in temp[label].unique():
            data_sampled = temp[temp[label]==i].sample(2500)
            temp_sampled = pd.concat([temp_sampled, data_sampled])
    else:
        for i in temp[label].unique():
            data_sampled = temp[temp[label]==i].sample(min(label_class))
            temp_sampled = pd.concat([temp_sampled, data_sampled])
    # ----------------------------- Freq & Clean ---------------------------------------
    most_common_words = word_frequency(data, text, bahasa, name)
    most_common_words = [word for word, _ in most_common_words]
    temp_sampled["cleaned"] = temp_sampled[text].apply(lambda x : text_preprocessing(x, lang, name, most_common_words))
    # ----------------------------- Grid Search ---------------------------------------
    vectorizer = TfidfVectorizer(max_features=500)
    vectorized = vectorizer.fit_transform(temp_sampled['cleaned']).toarray()
    X_train, X_test, y_train, y_test = train_test_split(vectorized, temp_sampled[label], test_size=0.2, random_state=18)
    # --------------- SVC -----------------
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    grid_svc = GridSearchCV(SVC(), param_grid, cv=2)
    grid_svc.fit(X_train,y_train)
    # --------------- NB -------------------
    param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]}
    grid_nb = GridSearchCV(MultinomialNB(), param_grid, cv=5)
    grid_nb.fit(X_train,y_train)
    # --------------- DT -------------------
    param_grid = { 'criterion':['gini','entropy'], 'splitter': ['best', 'random']}
    grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid_tree.fit(X_train,y_train)
    # --------------- RF -------------------
    param_grid = { 'criterion':['gini','entropy'],'max_depth': [3, 4, 5, 6, 7, 8, 9 , 10], 'n_estimators': [100, 200, 300, 400, 500]}
    grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, cv=2)
    grid_forest.fit(X_train,y_train)
    # -------------------------------- Latih Model ----------------------------------------
    vectorizer_5000 = TfidfVectorizer(max_features=5000)
    vectorizer_model = vectorizer_5000.fit(temp_sampled['cleaned'])
    with open(model_path+f"vector_{name.replace('.csv', '')}.pkl", 'wb') as file:
        pickle.dump(vectorizer_model, file)
    vectorized_model = vectorizer_model.transform(temp_sampled['cleaned']).toarray()
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(vectorized_model, temp_sampled[label], test_size=0.2, random_state=18)
    # --------------- SVC --------------------
    model_svc = SVC(kernel=grid_svc.best_params_["kernel"], gamma=grid_svc.best_params_["gamma"], C=grid_svc.best_params_["C"])
    model_svc.fit(Xm_train,ym_train)
    with open(model_path+f"svc_{name.replace('.csv', '')}.pkl", 'wb') as file:
        pickle.dump(model_svc, file)
    svc_prediction = model_svc.predict(Xm_test)
    svc_acc = accuracy_score(ym_test, svc_prediction)
    svc_pred = model_svc.predict(Xm_test)
    svc_con = confusion_matrix(ym_test,svc_pred)
    svc_report = classification_report(y_test,svc_pred, output_dict=True)
    # --------------- NB --------------------
    model_nb = MultinomialNB(alpha=grid_nb.best_params_["alpha"])
    model_nb.fit(Xm_train,ym_train)
    with open(model_path+f"nb_{name.replace('.csv', '')}.pkl", 'wb') as file:
        pickle.dump(model_nb, file)
    nb_prediction = model_nb.predict(Xm_test)
    nb_acc = accuracy_score(ym_test, nb_prediction)
    nb_pred = model_nb.predict(Xm_test)
    nb_con = confusion_matrix(ym_test,nb_pred)
    nb_report = classification_report(y_test,nb_pred, output_dict=True)
    # --------------- DT ----------------------
    model_tree = DecisionTreeClassifier(criterion=grid_tree.best_params_["criterion"], splitter=grid_tree.best_params_["splitter"])
    model_tree.fit(Xm_train,ym_train)
    with open(model_path+f"tree_{name.replace('.csv', '')}.pkl", 'wb') as file:
        pickle.dump(model_tree, file)
    tree_prediction = model_tree.predict(Xm_test)
    tree_acc = accuracy_score(ym_test, tree_prediction)
    tree_pred = model_tree.predict(Xm_test)
    tree_con = confusion_matrix(ym_test,tree_pred)
    tree_report = classification_report(y_test,tree_pred, output_dict=True)
    # --------------- RF ----------------------
    model_forest = RandomForestClassifier(criterion=grid_forest.best_params_["criterion"], max_depth=grid_forest.best_params_["max_depth"], n_estimators=grid_forest.best_params_["n_estimators"])
    model_forest = model_forest.fit(Xm_train,ym_train)
    with open(model_path+f"forest_{name.replace('.csv', '')}.pkl", 'wb') as file:
        pickle.dump(model_forest, file)
    forest_prediction = model_forest.predict(Xm_test)
    forest_acc = accuracy_score(ym_test, forest_prediction)
    forest_pred = model_forest.predict(Xm_test)
    forest_con = confusion_matrix(ym_test,forest_pred)
    forest_report = classification_report(y_test,forest_pred, output_dict=True)
    all_acc = {
        "svc":{ "acc": svc_acc, "con": svc_con, "report": svc_report },
        "nb":{ "acc": nb_acc, "con": nb_con, "report": nb_report },
        "tree":{ "acc": tree_acc, "con": tree_con, "report": tree_report },
        "forest":{ "acc": forest_acc, "con": forest_con, "report": forest_report },
    }
    return all_acc

def make_conmatrix(con, name):
    fig, ax = plt.subplots(ncols=1, nrows=1)
    sns.heatmap(con, annot=True, cmap='gist_earth_r', ax=ax)
    # Pengaturan title dan label
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    return fig    

@st.cache_data
def predict_model(model_name, line, name):
    if model_name == "Support Vector Classifier":
        model = "svc"
    elif model_name == "Naive Baye":
        model = "nb"
    elif model_name == "Decision Tree":
        model = "tree"
    else:
        model = 'forest'
    with open(model_path+f"vector_{name.replace('.csv', '')}.pkl", 'rb') as file:
        # Memuat objek dari file menggunakan pickle.load()
        loaded_vector = pickle.load(file)
    with open(model_path+f"{model}_{name.replace('.csv', '')}.pkl", 'rb') as file:
        # Memuat objek dari file menggunakan pickle.load()
        loaded_model = pickle.load(file)
    vectorized = loaded_vector.transform([line]).toarray()
    predict = loaded_model.predict(vectorized)
    return predict

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
    dataset_name=uploaded_file.name


# --- Tabs ---
eda, sentiment, train = st.tabs(["EDA", "Sentiment Analysis", "Predict"])
# Tab EDA
with eda:
    if data.empty:
        st.text("Silahkan import dataset anda")
    else:
        # Judul
        st.header(f"Dataset {dataset_name}")
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
            st.header("Distribusi Label")
            data_count = label_count(data, label)
            st.bar_chart(data=data_count, x=label, y="count", height=500)
            # Wordcloud
            wordcloud_col, wordanalys_col = st.columns([6, 6])
            with wordcloud_col:
                st.header("Wordcloud")
                fig = make_wordcloud(data, text, bahasa, dataset_name)
                st.pyplot(fig)        
            with wordanalys_col:
                st.header("Kata paling sering muncul")
                most_common = word_frequency(data, text, bahasa, dataset_name)
                common_df = pd.DataFrame(most_common, columns=['Kata Umum', 'Frekuensi'])
                st.dataframe(common_df, use_container_width=True)
                st.header("Kombinasi kata paling sering muncul")
                ngram2_col, ngram3_col = st.columns([11, 13])
                with ngram2_col:
                    st.subheader("N Grams 2")
                    ngram_2 = ngram_dist(data, text, bahasa, dataset_name, 2)
                    ngram2_df = pd.DataFrame(ngram_2, columns=['Kombinasi Kata', 'Frekuensi'])
                    st.dataframe(ngram2_df, use_container_width=True)
                with ngram3_col:
                    st.subheader("N Grams 3")
                    ngram_3 = ngram_dist(data, text, bahasa, dataset_name, 3)
                    ngram3_df = pd.DataFrame(ngram_3, columns=['Kombinasi Kata', 'Frekuensi'])
                    st.dataframe(ngram3_df, use_container_width=True)
with sentiment:
    if label is None or text is None or bahasa is None or data.empty:
        st.text("Selesaikan proses import dataset dan exploratory data analysis terlebih dahulu")
    else:
        st.header("Latih dataset anda dengan model yang ada")
        if st.button(label="Latih"):
            model_trained = train_model(data, bahasa, label, text, dataset_name)
            for key, val in model_trained.items():
                infomodel_col, con_col, empty_col = st.columns([4, 4, 4])
                with infomodel_col:
                    if key =="svc":
                        st.subheader("Support Vector Classifier")
                    elif key == "nb":
                        st.subheader("Naive Bayes")
                    elif key == "tree":
                        st.subheader("Decision Tree")
                    else:
                        st.subheader("Random Forest")
                    st.text(f"Akurasi model : {val['acc']}")
                    st.text("Hasil performa klasifikasi :")
                    data_report = pd.DataFrame(val["report"]).transpose()
                    st.dataframe(data_report)
                with con_col:
                    st.pyplot(make_conmatrix(val["con"], dataset_name))
with train:
    selectmodel_col, empty_col = st.columns([3, 9])
    with selectmodel_col:
        pilih_model = st.selectbox(
            "Pilih field text pada dataset anda",
            ["Support Vector Classifier", "Naive Bayes", "Decision Tree", "Random Forest"],
            index=None,
        )
    kalimat = st.text_area("Kalimat yang ingin dianalisis")
    choose_model=pilih_model
    if st.button("Prediksi"):
        result = predict_model(pilih_model, kalimat, dataset_name)
        st.text(f"Hasil analisis : {result[0]}")
