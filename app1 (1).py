import streamlit as st
import numpy as np
import pickle
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mental Health Sentiment Analysis",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background-color: #ffffff;
}

h1 {
    color: #333333;
}

textarea {
    background-color: #f5f5f5 !important;
    color: #000000 !important;
    border-radius: 10px;
}

div.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 220px;
    font-size: 16px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #ff2b2b;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🧠 Sentimental analysis for mental health")
st.write("Analyze emotions from text using RNN (LSTM)")

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("mental_health_sentiment_rnn.h5")

# ---------------- LOAD TOKENIZER ----------------
tokenizer = pickle.load(open("tokenizer.pkl","rb"))

# ---------------- LOAD LABEL ENCODER ----------------
encoder = pickle.load(open("label_encoder.pkl","rb"))

# ---------------- STOPWORDS ----------------
stop_words = set(stopwords.words("english"))

# ---------------- TEXT CLEANING ----------------
def clean_text(text):

    text = text.lower()
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'[^a-zA-Z ]','',text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# ---------------- INPUT ----------------
user_input = st.text_area("Enter your text here")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:

        cleaned = clean_text(user_input)

        seq = tokenizer.texts_to_sequences([cleaned])

        padded = pad_sequences(seq, maxlen=100)

        prediction = model.predict(padded)

        label = np.argmax(prediction)

        result = encoder.inverse_transform([label])[0]

        emoji = {
            "Depression":"😔",
            "Anxiety":"😟",
            "Stress":"😣",
            "Normal":"😊",
            "Suicidal":"🚨"
        }

        st.success(f"Predicted Mental State: {result} {emoji.get(result,'')}")