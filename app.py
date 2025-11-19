# app.py - minimal SMS spam predictor (text box + predict button)
import streamlit as st
import pandas as pd
from pathlib import Path
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="SMS Spam Predictor", layout="centered")
st.title("📩 SMS Spam Predictor")

BASE = Path(__file__).parent
CSV_PATH = BASE / "spam.csv"
MODEL_PATH = BASE / "model.joblib"

# -------- simple text cleaning --------
def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------- load dataset or ask for upload (but do NOT display it) --------
df = None
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
else:
    uploaded = st.file_uploader("Upload spam.csv (optional, needed if no model exists)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, encoding="latin-1")
        except Exception as e:
            st.error(f"Can't read uploaded file: {e}")

# -------- build & train model if needed --------
def build_and_train(dataframe):
    # keep first two meaningful columns if dataset has extra unnamed cols
    cols = list(dataframe.columns)
    if len(cols) >= 2:
        dataframe = dataframe.iloc[:, :2]
        dataframe.columns = ["label", "text"]
    else:
        dataframe.columns = ["label", "text"]

    dataframe = dataframe.dropna(subset=["text"])
    dataframe["label"] = dataframe["label"].astype(str).str.strip().str.lower()
    dataframe["clean_text"] = dataframe["text"].apply(clean_text)

    X = dataframe["clean_text"]
    y = dataframe["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = make_pipeline(CountVectorizer(stop_words="english", ngram_range=(1,2)), MultinomialNB())
    pipe.fit(X_train, y_train)
    return pipe

model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        MODEL_PATH.unlink(missing_ok=True)
        model = None

if model is None and df is not None:
    with st.spinner("Training model (one-time)..."):
        model = build_and_train(df)
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception:
            pass

if model is None:
    st.info("Model not available. Upload a dataset or place spam.csv next to app.py to train the model.")
    # still show text box to let user try (but we can't predict)
    user_text = st.text_area("Type an SMS message here", height=150)
    if st.button("Predict"):
        st.warning("No model available to predict. Upload dataset or place spam.csv next to app.py and restart.")
    st.stop()

# -------- Prediction UI (minimal) --------
user_text = st.text_area("Type an SMS message here", height=170, placeholder="Enter message to classify as ham or spam")

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter a message to predict.")
    else:
        clean = clean_text(user_text)
        pred = model.predict([clean])[0]
        probs = model.predict_proba([clean])[0]
        ham_prob, spam_prob = float(probs[0]), float(probs[1])
        label = "SPAM" if pred == 1 else "HAM"
        # show a clear colored message
        if pred == 1:
            st.error(f"Prediction: {label}  (spam probability {spam_prob:.2f})")
        else:
            st.success(f"Prediction: {label}  (ham probability {ham_prob:.2f})")

# small helpful footer
st.caption("Model: CountVectorizer + MultinomialNB. To retrain, replace model.joblib or upload a dataset.")
