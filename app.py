import streamlit as st
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from newspaper import Article

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- PREPROCESSING ----------------
ps = PorterStemmer()
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fake News Detector")

# ---------------- SESSION STATE ----------------
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

# ---------------- TITLE ----------------
st.title("📰 Fake News Detection System")
st.write("Enter news text OR paste a news article URL")

# ---------------- TEXT INPUT ----------------
st.session_state.news_text = st.text_area(
    "📝 Enter News Text", value=st.session_state.news_text, height=200
)

# ---------------- URL INPUT ----------------
url_input = st.text_input("🌐 Paste News Article URL")

if url_input.strip() != "":
    try:
        article = Article(url_input)
        article.download()
        article.parse()

        if article.text.strip() == "":
            st.warning("⚠️ Unable to extract article text. Please use a direct news article URL.")
        else:
            st.session_state.news_text = article.text
            st.success("✅ News text extracted from URL")

    except:
        st.error("❌ Invalid or unsupported URL")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload a .txt file", type=["txt"])
if uploaded_file is not None:
    st.session_state.news_text = uploaded_file.read().decode("utf-8")

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🔍 Predict")
with col2:
    if st.button("🔄 Clear"):
        st.session_state.news_text = ""

# ---------------- PREDICTION ----------------
if predict_btn:
    if st.session_state.news_text.strip() == "":
        st.warning("⚠️ Please enter text or URL")
    else:
        processed_text = preprocess(st.session_state.news_text)
        data = vectorizer.transform([processed_text])

        prediction = model.predict(data)[0]
        confidence = np.max(model.predict_proba(data)) * 100

        st.divider()

        if prediction == 1:
            st.error(f"❌ FAKE NEWS ({confidence:.2f}%)")
            st.warning("🔴 Explanation: Contains misleading or sensational patterns.")
        else:
            st.success(f"✅ REAL NEWS ({confidence:.2f}%)")
            st.info("🟢 Explanation: Language resembles reliable news sources.")

        st.subheader("📊 Confidence")
        st.progress(int(confidence))

        word_count = len(st.session_state.news_text.split())
        st.write(f"🧮 Word Count: {word_count}")

        if word_count < 20:
            st.warning("⚠️ Short text — prediction may be unreliable")

# ---------------- ABOUT ----------------
with st.expander("📂 About Model"):
    st.write("""
    • Dataset: WELFake Dataset  
    • Algorithm: Logistic Regression  
    • Vectorizer: TF-IDF  
    • Features: Text, File Upload, URL Input  
    """)

st.caption("⚠️ Educational project — predictions may not be 100% accurate.")
