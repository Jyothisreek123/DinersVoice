import streamlit as st
import joblib
import re
import nltk

from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


# Page configuration
st.set_page_config(
    page_title="Diner'sVoice",
    page_icon="🍴",
    layout="centered"
)


# Load model
saved_data = joblib.load("models/nlp_model.pkl")

model = saved_data["model"]
tfidf = saved_data["vectorizer"]


# NLP preprocessing
sp = stopwords.words('english')
sn = SnowballStemmer('english')
tk = TweetTokenizer()


def preprocess_text(tweets):

    tweets = re.sub('[^a-zA-Z0-9]', ' ', tweets)
    tweets = tk.tokenize(tweets)
    tweets = [w for w in tweets if len(w) >= 3]
    tweets = [sn.stem(w.lower()) for w in tweets]
    tweets = [w for w in tweets if w not in sp]

    return ' '.join(tweets)


# Website styling
st.markdown("""
<style>

/* Main background */
.stApp {
    # background-color: #F8F7F2;
    # color: #26352D;
    background-color: #000000;
}

/* Top header */
header[data-testid="stHeader"] {
    # background-color: #F8F7F2;
    background-color: #000000;
}

/* Main container */
.block-container {
    max-width: 750px;
    padding-top: 3rem;
}

/* Main heading */
.app-heading {
    font-size: 40px;
    font-weight: 750;
    # color: #183D32;
    color: #FFFFFF;
    text-align: center;
    margin-bottom: 8px;
}

/* Heading accent */
.app-heading span {
    color: #E8793E;
}

/* Subtitle */
.app-subtitle {
    font-size: 16px;
    # color: #718078;
    color: #CCCCCC;
    text-align: center;
    margin-bottom: 30px;
}

/* Divider */
hr {
    border-color: #E5E2D8;
}

/* Text area */
.stTextArea textarea {
    # background-color: #FFFFFF;
    background-color: #1C1C1C;
    # color: #26352D;
    color: #FFFFFF;
    # border: 1px solid #D9DED6;
    border: 1px solid #444444;
    border-radius: 12px;
}

/* Text area focus */
.stTextArea textarea:focus {
    border: 1px solid #E8793E;
    box-shadow: 0 0 0 1px #E8793E;
}

/* Button */
.stButton button {
    background-color: #245B47;
    # color: #FFFFFF;
    color: white;
    border: none;
    border-radius: 9px;
    height: 45px;
    width: 100%;
    font-weight: 600;
}

.stButton button:hover {
    background-color: #183D32;
    color: #FFFFFF;
    border: none;
}

/* Footer */
.footer {
    text-align: center;
    # color: #89948D;
    color: #AAAAAA;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="app-heading">🍴 Diner’s<span>Voice</span></div>

<div class="app-subtitle">
    Restaurant Review Sentiment Analysis
</div>
""", unsafe_allow_html=True)

st.divider()


# Main section
st.subheader("Analyze Your Review")

st.write(
    "Enter your restaurant experience to identify "
    "whether the review expresses a positive or negative sentiment."
)


# Text input
user_text = st.text_area(
    "Restaurant Review",
    placeholder="Write your food and dining experience here...",
    height=130
)


# Prediction
if st.button("Analyze Review"):

    if user_text.strip():

        cleaned_text = preprocess_text(user_text)

        if cleaned_text.strip():

            text_vector = tfidf.transform([cleaned_text])
            prediction = model.predict(text_vector)

            if prediction[0] == 0:
                # st.error("Negative Sentiment 😞")
                st.error("Negative Review ")
            else:
                # st.success("Positive Sentiment 😊")
                st.success("Positive Review ")

        else:
            st.warning("Please enter meaningful text.")

    else:
        st.warning("Please enter a restaurant review.")


# Footer
st.divider()

st.markdown("""
<div class="footer">
    Diner'sVoice | Restaurant Review Sentiment Analysis
</div>
""", unsafe_allow_html=True)






