import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Streamlit app title
st.title('SMS Spam Detector')

# User input
sms = st.text_area('Enter The Message')

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if (i not in stopwords.words('english')) 
                         and (i not in string.punctuation) 
                         and (len(i) != 1) 
                         and (not i.isdigit())]
    y = [ps.stem(i) for i in y]
    return ' '.join(y)

# Prediction button
if st.button('Detect'):
    trans_sms = transform_text(sms)
    vector = tfidf.transform([trans_sms])
    pred = model.predict(vector)

    if pred == 1:
        st.markdown('<h2 style="color:red">Spam</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="color:green">Not Spam</h2>', unsafe_allow_html=True)
