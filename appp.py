import streamlit as st 
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Preprocessing function
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

model=joblib.load("model.pkl")
vector=joblib.load("vectorizer.pkl")
expected_columns= joblib.load("column.pkl")
st.title("📩 Email/SMS Spam Detection - by Sapna ✨ ")
st.markdown("Provide your sms-")

text = st.text_area("Enter message here...")
if st.button("Predict"):
    raw_input = {
        'text':text
        }
    # Step 1: Preprocess
    transformed = transform_text(text)

    # Step 2: Vectorize text
    text_vector = vector.transform([transformed]).toarray()

    # Step 3: Extract extra feature(s)
    num_characters = len(text)
    extra_feature = np.array([[num_characters]])

    # Step 4: Concatenate
    final_input = np.concatenate([text_vector, extra_feature], axis=1)

    # Step 5: Predict
    prediction = model.predict(final_input)[0]
    
    
    if prediction == 1:
        st.error(f"❗ Spam Message Detected ")
    else:
        st.success(f"✅ Not Spam (Ham)")