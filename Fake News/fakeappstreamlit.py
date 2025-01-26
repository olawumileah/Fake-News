import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load models and tokenizer
@st.cache_resource
def load_random_forest():
    with open("rf.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_lstm_cnn_model():
    return load_model("lstm_cnn_model.h5")

# Load all resources
rf_model = load_random_forest()
tokenizer = load_tokenizer()
lstm_cnn_model = load_lstm_cnn_model()

# Function for prediction
def predict(news_title, tweet_count):
    # 1. Preprocess the input text
    sequences = tokenizer.texts_to_sequences([news_title])
    padded_sequence = pad_sequences(sequences, maxlen=100, padding="post")
    
    # 2. Extract features using the LSTM-CNN model
    lstm_cnn_features = lstm_cnn_model.predict([padded_sequence, np.array([tweet_count])])
    
    # 3. Make predictions using the Random Forest model
    prediction = rf_model.predict(lstm_cnn_features)
    
    # 4. Return the result
    return "Fake News" if prediction[0] == 1 else "Real News"

# Streamlit App
st.title("Fake News Detection App")

# Add radio button navigator
page = st.sidebar.selectbox("Choose a page", ("Home", "Prediction"))

if page == "Home":
    # Home page content
    st.header("Welcome to the Fake News Detection App")
    st.write(
        """
        This app helps you classify news articles as either Fake News or Real News using a powerful machine learning model. 
        It leverages a combination of LSTM-CNN and Random Forest to process and analyze news titles along with their metadata.
        
        ### How to Use:
        1. Navigate to the Prediction page.
        2. Enter the news title and relevant tweet count.
        3. Click the Detect button to see the result.
        """
    )
    st.write("Navigate to the Prediction page using the sidebar buttons at the top left side to get started.")

elif page == "Prediction":
    # Prediction page content
    st.header("News Prediction")
    st.write("Enter the news title and tweet count to classify the news as Fake or Real.")

    # Input fields
    news_title = st.text_input("News Title:", value="Breaking news headline goes here")
    tweet_count = st.number_input("Number of Tweets:", min_value=0, step=1, value=0)

    # Prediction button
    if st.button("Detect"):
        if news_title.strip() == "":
            st.error("Please enter a news title.")
        else:
            result = predict(news_title, tweet_count)
            st.write(f"The news is classified as: {result}")

