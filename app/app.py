import os
import pickle
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import load_model


# ----------------------------- Page config -----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ----------------------------- Loaders ---------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_sentiment_model(model_path: str):
    """Load and cache the trained Keras model."""
    return load_model(model_path)


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer(tokenizer_path: str):
    """Load and cache the tokenizer + max_len."""
    with open(tokenizer_path, "rb") as f:
        data = pickle.load(f)
    return data["tokenizer"], data["max_len"]


def predict_sentiment(review: str, model, tokenizer, max_len: int):
    """Return (sentiment_label, confidence_score, raw_probability)."""
    sequence = tokenizer.texts_to_sequences([review])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    prob = float(model.predict(padded, verbose=0)[0, 0])
    label = "positive" if prob > 0.5 else "negative"
    confidence = prob if label == "positive" else 1 - prob
    return label, confidence, prob


# ----------------------------- Sidebar ---------------------------------
st.sidebar.title("⚙️ Settings")
model_path = st.sidebar.text_input(
    "Model path",
    value="models/gru_imdb.keras",
    help="Path to the trained Keras model file.",
)
tokenizer_path = st.sidebar.text_input(
    "Tokenizer path",
    value="models/tokenizer.pkl",
    help="Path to the pickled tokenizer (must contain 'tokenizer' and 'max_len').",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\n"
    "This app uses a GRU-based deep-learning model trained on the IMDB dataset "
    "to classify a movie review as **positive** or **negative**."
)


# ----------------------------- Main UI ---------------------------------
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown(
    "Type or paste a movie review below and the model will predict whether the "
    "sentiment is **positive** or **negative**, along with a confidence score."
)

# Example reviews users can quickly try
examples = {
    "— Choose an example —": "",
    "Positive example": "This movie was absolutely fantastic! The acting was superb and the story kept me hooked from start to finish.",
    "Negative example": "What a complete waste of time. The plot was boring, the acting was terrible, and I couldn't wait for it to end.",
    "Mixed example": "The visuals were stunning but the storyline felt weak and the pacing was off.",
}
choice = st.selectbox("Try an example:", list(examples.keys()))

default_review = examples[choice] if examples[choice] else "This movie was fantastic!"
review = st.text_area(
    "Your review:",
    value=default_review,
    height=160,
    placeholder="Write a movie review here...",
)

col1, col2 = st.columns([1, 1])
analyze_clicked = col1.button("🔍 Analyze sentiment", type="primary", use_container_width=True)
clear_clicked = col2.button("🧹 Clear", use_container_width=True)

if clear_clicked:
    st.rerun()


# ----------------------------- Prediction ------------------------------
if analyze_clicked:
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        # Validate paths
        if not os.path.exists(model_path):
            st.error(f"Model file not found: `{model_path}`")
        elif not os.path.exists(tokenizer_path):
            st.error(f"Tokenizer file not found: `{tokenizer_path}`")
        else:
            try:
                model = load_sentiment_model(model_path)
                tokenizer, max_len = load_tokenizer(tokenizer_path)

                with st.spinner("Analyzing review..."):
                    label, confidence, prob = predict_sentiment(
                        review, model, tokenizer, max_len
                    )

                st.markdown("### Result")

                if label == "positive":
                    st.success(f"😊 **Positive** review")
                else:
                    st.error(f"😞 **Negative** review")

                # Confidence metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Sentiment", label.capitalize())
                m2.metric("Confidence", f"{confidence * 100:.1f}%")
                m3.metric("Raw probability", f"{prob:.3f}")

                # Confidence bar
                st.markdown("**Confidence**")
                st.progress(float(confidence))

                # Probability visualization (positive vs negative)
                st.markdown("**Probability distribution**")
                st.bar_chart(
                    {
                        "Sentiment": {
                            "Negative": 1 - prob,
                            "Positive": prob,
                        }
                    }
                )

                with st.expander("Show review text"):
                    st.write(review)

            except Exception as e:
                st.exception(e)


st.markdown("---")
st.caption("Built with Streamlit · Powered by TensorFlow/Keras GRU model trained on IMDB")