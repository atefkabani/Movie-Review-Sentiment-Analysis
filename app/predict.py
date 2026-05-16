import argparse
import pickle
import os
import warnings

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import load_model


def parse_args():
    p = argparse.ArgumentParser(description="Movie Review Sentiment Analysis")
    p.add_argument("--review",    default="This movie was fantastic!")
    p.add_argument("--model",   default="models/gru_imdb.keras")
    p.add_argument("--tokenizer", default="models/tokenizer.pkl")
    return p.parse_args()


def main():
    args = parse_args()

   

    model = load_model(args.model)
    with open(args.tokenizer, "rb") as f:
        data = pickle.load(f)
    tokenizer   = data["tokenizer"]
    max_len = data["max_len"] 

    sequence = tokenizer.texts_to_sequences([args.review])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)[0, 0]
    sentiment = "positive" if prediction > 0.5 else "negative"
    print(f"Review: {args.review}")
    print(f"Sentiment: {sentiment} (confidence: {prediction:.2f})")



if __name__ == "__main__":
    main()