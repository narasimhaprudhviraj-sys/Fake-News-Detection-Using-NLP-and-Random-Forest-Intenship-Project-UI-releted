# ==========================================
# FAKE NEWS DETECTION USING NLP
# Random Forest + TF-IDF + Streamlit UI
# ==========================================

import os
import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Fake News Detection System")
st.caption("NLP • TF-IDF Feature Extraction • Random Forest Classifier")

st.info(
    "This application estimates whether an article resembles the fake or real "
    "news patterns in its training dataset. It does not independently verify "
    "the truth of a news story."
)

# ==========================================
# LOAD STOPWORDS
# ==========================================

@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

STOP_WORDS = load_stopwords()

# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(words)

# ==========================================
# LOAD AND TRAIN MODEL
# ==========================================

@st.cache_resource
def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(current_dir, "Fake_200.csv")
    true_path = os.path.join(current_dir, "True_200.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            "Could not find Fake_200.csv and True_200.csv. "
            "Place both files in the same folder as app.py."
        )

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    if "text" not in fake.columns or "text" not in true.columns:
        raise ValueError("Both CSV files must contain a 'text' column.")

    # Use up to 200 rows from each source file.
    fake_small = fake.sample(
        n=min(200, len(fake)), random_state=42
    ).copy()
    true_small = true.sample(
        n=min(200, len(true)), random_state=42
    ).copy()

    fake_small["label"] = 0
    true_small["label"] = 1

    data = pd.concat([fake_small, true_small], ignore_index=True)
    data = data.dropna(subset=["text"])
    data["text"] = data["text"].astype(str)
    data = data[data["text"].str.strip().ne("")]
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    if data["label"].nunique() < 2 or len(data) < 10:
        raise ValueError("Not enough valid data to train the model.")

    data["cleaned_text"] = data["text"].apply(clean_text)
    X = data["cleaned_text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Save the same reduced dataset format as the original script.
    output_path = os.path.join(current_dir, "fake_news_400.csv")
    data.drop(columns=["cleaned_text"]).to_csv(output_path, index=False)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "records": len(data),
        "train_records": len(X_train),
        "test_records": len(X_test),
    }

# ==========================================
# INITIALIZE MODEL
# ==========================================

with st.spinner("Loading dataset and training Random Forest model..."):
    try:
        results = train_model()
    except Exception as error:
        st.error(f"Could not initialize the model: {error}")
        st.stop()

model = results["model"]
vectorizer = results["vectorizer"]

st.success("Model is ready for predictions.")

# ==========================================
# SIDEBAR: MODEL OVERVIEW
# ==========================================

with st.sidebar:
    st.header("📊 Model Overview")
    st.metric("Dataset records", results["records"])
    st.metric("Training records", results["train_records"])
    st.metric("Testing records", results["test_records"])
    st.metric("Test accuracy", f"{results['accuracy'] * 100:.2f}%")
    st.caption("Accuracy is measured on the held-out test split.")

    st.divider()
    st.subheader("How it works")
    st.write("1. Load fake and real news datasets")
    st.write("2. Clean article text")
    st.write("3. Convert text to TF-IDF features")
    st.write("4. Train Random Forest")
    st.write("5. Predict the submitted article")

# ==========================================
# SAMPLE ARTICLES
# ==========================================

samples = {
    "Choose a sample article...": "",
    "Sample 1 — Government programme": (
        "Government Announces New Programme to Support Small Businesses\n\n"
        "WASHINGTON — The government announced a new programme on Tuesday "
        "aimed at helping small businesses expand their operations and create "
        "jobs. Officials said the initiative would provide financial assistance "
        "and training opportunities to eligible businesses. The programme is "
        "expected to focus on improving access to funding and supporting local "
        "economic development. Further details will be released in the coming weeks."
    ),
    "Sample 2 — Unverified election claim (fictional)": (
        "SHOCKING: Politicians Secretly Agree to Cancel All Future Elections\n\n"
        "A fictional report circulating online claims that several senior "
        "politicians have secretly agreed to cancel all future elections. "
        "Anonymous sources allegedly revealed that the decision was made during "
        "a private meeting, but no official documents or credible evidence have "
        "been provided. This example was created for model testing and is not a "
        "real news report."
    ),
    "Sample 3 — Economic policy debate": (
        "Lawmakers Debate New Economic Policy Ahead of National Vote\n\n"
        "WASHINGTON — Lawmakers debated a proposed economic policy on Wednesday "
        "as political parties prepared for the upcoming national vote. Supporters "
        "said the proposal could encourage investment and create employment "
        "opportunities, while critics questioned its potential cost and long-term "
        "effects. The proposal is expected to face further discussion before a "
        "final decision is made."
    ),
}

st.subheader("🔎 Analyze a News Article")
selected_sample = st.selectbox("Try a sample article (optional)", list(samples.keys()))

default_text = samples[selected_sample]
headline = st.text_input(
    "News headline (optional)",
    placeholder="Enter the news headline...",
)
article = st.text_area(
    "News article",
    value=default_text,
    height=260,
    placeholder="Paste the news article here...",
)

analyze = st.button("🔍 Analyze News", type="primary", use_container_width=True)

if analyze:
    combined_text = "\n".join(part for part in [headline.strip(), article.strip()] if part)

    if not combined_text:
        st.warning("Please enter a headline or article before analyzing.")
    else:
        cleaned = clean_text(combined_text)
        if not cleaned.strip():
            st.warning("The input has no usable text after cleaning. Please try a longer article.")
        else:
            vector = vectorizer.transform([cleaned])
            prediction = int(model.predict(vector)[0])
            probabilities = model.predict_proba(vector)[0]

            # Read probability positions by class label instead of assuming order.
            class_positions = {
                int(class_label): index
                for index, class_label in enumerate(model.classes_)
            }
            fake_score = float(probabilities[class_positions[0]] * 100)
            real_score = float(probabilities[class_positions[1]] * 100)
            confidence = fake_score if prediction == 0 else real_score

            st.divider()
            st.subheader("📋 Analysis Report")

            if prediction == 0:
                st.error("🚨 Model Prediction: FAKE NEWS")
            else:
                st.success("✅ Model Prediction: REAL NEWS")

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Prediction confidence", f"{confidence:.2f}%")
            metric2.metric("Fake-news score", f"{fake_score:.2f}%")
            metric3.metric("Real-news score", f"{real_score:.2f}%")

            st.write("**Probability comparison**")
            probability_df = pd.DataFrame(
                {"Probability (%)": [fake_score, real_score]},
                index=["Fake News", "Real News"],
            )
            st.bar_chart(probability_df)

            with st.expander("View submitted text"):
                st.write(combined_text)

            st.warning(
                "This is a model prediction, not a fact-check. A high score "
                "does not prove an article is false or true. Verify claims using "
                "reliable independent sources."
            )

# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.divider()
st.subheader("📈 Model Performance")

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
m2.metric("Total records", results["records"])
m3.metric("Features limit", "3,000")

with st.expander("View classification report"):
    report_df = pd.DataFrame(results["report"]).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

with st.expander("View confusion matrix"):
    fig, ax = plt.subplots()
    display = ConfusionMatrixDisplay(
        confusion_matrix=results["confusion_matrix"],
        display_labels=["Fake", "Real"],
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Fake News Detection — Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

st.caption(
    "Academic demonstration only. Results depend on the dataset and training "
    "method. The model may learn source or writing-style patterns rather than truth."
)
