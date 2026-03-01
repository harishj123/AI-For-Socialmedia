import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv("dataset.csv")

X = data["text"]
y = data["risk"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Build Model Pipeline
# -------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier(n_estimators=100))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🚨 SilentShield - AI Distress Detection System")
st.write("AI-Based Early Emotional Distress & Risk Prediction for Social Safety")

user_input = st.text_area("Enter a message to analyze emotional risk:")

if st.button("Analyze Risk"):
    if user_input.strip() != "":
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]

        risk_levels = model.classes_
        risk_scores = dict(zip(risk_levels, probabilities))

        st.subheader("Prediction Result")
        st.write("Predicted Risk Level:", prediction)

        if prediction == "Low":
            st.success("Low Risk – No immediate threat detected.")
        elif prediction == "Medium":
            st.warning("Medium Risk – Signs of emotional stress detected.")
        else:
            st.error("High Risk – Potential danger detected. Consider seeking help.")

        st.subheader("Risk Probability Scores")
        st.write(risk_scores)
    else:
        st.warning("Please enter a message.")

# -------------------------
# Show Model Performance
# -------------------------
st.sidebar.header("Model Information")
st.sidebar.write("Model Accuracy:", round(accuracy * 100, 2), "%")

st.sidebar.write("Classes:", list(model.classes_))
