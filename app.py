import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(
        "models/final_fraud_detection_model.pkl"
    )

artifact = load_model()

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning Based Fraud Detection System")

st.metric("Selected Model", artifact["model_name"])
st.metric("Decision Threshold", f"{threshold:.4f}")

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Transaction Data")
    st.dataframe(data.head())

    missing = set(features) - set(data.columns)

    if missing:
        st.error(f"Missing columns: {list(missing)}")

    else:
        X = data[features]

        probabilities = model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results = data.copy()

        results["Fraud_Probability"] = probabilities
        results["Prediction"] = predictions
        results["Status"] = results["Prediction"].map({
            0: "Legitimate",
            1: "Potential Fraud"
        })

        st.subheader("Prediction Results")

        st.dataframe(results)

        fraud_count = int(predictions.sum())
        legitimate_count = len(predictions) - fraud_count

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(results))
        col2.metric("Potential Fraud", fraud_count)
        col3.metric("Legitimate", legitimate_count)

        st.download_button(
            "Download Prediction Results",
            results.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )
