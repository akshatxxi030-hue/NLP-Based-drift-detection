import streamlit as st
import requests
import pandas as pd


st.set_page_config(
    page_title="News Drift Monitoring",
    layout="wide"
)

st.title("📊 News Drift Monitoring Dashboard")
st.write("Automated drift detection between baseline and current datasets")


API_URL = "https://nlp-based-drift-detection.onrender.com/predict"  # change if deployed


if st.button("Run Drift Analysis 🚀"):
    with st.spinner("Calculating drift..."):
        try:
            response = requests.get(API_URL)

            if response.status_code == 200:
                result = response.json()

                st.subheader("📌 Overall Drift")
                st.metric(
                    label="Drift Score",
                    value=result["overall_drift"]
                )

                st.info(f"**Drift Level:** {result['drift_level']}")
                st.warning(result["message"])

               
                st.subheader("📦 Batch-wise Drift")
                batch_df = pd.DataFrame(result["batch_drift"])
                st.line_chart(
                    batch_df.set_index("batch")["drift"]
                )

            
                st.subheader("🔑 Top TF-IDF Words")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Baseline Dataset**")
                    st.table(
                        pd.DataFrame(
                            result["top_words"]["baseline"],
                            columns=["Word", "Score"]
                        )
                    )

                with col2:
                    st.write("**Current Dataset**")
                    st.table(
                        pd.DataFrame(
                            result["top_words"]["current"],
                            columns=["Word", "Score"]
                        )
                    )

            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
