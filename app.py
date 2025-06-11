import streamlit as st
import pandas as pd
from ai_agent import analyze_data_with_agent

st.set_page_config(page_title="AI-Powered CSV Analyzer", layout="wide")
st.title("AI-Powered CSV Analyzer")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", df.head())

    # Add user prompt input
    user_prompt = st.text_input("Enter your analysis goal / prompt:", "Explore trends in the dataset")

    if st.button("Run AI Agent"):
        with st.spinner("Running AI Agent..."):
            profile_summary, viz_ideas, plot_path = analyze_data_with_agent(df, user_prompt)

        st.success("AI Agent Analysis Complete!")

        st.write("### Data Profile Summary")
        st.markdown(profile_summary)

        st.write("### Visualization Ideas & Inferences")
        st.markdown(viz_ideas)

        if plot_path:
            st.write("### Example Auto-Generated Plot")
            st.image(plot_path)

st.markdown("---")
st.caption("Built with Streamlit + LangChain + LangGraph + Gemini")
