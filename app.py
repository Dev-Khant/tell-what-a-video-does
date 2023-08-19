import streamlit as st
from process.video_audio_explanation import Explain

st.title("Video Understanding and Q&A Tool")

video_link = st.text_input("Enter Youtube Video Link")

huggingface_token = st.text_input("Enter Hugging Face Token")

serpapi_token = st.text_input("Enter SerpAPI Token")

if st.button("Explain"):
    if video_link and huggingface_token:
        get_explanation = Explain(video_link, huggingface_token)
        result_text = get_explanation.run()

        st.text_area("Processed Text", result_text, height=200)
    else:
        st.warning("Please provide both the YouTube video link and Hugging Face Token.")
