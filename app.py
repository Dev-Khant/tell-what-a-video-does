import streamlit as st
from process.video_audio_explanation import Explain
from process.qa_bot import QA_Bot

st.title("Video Understanding and Q&A Tool")

video_link = st.text_input("Enter Youtube Video Link")

openai_token = st.text_input("Enter OpenAI Token")

huggingface_token = st.text_input("Enter Hugging Face Token")

serpapi_token = st.text_input("Enter SerpAPI Token")

if st.button("Explain"):
    with st.spinner("Loading..."):
        if video_link and huggingface_token:
            get_explanation = Explain(
                video_link, openai_token, huggingface_token, serpapi_token
            )
            result_text = get_explanation.run()

            st.subheader("Short Explanation")
            st.markdown(result_text)

            bot = QA_Bot(result_text)
            # TODO
            # take prompt from user

        else:
            st.warning("Please provide both the YouTube video link and all tokens.")
