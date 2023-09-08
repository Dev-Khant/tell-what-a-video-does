import streamlit as st
from process.video_audio_explanation import Explain
from process.qa_bot import QA_Bot

st.title("Video Understanding and Q&A Tool")

st.sidebar.title("Q&A Chat")
user_input = st.sidebar.text_input("Ask question")

# Create session state to store app state
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "bot" not in st.session_state:
    st.session_state.bot = None

video_link = st.text_input("Enter Youtube Video Link")
openai_token = st.text_input("Enter OpenAI Token")
huggingface_token = st.text_input("Enter Hugging Face Token")
serpapi_token = st.text_input("Enter SerpAPI Token")

if st.button("Explain"):
    with st.spinner("Loading..."):
        if video_link and huggingface_token and openai_token and serpapi_token:
            get_explanation = Explain(
                video_link, openai_token, huggingface_token, serpapi_token
            )
            result_text = get_explanation.run()

            st.session_state.explanation = result_text

            # Initialize the bot if the OpenAI token is provided
            st.session_state.bot = QA_Bot(openai_token)
            st.session_state.bot.store_in_vectordb(result_text)
        else:
            st.warning("Please provide both the YouTube video link and all tokens.")

# Display the explanation and chatbot response
if st.session_state.explanation:
    st.subheader("Short Explanation")
    st.markdown(st.session_state.explanation)

# Check if the user has entered a question and bot is initialized
if user_input and st.session_state.bot:

    def retrieve_answer(question, bot):
        return "Your chatbot response here"

    # Use st.empty to update only the answer section
    answer_container = st.sidebar.empty()

    with st.spinner("Searching for an answer..."):
        response = retrieve_answer(user_input, st.session_state.bot)

    # Update the answer_container with the response
    answer_container.markdown(response)
