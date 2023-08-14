# Video Understanding and Q&A Tool

This project allows you to input a YouTube video link, and it provides a comprehensive understanding of the video's content through audio transcription and image captioning. Additionally, you can ask questions and it will provide responses according to video content 🚀

## Features

👉 **Video Understanding** : The tool utilizes the Hugging Face **OpenAI Whisper** model for audio transcription, converting spoken words into textual format. It also employs image captioning techniques to extract text from images within the video.

👉 **Question & Answer** : Users can ask questions about the video's content. The tool leverages the power of **Chromadb** as a vector database to provide accurate and contextually relevant answers.

## How to Use

• Clone this repository : `git clone https://github.com/Dev-Khant/tell-what-a-video-does.git`

• Install the required dependencies : `pip install -r requirements.txt`

• Run the streamlit app : `streamlit run app.py`

## Technologies

• [Hugging Face Transformers](https://huggingface.co/) : Utilized to access the OpenAI Whisper model for audio transcription.

• [Streamlit](https://streamlit.io/) : Used to create the interactive web interface for the project.

• [Chromadb](https://www.trychroma.com/) : The vector database used for storing and retrieving Q&A information.
