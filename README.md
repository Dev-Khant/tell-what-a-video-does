# Video Understanding and Q&A Tool

This project allows you to input a YouTube video link, and it provides a comprehensive understanding of the video's content through audio transcription and image captioning. **LLM** is used to combine audio and video context. Additionally, you can ask questions and it will provide responses according to video content 🚀

## Features ✨

👉 **Video Understanding** : The tool utilizes the **Transformer** model for audio transcription, converting spoken words into textual format. It also employs image captioning techniques to extract text from images within the video. **Image embeddings** are also used to compare images and only use images unqiue for extracting info. Video and Audio are processed **parallelly**.

👉 **Question & Answer** : Users can ask questions about the video's content. The tool leverages the power of **Chromadb** as a vector database to provide accurate and contextually relevant answers.

## How to Use ⚙️

• Clone this repository : `git clone https://github.com/Dev-Khant/tell-what-a-video-does.git`

• Install the required dependencies : `pip install -r requirements.txt`

• Run the streamlit app : `streamlit run app.py`

## Technical 🖥️

• [Hugging Face](https://huggingface.co/) : Utilized to access the OpenAI Whisper model for audio transcription.

• [SerpApi](https://serpapi.com/) : Used it to access Google Lens API for getting image information.

• [Streamlit](https://streamlit.io/) : Used to create the interactive web interface for the project.

• [Chromadb](https://www.trychroma.com/) : The vector database used for storing and retrieving Q&A information.

## Work in Progress 🚧

1. Add Chromadb for Q&A. 
