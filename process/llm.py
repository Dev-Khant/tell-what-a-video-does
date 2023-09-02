import logging

from langchain.chat_models import ChatOpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM")


class SummaryGeneration:
    def __init__(self, OPENAI_KEY, video_text, audio_text):
        self.llm = ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_KEY)
        self.video_text = video_text
        self.audio_text = audio_text
        self.prompt = """
                    You have to tell what a video is about given its video and audio explanation in 500 words.\n\n
                    Video explanation will be list of dicts containing given keys and its priority starting with highest first : title, texts, visual_titles. If any key is empty, ignore it.\n
                    Audio explanation will be a paragraph.\n\n

                    Video explanation : {video_text}\n
                    
                    Audio explanation : {audio_text}
                    """

    def run(self):
        pass
