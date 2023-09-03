import logging

from langchain.chat_models import ChatOpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM")


class SummaryGeneration:
    def __init__(self, OPENAI_KEY, title, description, video_text, audio_text):
        self.llm = ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_KEY)
        self.prompt = f"""
                    You have to tell what a video is about given its main title, description, video and audio explanation in 100 simple words. Rephrase final explanation\n
                    Main title and Description has the highest priority, don't write it explicitly.\n
                    Video explanation will be list of dicts containing given keys and its priority starting with highest first : title, texts, visual_titles. If any visual_titles is different from others and main title then exclude only that from explanation and don't write about it.\n
                    Audio explanation will be a paragraph.\n

                    Main Title : {title}
                    Description : {description}
                    Video explanation : {video_text}
                    Audio Explanation : {audio_text}
                    """

    def run(self):
        logger.info("Generating explanation....")
        complete_explanation = self.llm.predict(self.prompt)
        logger.info("Final explanation generated!")

        return complete_explanation
