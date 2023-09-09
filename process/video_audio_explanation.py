from multiprocessing import Pool

from process.llm import SummaryGeneration
from process.video.explain_video import VideoExplanation
from process.audio.transcribe_audio import GenerateTranscription


class Explain:
    def __init__(self, yt_link, OPENAI_KEY, HF_KEY, SERP_KEY):
        self.yt_link = yt_link
        self.OPENAI_KEY = OPENAI_KEY
        self.HF_KEY = HF_KEY
        self.SERP_KEY = SERP_KEY

    def process_video(self):
        """Video processing"""
        video_part = VideoExplanation(self.yt_link, self.SERP_KEY)
        video_explanation, title, description = video_part.process_video()
        return video_explanation, title, description

    def process_audio(self):
        """Audio processing"""
        audio_part = GenerateTranscription(self.yt_link, self.HF_KEY)
        audio_transcription = audio_part.process_audio()
        return audio_transcription

    def run(self):
        """Synchronously process video and audio"""

        with Pool(processes=2) as pool:
            # apply the functions to the inputs in parallel
            video_part = pool.apply_async(self.process_video)
            audio_part = pool.apply_async(self.process_audio)

            # get the results from the worker processes
            video_explanation, title, description = video_part.get()
            audio_transcription = audio_part.get()

        # merge both video and audio
        llm_part = SummaryGeneration(
            self.OPENAI_KEY, title, description, video_explanation, audio_transcription
        )
        complete_explanation = llm_part.run()

        return complete_explanation
