from video.explain_video import VideoExplanation
from audio.transcribe_audio import GenerateTranscription


class Explain:
    def __init__(self, yt_link, HF_KEY, SERP_KEY):
        self.yt_link = yt_link
        self.HF_KEY = HF_KEY
        self.SERP_KEY = SERP_KEY

    def run(self):
        # for video
        video_part = VideoExplanation(self.yt_link, self.SERP_KEY)
        video_explanation = video_part.process_video()

        # for audio
        audio_part = GenerateTranscription(self.yt_link, self.HF_KEY)
        audio_transcription = audio_part.process_audio()
