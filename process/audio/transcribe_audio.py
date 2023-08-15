import os
import json, requests

import mimetypes
from pydub import AudioSegment


class GenerateTranscription:
    def __init__(self, HF_KEY, audio_path):
        self.HF_KEY = HF_KEY
        self.API_URL = (
            "https://api-inference.huggingface.co/models/openai/whisper-large"
        )
        self.content_type = mimetypes.guess_type(audio_path)[0]
        self.chunk_duration = 30 * 1000
        self.text = []

    def get_transcription(self, data):
        headers = {
            "Authorization": f"Bearer {self.HF_KEY}",
            "Content-Type": self.content_type,
        }
        response = requests.request("POST", self.API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))["text"]

    def segment_audio(self):
        audio = AudioSegment.from_file()

        # Calculate the total number of chunks
        total_chunks = len(audio) // self.chunk_duration

        for i in range(total_chunks):
            start_time = i * self.chunk_duration
            end_time = (i + 1) * self.chunk_duration
            chunk = audio[start_time:end_time]

            chunk.export(os.path.join("", f"chunk_{i + 1}.mp3"), format="mp3")

            with open(f"chunk_{i + 1}.mp3", "rb") as f:
                data = f.read()

            os.remove(f"chunk_{i + 1}.mp3")

            self.text.append(self.get_transcription(data))

        return self.text
