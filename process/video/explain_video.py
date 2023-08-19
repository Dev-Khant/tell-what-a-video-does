import cv2
import time

import torch
from pytube import YouTube
from serpapi import GoogleSearch
from transformers import AutoImageProcessor, SwiftFormerModel


class VideoExplanation:
    """
    Explanation of only video
    """

    def __init__(self, video_link, SERP_KEY):
        self.video_link = video_link
        self.SERP_KEY = SERP_KEY

        self.image_processor = AutoImageProcessor.from_pretrained(
            "MBZUAI/swiftformer-xs"
        )
        self.model = SwiftFormerModel.from_pretrained("MBZUAI/swiftformer-xs")

        self.previous_embeddings = []
        self.similarity_threshold = 0.2

    def image_info(self, path):
        """
        Get image info using SerpApi
        """

        params = {
            "engine": "google_lens",
            "url": "",
            "api_key": "93795a8ee93145e79651985b94fe655769bd217b6ca98c1128153125d97b8b49",
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        titles = sorted([res["title"] for res in results["visual_matches"]], key=len)
        return ".".join(titles)

    def check_similarity(self, img):
        """
        Calculate similarity between current and previous all considered images
        """
        inputs = self.image_processor(img, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).last_hidden_state[:, 0][0]

        output = torch.nn.functional.normalize(output.view(-1), p=2, dim=0)
        for embedding in self.previous_embeddings:
            score = torch.nn.functional.cosine_similarity(
                output, embedding, dim=0
            ).numpy()
            if score > self.similarity_threshold:
                return True

        self.previous_embeddings.append(output)

        return False

    def process_video(self):
        """
        Get images from video for explanation
        """

        # Path to save the frames
        output_path = ""

        # Time interval in seconds
        interval = 2

        frame_count = 0
        last_capture_time = time.time()

        # Create a YouTube object and get the highest resolution stream
        yt = YouTube(self.video_link)
        video_stream = yt.streams.filter(res="720p", progressive=True)[0]

        # Open the video stream with OpenCV
        cap = cv2.VideoCapture(video_stream.url)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            current_time = time.time()

            # Check if the time interval has passed
            if current_time - last_capture_time >= interval:
                last_capture_time = current_time
                frame_count += 1

                result = self.check_similarity(frame)
                if result:
                    continue

                # Save the captured frame
                image_filename = f"{output_path}frame_{frame_count:04d}.jpg"
                cv2.imwrite(image_filename, frame)
                print(f"Captured: {image_filename}")

        # Release the video capture object and close windows
        cap.release()
