import cv2
import time
import requests
import base64
import logging

import torch
from pytube import YouTube
from serpapi import GoogleSearch
from transformers import AutoImageProcessor, SwiftFormerModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Video")


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
        self.img_results = []

    def upload_img(self, frame):
        """
        Upload image to Imgur
        """

        # Set API endpoint and headers
        url = "https://api.imgur.com/3/image"
        headers = {"Authorization": "Client-ID c17d9d047012640"}

        # Convert the frame to JPEG format
        _, buffer = cv2.imencode(".jpg", frame)

        # Convert the image data to a base64 encoded string
        base64_data = base64.b64encode(buffer).decode("utf-8")

        # Upload image to Imgur and get URL
        response = requests.post(url, headers=headers, data={"image": base64_data})
        url = response.json()["data"]["link"]
        return url

    def image_info(self, img_url):
        """
        Get image info using SerpApi
        """

        # Set params for google lens
        params = {
            "engine": "google_lens",
            "url": img_url,
            "api_key": self.SERP_KEY,
        }

        # Do google search
        search = GoogleSearch(params)
        results = search.get_dict()
        title = results.get("knowledge_graph", [{"title": ""}])[0]["title"]
        texts = [res.get("text", "") for res in results.get("text_results", [])]
        visual_titles = sorted(
            [res["title"] for res in results["visual_matches"]], key=len
        )[:5]
        return {"title": title, "texts": texts, "visual_titles": visual_titles}

    def check_similarity(self, img):
        """
        Calculate similarity between current and previous all considered images
        """

        # Get embedding from model
        inputs = self.image_processor(img, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).last_hidden_state[:, 0][0]
        output = torch.nn.functional.normalize(output.view(-1), p=2, dim=0)

        # Compare with previous embeddings
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

        logger.info("Starting video processing")
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
                logger.info(f"Image used : {frame_count}")
                img_url = self.upload_img(frame)
                self.img_results.append(self.image_info(img_url))
                logger.info("Image info extracted")

        # Release the video capture object and close windows
        cap.release()
        logger.info("Video processing Done!")
        return self.img_results
