from serpapi import GoogleSearch

params = {
  "engine": "google_lens",
  "url": "https://imgtr.ee/images/2023/08/17/7945d882ff66a227cdd80b9f6dbcbaac.jpeg",
  "api_key": "93795a8ee93145e79651985b94fe655769bd217b6ca98c1128153125d97b8b49"
}

search = GoogleSearch(params)
results = search.get_dict()
visual_matches = results["visual_matches"]


from transformers import AutoImageProcessor, SwiftFormerModel
import torch

image_processor = AutoImageProcessor.from_pretrained("MBZUAI/swiftformer-xs")
model = SwiftFormerModel.from_pretrained("MBZUAI/swiftformer-xs")


previous_embeddings = []

def get_score(img):

    if len(previous_embeddings) == 0:
        False

    inputs = image_processor(img, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0][0]

    output = torch.nn.functional.normalize(output.view(-1), p=2, dim=0)
    for embedding in previous_embeddings:
        score = torch.nn.functional.cosine_similarity(output, embedding, dim=0).numpy()
        if score > 0.2:
            return True

    previous_embeddings.append(output)

    return False


import cv2
import time
from pytube import YouTube

# YouTube video URL
video_url = "https://www.youtube.com/watch?v=Ic_7K5Nk5gg"

# flag
flag = False

# Path to save the frames
output_path = "/content/"

# Time interval in seconds
interval = 2

# Initialize variables for frame capture and time tracking
frame_count = 0
last_capture_time = time.time()

# Create a YouTube object and get the highest resolution stream
yt = YouTube(video_url)
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

        result = get_score(frame)
        if result:
            continue

        # Save the captured frame
        image_filename = f"{output_path}frame_{frame_count:04d}.jpg"
        cv2.imwrite(image_filename, frame)
        print(f"Captured: {image_filename}")


    # Display the frame (optional)
    # cv2.imshow("Video", frame)

    # Break the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

# Release the video capture object and close windows
cap.release()
# cv2.destroyAllWindows()
