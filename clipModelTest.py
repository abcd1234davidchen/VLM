from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np


def videoMaxPooling(video_path, pool_size):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    pooled_frames = []
    for i in range(0, len(frames), pool_size):
        pool = frames[i:i + pool_size]
        max_frame = np.max(pool, axis=0)
        pooled_frames.append(max_frame)
    print(f"Pooled {len(frames)} frames into {len(pooled_frames)} frames.")
    return pooled_frames

def main():

    video_path = "dataset/V0WK7.mp4"
    pool_size = 64

    pooled_frames = videoMaxPooling(video_path, pool_size)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=["wearing clothes","cleaning up"], images=pooled_frames, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print(probs)

if __name__ == "__main__":
    main()