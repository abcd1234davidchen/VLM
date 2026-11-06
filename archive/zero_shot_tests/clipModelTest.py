from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import av
import numpy as np
import torch
import time


np.random.seed(0)


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def main():
    video_path = "dataset/subset264/S19/S019C001P008R001A103_rgb.mp4"
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames

    # sample 8 frames from the specified video
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=total_frames)
    video = read_video_pyav(container, indices)
    print(f"Sampled frames shape: {video.shape}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    pil_frames = [Image.fromarray(frame.astype("uint8")) for frame in video]
    texts = ["yawning", "sleeping", "writing"]

    for frame_index, frame in enumerate(pil_frames):
        start = time.perf_counter()
        inputs = processor(
            text=texts,
            images=frame,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.perf_counter() - start

        logits_per_image = outputs.logits_per_image  # (1, 3) for a single frame
        probs = logits_per_image.softmax(dim=1)
        print(f"frame {frame_index} CLIP processor + model time: {elapsed:.4f}s")
        print(probs.squeeze(0))

if __name__ == "__main__":
    main()