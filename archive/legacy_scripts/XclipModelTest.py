import av
import numpy as np
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


container = av.open("dataset/crop.mov")
total_frames = container.streams.video[0].frames


# sample 8 frames
indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=total_frames)
video = read_video_pyav(container, indices)
print(video.shape)

from transformers import XCLIPProcessor, XCLIPModel
import torch

model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

start = time.perf_counter()
inputs = processor(
    text=["yawning", "fishing", "walking"],
    images=list(video),
    return_tensors="pt",
    padding=True,
)

# forward pass
with torch.no_grad():
    outputs = model(**inputs)
elapsed = time.perf_counter() - start
print(f"Processor + model time: {elapsed:.4f}s")

probs = outputs.logits_per_video.softmax(dim=1)
print(probs)