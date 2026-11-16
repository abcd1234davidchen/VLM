import av
import time
import numpy as np
import torch

from transformers import AutoProcessor, XCLIPVisionModel, XCLIPVisionConfig

np.random.seed(0)
FRAME_NUM = 32


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


container = av.open("dataset/subset264/S19/S019C001P008R001A103_rgb.mp4")
total_frames = container.streams.video[0].frames


# sample 8 frames
indices = sample_frame_indices(clip_len=FRAME_NUM, frame_sample_rate=1, seg_len=total_frames)
video = read_video_pyav(container, indices)
print(video.shape)

model_name = "microsoft/xclip-base-patch32"
processor = AutoProcessor.from_pretrained(model_name)
configuration = XCLIPVisionConfig(num_frames=FRAME_NUM)
model = XCLIPVisionModel(config=configuration).from_pretrained(model_name)

start = time.perf_counter()
pixel_values = processor(images=list(video), return_tensors="pt").pixel_values

batch_size, num_frames, num_channels, height, width = pixel_values.shape
pixel_values = pixel_values.reshape(-1, num_channels, height, width)

# forward pass
with torch.no_grad():
    outputs = model(pixel_values)
elapsed = time.perf_counter() - start
print(f"Processor + model time: {elapsed:.4f}s")

last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)