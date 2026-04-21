# CLIP Video Prototype

This prototype classifies videos by:

1. sampling every 5th frame from each video,
2. encoding sampled frames with `openai/clip-vit-base-patch32`,
3. padding or truncating the per-video frame embedding sequence to a fixed length,
4. training a bidirectional GRU classifier with temporal mean/max pooling on those embedding sequences.

## Dataset Assumption

The label comes from the `A###` suffix in each filename inside `dataset/prototype/`.

- `S008C001P001R001A001_rgb.avi` -> class `1`
- `S008C001P001R001A042_rgb.avi` -> class `42`

The prototype uses only classes `1` through `10`.

For each class:

- `10` videos are selected for training
- `10` videos are selected for testing

The split is deterministic because the script uses a fixed random seed.

## Files

- `video_dataset.py`: dataset discovery, deterministic splitting, frame sampling
- `train_video_classifier.py`: CLIP embedding extraction, cache creation, stronger bidirectional GRU training, evaluation
- `main.py`: thin entry point that forwards to the training script

## Run

Extract CLIP embeddings only:

```bash
python3 main.py --extract-only
```

Run the full prototype:

```bash
python3 main.py
```

Run with a smaller number of epochs:

```bash
python3 main.py --epochs 5
```

Force the embedding cache to rebuild:

```bash
python3 main.py --force-extract
```

## Outputs

The pipeline writes:

- `cache/selected_split.json`: chosen train/test videos
- `cache/embedding_manifest.json`: cached embedding records
- `cache/embeddings/`: one cached tensor file per selected video
- `artifacts/video_classifier.pt`: trained GRU checkpoint
- `artifacts/evaluation_results.json`: accuracy history and per-video predictions

## Notes

- This is a proof-of-concept baseline, not a final video recognition system.
- CLIP is used only as an image encoder here; the temporal classifier is learned separately.
- With only `100` training videos total, expect some instability across runs even though the stronger training setup is much better than naive guessing.
