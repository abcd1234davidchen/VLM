# Early prototypes

The first scripts of the project, kept as they were before the code moved into the `src/vlm` package. They may no longer run and will not be maintained. The same workflows live on in the `vlm` CLI.

## `zero_shot_tests/`

These scripts were used to test if the pretrained model could recognize an action from text labels alone without training.

| Script | Function | Now |
| --- | --- | --- |
| `clipModelTest.py` | CLIP image–text similarity on single frames | `vlm experiment clip` |
| `XclipModelTest.py` | X-CLIP video–text similarity on a whole clip | `vlm experiment xclip` |
| `XCVisionTest.py` | X-CLIP vision encoder as a feature extractor | `vlm experiment xclip-vision` |

## `clip_gru/`

This is the first classifier that we trained. It gets a CLIP embedding for each frame. Then a bidirectional GRU processes the sequence of embeddings. Run from inside the folder, since the scripts import each other by module name.

| Script | Purpose | Now |
| --- | --- | --- |
| `train_video_classifier.py` | Multiclass action classifier | `vlm train multiclass` |
| `train_binary_video_classifier.py` | Safe/danger classifier (A042, A043, A048 as danger), the first alert idea | `vlm train binary` |
| `video_embeddings_classifier.py` | GRU with learnable class prototypes instead of a linear head | `vlm train prototype` |
| `video_dataset.py`  | Video discovery from filenames, train/test split, frame sampling | `vlm.data` |
| `video_embedding_cache.py` | On-disk cache of per-video CLIP embeddings | `vlm.embeddings` |
| `main.py` | Entry point that ran `train_video_classifier.py` | `vlm train multiclass` |
