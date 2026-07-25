# TODO

## Dataset Splits and Evaluation

- [ ] Add a validation split to the multiclass and prototype pipelines so all
      three training commands use train/validation/test datasets.
- [ ] Use validation metrics to select the best multiclass and prototype
      checkpoints; evaluate the test set only once after checkpoint selection.
- [ ] Decide whether early stopping should be shared by all three training
      pipelines.
- [ ] Keep split generation deterministic and record the seed and selected
      videos in each split manifest.

## Video and Frame Sampling

- [ ] Decide whether the CLIP/XCLIP experiments should represent a random
      contiguous clip or the entire video.
- [ ] If whole-video coverage is intended, replace the experiments' random
      contiguous window with uniformly spaced frames across the full duration.
- [ ] Consider evaluating multiple temporal windows for long videos and
      aggregating their predictions.
- [ ] Expose the experiment sampling strategy and frame sample rate as CLI
      options.
- [ ] Document that training currently samples every fifth frame and then
      uniformly reduces long embedding sequences to 32 time steps.

## Possible Training Improvements

- Increase regularization and sequence augmentation, such as random frame dropout, temporal jitter, or small Gaussian noise on embeddings.
- Add a pooled-feature branch using temporal statistics like mean, max, and standard deviation, then fuse it with the GRU output.
- If the prototype scope can be relaxed, use more than `10` training videos per class because the current training curve suggests data is the main bottleneck.
