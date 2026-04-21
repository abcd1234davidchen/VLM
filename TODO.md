# TODO

## Possible Next Training Improvements

- Add a real validation split and early stopping so the test set is no longer checked every epoch.
- Save the best checkpoint by validation accuracy instead of only saving the final epoch checkpoint.
- Increase regularization and sequence augmentation, such as random frame dropout, temporal jitter, or small Gaussian noise on embeddings.
- Add a pooled-feature branch using temporal statistics like mean, max, and standard deviation, then fuse it with the GRU output.
- If the prototype scope can be relaxed, use more than `10` training videos per class because the current training curve suggests data is the main bottleneck.
