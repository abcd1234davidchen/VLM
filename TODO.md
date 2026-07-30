# TODO

## Dataset Splits and Evaluation

- [x] Split train/validation/test by participant rather than by individual
      video so the same participant cannot appear in more than one split.
- [x] Add a validation split to the multiclass and prototype pipelines so all
      three training commands use train/validation/test datasets.
- [x] Use validation metrics to select the best multiclass and prototype
      checkpoints; evaluate the test set only once after checkpoint selection.
- [ ] Decide whether early stopping should be shared by all three training
      pipelines.
- [x] Keep split generation deterministic and record the seed and selected
      videos in each split manifest.

## Benchmark Fairness and Reproducibility

- [ ] Generate one canonical split manifest and require multiclass, prototype,
      binary, and future XCLIP pipelines to consume it instead of independently
      regenerating splits. Derive task-specific labels from the same selected
      video records.
- [ ] Make split identity explicit in every experiment report with a manifest
      hash or stable split ID, and fail fast when runs being compared use
      different manifests.
- [ ] Decide whether to use the official NTU cross-subject protocol or a custom
      seeded participant split. Document the choice and do not compare custom
      split results directly with official-protocol benchmarks.
- [ ] Add participant-level stratification or a deterministic allocation
      search so train/validation/test retain reasonable action and safe/danger
      distributions. Report both participant counts and actual video counts,
      since participant fractions do not guarantee video-count fractions.
- [ ] When a data cap is enabled, make its semantics comparable across tasks.
      The multiclass cap is currently per action class while the binary cap is
      per aggregated safe/danger category, which can change the original-action
      distribution.
- [ ] Report the exact raw and effective training-set sizes for every run,
      including augmented samples, batches per epoch, total optimizer steps,
      and approximate compute time. Equal epoch counts are not equal compute
      budgets when only one pipeline expands its dataset.
- [ ] Add a controlled benchmark mode that applies the same augmentation and
      class-balancing policy to all pipelines. Keep task-specific policies as
      separately named experiments rather than implicit defaults.
- [ ] Add multiclass macro F1, per-class precision/recall/F1, balanced
      accuracy, and a confusion matrix so multiclass and prototype runs are not
      selected and reported using accuracy alone.
- [ ] Keep task-specific primary metrics but define a common reporting schema.
      Do not compare binary accuracy directly with multiclass accuracy as if
      they represented the same prediction problem.
- [ ] Report model parameter counts, trainable parameter counts, inference
      latency, and memory use. Distinguish a controlled comparison with shared
      hyperparameters from a best-achievable comparison with validation-based
      tuning for each architecture.
- [ ] Run every benchmark over a fixed seed set, report mean and standard
      deviation, and enable deterministic PyTorch algorithms where supported.
      Record any CUDA/MPS operations that remain nondeterministic.
- [ ] Give multiclass, prototype, binary, and XCLIP distinct default artifact
      directories and filenames so sequential runs cannot overwrite
      checkpoints, plots, manifests, or evaluation JSON files.
- [ ] Add a run ID and record package versions, device, command-line arguments,
      git revision, split ID, cache configuration, and model configuration in
      every experiment report.
- [ ] Make embedding cache paths collision-safe by including a normalized
      relative path or path hash rather than only `video_path.stem`. Add a test
      with identically named videos in different subdirectories and verify that
      they load distinct tensors.
- [ ] Preserve cache manifests per run or make the shared manifest cumulative;
      do not overwrite the only cache audit record whenever another pipeline
      runs.
- [ ] Add an end-to-end benchmark validation command that checks participant
      disjointness, manifest equality, label coverage, class distributions,
      cache provenance, augmentation policy, metric schema, and artifact
      isolation before accepting a comparison.

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

## XCLIP Video Classifier

- [ ] Add a visual-only XCLIP video-classification baseline to compare against
      the current frozen CLIP frame encoder plus BiGRU.
- [ ] Use `XCLIPModel.get_video_features()` rather than
      `XCLIPVisionModel` alone. `XCLIPVisionModel` includes cross-frame
      communication but returns per-frame features; the full video path also
      retains the visual projection and Multiframe Integration Transformer
      (MIT) that produce a video-level representation without running the text
      model.
- [ ] Start with the pretrained checkpoint's native eight-frame input and feed
      its 512-dimensional video feature into a LayerNorm, dropout, and linear
      classification head.
- [ ] Train in stages: first freeze all XCLIP parameters and train only the
      classification head; then evaluate unfreezing the MIT and visual
      projection; only consider partial or full vision-encoder fine-tuning if
      validation results justify it.
- [ ] For long videos, evaluate several eight-frame temporal windows and
      aggregate their logits instead of relying on one random window.
- [ ] Compare CLIP+BiGRU and XCLIP using the same participant-disjoint split,
      validation-based checkpoint selection, metrics, and random seeds.

## Hierarchical Danger and Action Prediction

- [ ] Define and document the action-to-danger label ontology before training;
      do not assume every action outside IDs `42`, `43`, and `48` is safe
      without validating that product definition.
- [ ] Prefer one shared XCLIP video encoder with two prediction heads over two
      separately executed models:
      - a binary head that predicts the probability of danger;
      - a multiclass head that predicts the specific action.
- [ ] Train both heads jointly with a weighted binary danger loss and
      multiclass action loss, then evaluate whether shared multi-task features
      improve either task.
- [ ] Avoid hard routing where an initial safe prediction prevents action
      classification. Always compute the action output so a binary
      false-negative does not silently discard potentially useful evidence.
- [ ] Select the danger alert threshold on the validation set according to the
      required safety tradeoff rather than fixing it at `0.5`; start by
      evaluating a target such as danger recall of at least 95%.
- [ ] Report danger recall, false-negative rate, precision, PR-AUC, and false
      alarms in addition to overall accuracy. Report macro F1 and a confusion
      matrix for the action head.
- [ ] Add an end-to-end metric that requires both the correct danger decision
      and the correct specific action, and evaluate all metrics on the
      participant-disjoint test set.

## Augmentation Audit and Redesign

- [ ] Separate class balancing from augmentation. Use class weights or a
      weighted sampler for imbalance, and give safe and danger samples the same
      augmentation types and probabilities so transform artifacts cannot become
      a shortcut for the danger label.
- [ ] Replace the current one-time, offline augmented embedding copies with
      training-time clip augmentation where practical. Resample augmentation
      parameters across epochs while applying the same spatial transform to
      every frame within one clip.
- [ ] Add video-specific temporal augmentation experiments, including random
      temporal windows, temporal jitter, frame dropout, and playback-speed
      variation.
- [ ] Prevent artificial rotation borders by choosing and documenting an
      appropriate fill strategy instead of PIL's default black fill.
- [ ] Run an ablation with four controlled configurations:
      - neither augmentation nor class weights;
      - class weights only;
      - augmentation only;
      - augmentation and class weights together.
- [ ] Extend cache compatibility metadata with a source-video fingerprint, such
      as file size and modification time or a content hash, so replacing a
      video at the same path invalidates its embeddings.
- [ ] Add an explicit, reviewable cache-pruning command for augmented embeddings
      that are no longer referenced by the active manifest. Do not delete stale
      cache entries automatically during training.
- [ ] Record each augmented sample's split, class, seed, and full transform
      parameters in the cache manifest and experiment report.
- [ ] Add tests covering train-only augmentation, identical transforms across
      all frames in a clip, deterministic seeds, distinct cache keys, cache
      invalidation, and absence of augmented samples in validation/test.

## Possible Training Improvements

- Increase regularization and sequence augmentation, such as random frame dropout, temporal jitter, or small Gaussian noise on embeddings.
- Add a pooled-feature branch using temporal statistics like mean, max, and standard deviation, then fuse it with the GRU output.
- [x] Use all discovered multiclass and prototype videos by default because the
      earlier fixed per-class sample budget made data the main bottleneck.
      Keep `--max-videos-per-class` as an opt-in limit for quick experiments.
