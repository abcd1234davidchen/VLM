# XCLIP MIT LoRA + CE-only 實作計畫

## 1. 目標

在現有 `microsoft/xclip-base-patch32` visual-only XCLIP baseline 上，只對 Multiframe Integration Transformer（MIT）的 self-attention 加入 LoRA，並以 ground-truth 120-class cross-entropy loss 訓練 LoRA 與 classification head。

本實驗的目的不是完成 SkeletonCLIP distillation，而是建立必要的 control baseline，回答「在沒有 skeleton teacher 的情況下，單純讓 XCLIP temporal representation 適應 NTU120，可以比 frozen XCLIP 提升多少」。

目前 frozen baseline 的正式參考結果為：best validation epoch 49、validation accuracy 0.3927、test accuracy 0.4254。新實驗必須沿用相同 split manifest、八幀取樣方式與 checkpoint selection 規則，才可直接比較。

## 2. 成功條件

- LoRA 只注入 `XCLIPModel.mit.encoder.layers[*].self_attn`，第一版只訓練 `q_proj` 與 `v_proj` adapter。
- XCLIP vision encoder、visual projection、MIT 原始權重、text model與 prompt generator 全部維持 frozen。
- 可訓練參數只有 MIT LoRA weights 與新的 120-class classification head；啟動時列印並保存 trainable parameter names、數量與比例。
- Training loss 只有帶既有 label smoothing 的 cross-entropy，不包含 KL distillation、feature cosine loss、text loss 或 skeleton input。
- 使用與 frozen baseline 完全相同的 participant-disjoint `selected_split.json`、class ID mapping、八幀 whole-video uniform sampling及 seed。
- Checkpoint 只依 validation accuracy 選擇，restore best checkpoint 後才執行一次 test evaluation。
- Frozen baseline 的 CLI、feature cache、checkpoint 與 artifacts 行為不被改壞。
- Smoke run 能完成 forward、backward、checkpoint save/load 與 LoRA merge parity 驗證；完整實驗產生可和 baseline 對照的結果檔。

## 3. 明確不在本階段處理的範圍

- 不加入 SkeletonCLIP teacher、skeleton data loader、sample pairing、KL loss 或 feature alignment。
- 不把 LoRA 擴展到 vision encoder、visual projection、MIT MLP、text encoder 或 prompt generator。
- 不更換 ActionCLIP，不改 XCLIP pretrained checkpoint，不增加 frame 數。
- 不使用 frozen baseline 的 post-MIT 512 維 video-feature cache 進行訓練，因為這會繞過 MIT，LoRA 無法收到 gradient；本實驗改用獨立的 pre-MIT frame-feature cache。
- 不因這個實驗重新設計全部 training pipelines；只抽出 LoRA 路徑實際需要的共用 split、evaluation 與 artifact 邏輯。
- 不根據 test accuracy 調整 hyperparameters；所有選擇只能根據 train/validation 結果。

## 4. 模型設計

首次建立 cache 的資料流如下：`video → deterministic 8-frame decode → XCLIPProcessor → frozen vision encoder → frozen visual projection → cache [8, 512] pre-MIT frame features`。

每個 training epoch 的資料流如下：`cached [8, 512] pre-MIT frame features → MIT with LoRA → LayerNorm/Dropout/Linear head → 120-class logits`。

建立 `XCLIPMITLoRAClassifier` wrapper，內含 pretrained XCLIP 的 MIT、LoRA adapters 與 classification head。Forward 接受 `[batch, 8, 512]` pre-MIT features，呼叫 MIT 取得 pooled video representation，再傳入 head。Vision encoder與visual projection只在建立cache時使用，不進入每個epoch的training graph。

第一版 LoRA target 固定為每個 MIT encoder layer self-attention 的 `q_proj` 與 `v_proj`。建議預設 `rank=8`、`alpha=16`、`dropout=0.05`，使用標準縮放 `alpha / rank`，LoRA B matrix 以零初始化，確保注入瞬間模型輸出與原 pretrained XCLIP 相同。

優先在專案內實作小型、可合併的 `LoRALinear`，直接包裝已解析到的 MIT projection modules，不新增 PEFT dependency。這可避免寬鬆的 `q_proj`／`v_proj` 名稱比對誤傷 vision encoder或 prompt generator，也讓 merge/unmerge 與 checkpoint schema完全由專案控制。

分類 head 沿用 frozen baseline 的 `LayerNorm → Dropout → Linear` 結構。Head 與 LoRA 使用不同 optimizer parameter groups，因為隨機初始化的 head 通常需要比 LoRA 更高的 learning rate。

## 5. Pre-MIT cache、資料與取樣

新增獨立的 pre-MIT cache pipeline。它必須重用 `read_uniform_video_frames()` 與 `FRAME_SAMPLING_STRATEGY = "whole_video_uniform"`，使用 frozen XCLIP vision encoder與visual projection，並在 MIT 之前保存每部影片的八個 projected frame features，tensor shape固定為 `[8, 512]`。

新 cache 必須寫入獨立路徑，例如 `cache/xclip_pre_mit_features/<config-key>/`，不得混入現有 `cache/xclip_features/`。現有路徑代表 post-MIT pooled 512-D features；新路徑代表 pre-MIT `[8, 512]` frame features，兩者的檔名、manifest、metadata與相容性檢查都必須能明確區分。

Pre-MIT cache metadata至少包含base model ID、Transformers版本、frame sampling strategy、frame count、feature stage `pre_mit_projected_frames`、tensor shape、dtype與video path。相容性檢查必須拒絕舊的post-MIT vector、錯誤frame count、不同model或不同sampling policy。

新增 cached dataset，從磁碟載入 `[8, 512]` features、class index與sample metadata。Collate後輸出固定為 `[batch, 8, 512]`，因此 training不需要再次decode影片、執行processor或跑vision encoder。

Cache extraction需要暴露 `--feature-batch-size` 與 `--decode-workers`；training DataLoader則使用`--batch-size`、`--loader-workers`、`--pin-memory`與`--persistent-workers`。先以能穩定運作的小batch啟動，再用`--gradient-accumulation-steps`維持有效batch size。

以FP16保存時，每部影片的純tensor大小約為`8 × 512 × 2 bytes = 8 KiB`，80k部影片約625 MiB，不含metadata與大量小檔案的filesystem overhead。實作前仍需用實際cache layout量測磁碟占用與讀取throughput。

訓練時仍採 deterministic whole-video uniform sampling，以隔離 LoRA 本身的效果；temporal jitter、multi-window sampling 與其他 augmentation 留待後續獨立 ablation。

## 6. Training 設定

Loss 使用 `CrossEntropyLoss(label_smoothing=0.05)`，這裡的 CE 定義必須和 frozen baseline 一致。

初始建議設定如下，正式 full run 前先以小 subset 做顯存與 throughput calibration：

- Epochs：60，上限與 frozen baseline 一致。
- LoRA learning rate：`1e-4`。
- Head learning rate：`8e-4`。
- Weight decay：`5e-4`，bias 與 normalization parameters不做 weight decay。
- Scheduler：Cosine annealing，`T_max=epochs`。
- LoRA：rank 8、alpha 16、dropout 0.05，targets 為 MIT `q_proj`、`v_proj`。
- Head dropout：0.4。
- Gradient clipping：1.0。
- Mixed precision：CUDA 預設 BF16；不支援 BF16 時使用 FP16，CPU/MPS 先使用 FP32。CLI 必須允許明確關閉 AMP。
- Gradient accumulation：預設 1，依伺服器顯存調整；log 中同時記錄 micro-batch size 與 effective batch size。

不得直接沿用 frozen head 的 optimizer state 或 scheduler state。若未實作完整 resume contract，第一版只支援從 pretrained XCLIP 開始的新 run，避免產生看似 resume、實際 scheduler 或 optimizer 不一致的結果。

## 7. CLI 與 artifacts

保留 `vlm train xclip` 作為 frozen baseline。新增明確命令 `vlm train xclip-lora`，避免同一命令因旗標不同而悄悄改變資料路徑、模型語意與 checkpoint 格式。`vlm embeddings extract --pipeline xclip-lora`則負責建立或補齊pre-MIT cache，不得產生post-MIT pooled features。

建議命令：

```bash
uv run vlm train xclip-lora \
  --split-manifest artifacts/xclip/selected_split.json \
  --cache-dir cache \
  --artifacts-dir artifacts/xclip_lora_ce \
  --model-name microsoft/xclip-base-patch32 \
  --frames 8 \
  --epochs 60 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-learning-rate 1e-4 \
  --head-learning-rate 8e-4
```

預設 artifacts 必須與 frozen baseline 隔離在 `artifacts/xclip_lora_ce/`，至少包含：

- `selected_split.json`：重寫本 run 的 sampling metadata，並保留 source manifest reference。
- `xclip_lora_classifier.pt`：best validation checkpoint，包含 LoRA、head、class mapping、base model ID與完整 config。
- `xclip_lora_merged.pt`：將 LoRA delta merge 到 MIT projection 後的 inference checkpoint；保存前必須通過 merge parity check。
- `evaluation_results.json`：history、best epoch、validation accuracy、final test accuracy、prediction rows、trainable parameter audit、sampling policy、AMP dtype與 effective batch size。
- `accuracy_graph.png`：train/validation accuracy 曲線。
- `run_config.json`：完整 CLI config、dependency versions與 checkpoint provenance，避免只靠 checkpoint 內部欄位追溯。

Pre-MIT cache另有`cache/xclip_pre_mit_feature_manifest.json`，記錄cache properties與每個video對應的feature path。它不是run artifact，且不得覆寫現有`cache/xclip_feature_manifest.json`。

Checkpoint 不應存 text-related runtime state或 optimizer backward graph。若完整 XCLIP state 太大，adapter checkpoint與 merged inference checkpoint要明確分開：adapter checkpoint保存 base model ID、LoRA state與head state；merged checkpoint則保存可直接重建 RGB-only student 的必要 state。

## 8. 預計程式變更

### 8.1 共用資料與模型元件

- `src/vlm/xclip_features.py`：保留既有post-MIT frozen feature cache行為；只抽出可安全共用的八幀decode與processor helper，不改既有cache identity。
- `src/vlm/xclip_pre_mit_features.py`（新增）：pre-MIT cache path、metadata、compatibility validation、batched extraction、manifest與`CachedXCLIPPreMITDataset`。
- `src/vlm/models/lora.py`（新增）：`LoRALinear`、MIT target injection、trainable parameter audit、merge/unmerge與LoRA state extraction/loading。
- `src/vlm/models/xclip_lora.py`（新增）：接受cached `[batch, 8, 512]` features的MIT LoRA classifier wrapper與head。
- `src/vlm/models/__init__.py`：匯出新的明確模型介面。

### 8.2 Training 與 evaluation

- `src/vlm/training/xclip_lora.py`（新增）：CLI、split preparation、pre-MIT cache extraction/loading、optimizer groups、AMP／gradient accumulation training loop、validation selection、test-once與artifacts。
- `src/vlm/training/engine.py`：只抽出可清楚重用的 gradient accumulation、autocast與scaler helper；不要把 sequence、feature與video batch硬塞進難以理解的單一函式。
- `src/vlm/evaluation/multiclass.py`：新增接受 `pixel_values` 的 video model evaluator，但保持 prediction row schema與既有 evaluator一致。
- `src/vlm/cli.py`：註冊`xclip-lora` training command，並讓`vlm embeddings extract --pipeline xclip-lora`只建立pre-MIT cache。
- `src/vlm/reporting/artifacts.py`：如需要，新增 adapter／merged checkpoint metadata helper，但維持既有 `save_checkpoint()` 相容。

### 8.3 文件與依賴

- `README.md`：新增 CE-only LoRA command、與 frozen cache差異、顯存注意事項、artifact paths及公平比較方式。
- `PROJECT_STRUCTURE.md`：記錄pre-MIT cache dataset、LoRA model與training module責任。
- `TODO.md`：完成後勾選「第一版 student 在 MIT attention 加入 LoRA」的 CE-only control部分，但 SkeletonCLIP distillation本身仍保持未完成。
- `pyproject.toml`／`uv.lock`：若採用專案內 `LoRALinear`，不新增依賴；只在實作證明 Transformers內部介面不足時才重新評估 PEFT。

## 9. 實作階段

### Phase 1：鎖定 baseline contract

- 使用現有 `artifacts/xclip/selected_split.json` 作 canonical manifest，記錄其 hash、sample counts、class IDs與baseline結果。
- 抽出共用 XCLIP frame decode／processing helper，執行現有 XCLIP feature tests，確認既有post-MIT feature tensor與cache metadata完全不變。
- 定義pre-MIT cache identity與manifest schema，確認它不會讀取、覆寫或誤判既有post-MIT cache。
- 用目前鎖定的 Transformers 5.3.0 檢查實際 module names，啟動時若 MIT layer結構或 target數量不符預期則 fail fast，不可靜默訓練錯誤 modules。

### Phase 2：LoRA 與模型 wrapper

- 實作 `LoRALinear`，加入 zero-delta initialization、merge/unmerge、state save/load與dtype/device propagation。
- 只替換 `model.mit.encoder.layers[*].self_attn.q_proj` 與 `v_proj`。
- Freeze全部 base XCLIP，再啟用 LoRA與head；assert trainable set沒有其他參數。
- 以固定 synthetic input驗證：注入前後 logits一致、LoRA收到非零gradient、base weights沒有gradient、merge前後 logits在容許誤差內一致。

### Phase 3：Pre-MIT cache與training path

- 建立pre-MIT extraction path，重用八幀deterministic sampling，並保存visual projection後、MIT前的`[8, 512]` features。
- 建立cached dataset與collate，驗證batch shape為`[batch, 8, 512]`。
- 實作MIT forward、CE loss、AMP、gradient accumulation、gradient clipping與optimizer groups。
- 加入 validation evaluator與best-checkpoint restore。
- 確認 model在 validation/test時為 eval mode，而且整段使用 inference mode。

### Phase 4：Smoke run

- 使用 canonical manifest 搭配小型 class/sample cap副本，跑 1–2 epochs，不修改正式 manifest。
- 驗證 loss可下降、只有 LoRA/head權重改變、checkpoint可重新載入、prediction rows與sample數一致。
- 分別記錄cache extraction的videos/second與training的batches/second、單GPU memory peak及每epoch時間，據此設定full run的micro-batch與accumulation。
- 執行 merged checkpoint parity：同一 batch的unmerged與merged logits最大絕對誤差應小於預先設定容許值，例如 FP32 `1e-5`、BF16／FP16 `1e-3`。

### Phase 5：完整 CE-only 實驗

- 使用 canonical split跑60 epochs，validation每個epoch評估並保存最佳狀態。
- 僅在訓練完成並restore最佳validation checkpoint後執行一次test。
- 輸出與 frozen baseline並排的 validation/test accuracy、best epoch、trainable parameter count、training time與peak memory。
- 若 validation沒有改善，不得用 test結果挑選其他LoRA hyperparameters；另開受控validation experiment。

## 10. 測試策略

依照專案的 testing restraint，只新增保護重要 contract的測試，不為每個helper建立重複性測試。

- `tests/test_xclip_lora.py`（新增）：精確 target selection、trainable parameter whitelist、zero-delta output parity、gradient isolation、merge parity及adapter checkpoint round trip。
- `tests/test_xclip_pre_mit_features.py`（新增）：cache path isolation、metadata compatibility、`[8, 512]` tensor contract、deterministic sampling、manifest與collate batch shape。
- `tests/test_xclip_features.py`：保留並視抽取共用helper需要更新，證明 frozen cache path、metadata與feature extraction contract不變。
- CLI contract test：`vlm train xclip-lora --help`可用，且`vlm embeddings extract --pipeline xclip-lora`只建立pre-MIT cache。
- 最小integration smoke test使用mock XCLIP，驗證一個training epoch、validation selection與checkpoint restore；不在一般test suite下載大型pretrained weights或decode完整dataset。

## 11. 實驗比較與判讀

主要比較表固定包含 frozen XCLIP與MIT LoRA + CE only，兩者使用相同split、model ID、frames、sampling與seed。

核心指標為 best validation accuracy與final test accuracy；同時保留每個epoch的train loss、train accuracy與validation accuracy。若後續補上macro-F1或top-5 accuracy，必須對兩個baseline用相同evaluation code重新計算，不能只替新模型報告。

本階段成功不要求一定超過0.3927 validation accuracy，但若沒有超過，必須先排除target錯誤、LoRA沒有gradient、learning rate不合適或pre-MIT cache與baseline preprocessing不一致，再判定MIT LoRA無效。

只有在 CE-only baseline穩定完成後，下一階段才加入 SkeletonCLIP teacher的KL distillation。屆時唯一主要變因應是teacher supervision，才能將 `LoRA + CE + KL` 相對於本計畫的提升歸因於skeleton knowledge。

## 12. 風險與緩解

- **首次cache extraction造成CPU bottleneck：** 量測decode與GPU utilization，再調整feature batch、workers與prefetch；cache完成後每個epoch不得重複decode或執行vision encoder。
- **大量小cache檔案降低I/O throughput：** 先量測目前per-video `.pt` layout；若成為瓶頸，再改用sharded cache，但必須保留逐sample identity與相容性驗證。
- **80k videos造成訓練時間大幅增加：** smoke run以pre-MIT cache估算完整epoch時間，保存清楚的throughput與ETA；必要時使用獨立的小型validation experiment調參，但正式結果仍回到完整canonical split。
- **顯存不足：** 降低micro-batch、啟用BF16／FP16與gradient accumulation；不先縮減frames或解凍更多backbone權重。
- **Transformers內部module名稱變動：** 針對5.3.0鎖定結構並fail fast，checkpoint記錄transformers版本與target module names。
- **LoRA誤注入非MIT attention：** 使用從`model.mit.encoder.layers`直接遍歷的結構化注入，不使用全模型模糊suffix搜尋。
- **Validation提升來自preprocessing差異：** 共用同一decode與processor helper，並驗證pre-MIT cache經原始MIT後可重現既有post-MIT feature。
- **Test leakage：** hyperparameter與checkpoint只看validation，test每個正式run只執行一次並完整記錄run identity。
- **Checkpoint過大或無法部署：** adapter與merged checkpoint分離，merge後以RGB-only forward做parity與load test。

## 13. 完成定義

- `vlm train xclip-lora`可在canonical split上完成end-to-end training。
- `vlm embeddings extract --pipeline xclip-lora`可建立、重用並驗證獨立的pre-MIT cache，且不修改既有post-MIT cache。
- Trainable parameter audit證明只有MIT q/v LoRA與classification head可更新。
- Frozen XCLIP pipeline的既有tests與行為全部通過。
- LoRA unit tests、pre-MIT cache tests與integration smoke test通過。
- Best-validation adapter checkpoint與merged RGB-only checkpoint皆可重載，且merge parity符合容許值。
- `evaluation_results.json`足以重現model、split、sampling、optimizer、LoRA、precision與batch設定。
- 完整60-epoch CE-only結果可和validation 0.3927／test 0.4254的frozen baseline直接比較。
- 文件清楚標示本階段沒有使用SkeletonCLIP，並把此結果定位為下一階段distillation的control baseline。
