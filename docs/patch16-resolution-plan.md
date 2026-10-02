# Patch16 resolution plan

日期：2026-10-01。狀態：Stage 0–2 已完成（2026-10-02）；patch16 單次 whole 推論 validation accuracy 由 `58.92%` 提升到 `74.12%`，whole＋zoom 為 `80.96%`。結果見第 8 節；尚未讀 test、只有 seed 42。

## 1. 動機

四種 skeleton guidance（class-logit KD、motion-guided spatial attention、SKI-style semantic、decision distillation）在 120-class validation 上都只差 ±0.1 percentage points 左右，沒有可用收益。相對地，不重訓、只把畫面中央 70% 放大的 center zoom，就在完整 12,960 筆 validation 上從 `58.92%` 提升到 `63.91%`，whole＋zoom 機率平均達 `66.94%`。錯誤率最高的類別集中在手指手勢與小物件操作（A072、A074、A084、A073、A082 等），NTU 25-joint skeleton 沒有手指與物件資訊，無法提供這類監督。因此瓶頸較可能是 RGB 輸入解析度，而不是缺少 skeleton 訊號。

目前 backbone 為 `microsoft/xclip-base-patch32`：224×224 輸入切成 7×7=49 個 32px patches，手部通常只落在少數 patches。`microsoft/xclip-base-patch16`（revision `d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6`，Kinetics-400、原生 8 幀）在相同 224 輸入下有 14×14=196 個 16px patches，空間解析度為 4 倍。

## 2. 研究問題

在訓練配方、split、取幀、LoRA 位置與 120-class head 全部不變的前提下，只把 backbone 換成 patch16，能否顯著提升單次 whole 推論的 120-class validation accuracy？提升後，whole＋zoom 推論是否仍有額外收益？

## 3. 為什麼選 patch16

- Processor 設定與 patch32 完全相同（shortest edge 224、center crop 224、相同 mean／std；`image_processor.to_dict()` 逐項相等），可以直接沿用既有 uint8 processed cache，不必重新解碼約 93k 支影片。
- 架構與 patch32 相同（12 層 vision encoder、MIT、512 維 projection），`inject_vision_attention_lora` 依 layer 結構注入，不依賴 patch size；upper-2 vision＋MIT q/v LoRA 仍是 6 個 target modules。
- 不採用 patch32 搭配 448 輸入與 position embedding 內插：token 數與 patch16 相同，但位置編碼不是在該解析度預訓練，且 448 的 uint8 cache 約為現有的 4 倍（約 450 GB），不適合共用伺服器。
- `xclip-large-patch14` 計算量再高數倍，只在 patch16 收益明確且 GPU 時間允許時才考慮。

## 4. 實作

- `src/vlm/training/skeletonclip_phase3_gate1.py`
  - `--stop-after-epoch N`：在第 N 個 epoch 後停止，但 cosine schedule 仍以 `--epochs`（60）計算。每個 epoch 的 learning rate 與 baseline 相同，因此可以直接和 baseline `run_report.json` 的逐 epoch history 比較，不需重跑 control。此參數不進入 run identity，之後可用 `--resume` 延長同一個 run。
  - `--cache-model-name`／`--cache-model-revision`：讓新模型沿用另一個模型的 processed-pixel cache identity；兩者 `image_processor.to_dict()` 不同時直接報錯。
- `src/vlm/training/rgb_input_audit.py`：backbone 改由 adapter checkpoint identity 的 `model_name`／`model_revision` 決定，同一個 runner 可對 patch32 或 patch16 checkpoint 跑 whole＋zoom validation。
- 不新增 persistent tests；本機 CPU 檢查與 GPU smoke 已涵蓋主要契約（見第 7 節）。

## 5. 實驗流程

所有 stage 只讀 canonical validation，不讀 test。

### Stage 0：GPU smoke 與吞吐量

在同一張 GPU 上分別以 patch32 與 patch16 跑 100 個 training batches 與少量 validation batches，確認 trainable parameter audit、finite loss、merge parity 與 merged RGB-only strict reload，並從 `progress.json` 讀 `examples_per_second` 與 peak memory。依實測速度換算 Stage 1 所需時間；若 batch 16 超出記憶體則停下討論，不自行改 batch size。

### Stage 1：patch16 pilot

配方完全沿用 baseline run（`artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/run_lock.json` 的 config）：60-epoch cosine、相同 LR／weight decay／label smoothing／dropout／feature noise、batch 16、seed 42、AMP，只改 `--model-name`／`--model-revision` 並加上 `--stop-after-epoch 20`。Baseline 最佳 epoch 為 13，20 個 epochs 足以涵蓋其峰值。

- 早期檢查（epoch 5）：若 validation accuracy 沒有領先 baseline 同 epoch 至少 `1.0` point，停止並判 no-go。
- 最終判讀：best whole accuracy 至少比 `58.92%` 高 `3.0` points，才值得約 3–4 倍的 vision 推論成本，進入 Stage 2。介於 `0` 與 `3.0` points 之間記為 inconclusive，由使用者決定是否延長。

### Stage 2：patch16 whole＋zoom validation

以 Stage 1 best adapter 在全部 12,960 筆 validation 跑 whole、center zoom70 與兩者機率平均，和 patch32 的 `66.94%` ensemble 比較，並報告 A042–A048 attention correct（patch32 ensemble 為 349）、A001–A060／A061–A120 與 previously ignored 類別。若 zoom 不再帶來明顯收益，代表 patch16 已吸收解析度收益，部署只需一次 forward。

### Stage 3：依 Stage 2 結果決定

- Zoom 仍有收益：在 patch16 上做完整 budget 的 crop-aware 訓練；需另建 zoom view 的 uint8 cache（約 100 GB），啟動前須確認伺服器空間。
- Patch16 收益很大且 GPU 時間允許：評估 `xclip-large-patch14`。
- Person-aware crop 暫不實作：需要額外 detector，違反目前 RGB-only 部署約定；但中央 zoom 會裁掉倒地人物（A043 樣本），之後仍值得研究。

## 6. 指令

伺服器需先下載 patch16 到 Hugging Face cache（runner 使用 `local_files_only=True`）：

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/xclip-base-patch16', revision='d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6', allow_patterns=['*.json', '*.txt', 'model.safetensors'])"
```

Stage 0 smoke（patch32 與 patch16 各一次，只換 model 參數與 artifacts 目錄）：

```bash
.venv/bin/python -m vlm.training.skeletonclip_phase3_gate1 --arm vision --uint8-cache-dir cache/skeletonclip_phase3_rgb_uint8 --workers 24 --max-train-batches 100 --max-val-batches 10 --stop-after-epoch 1 --model-name microsoft/xclip-base-patch16 --model-revision d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6 --cache-model-name microsoft/xclip-base-patch32 --cache-model-revision a2e27a78a2b5d802e894b8a1ef14f3a8ce490963 --artifacts-dir artifacts/patch16_resolution/smoke_patch16
```

Stage 1 pilot（其餘參數照 baseline `run_lock.json`）：

```bash
.venv/bin/python -m vlm.training.skeletonclip_phase3_gate1 --arm vision --uint8-cache-dir cache/skeletonclip_phase3_rgb_uint8 --workers 24 --stop-after-epoch 20 --model-name microsoft/xclip-base-patch16 --model-revision d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6 --cache-model-name microsoft/xclip-base-patch32 --cache-model-revision a2e27a78a2b5d802e894b8a1ef14f3a8ce490963 --artifacts-dir artifacts/patch16_resolution/pilot_seed42
```

Stage 2 whole＋zoom validation：

```bash
.venv/bin/python -m vlm.training.rgb_input_audit --full-validation --whole-zoom-only --workers 24 --adapter artifacts/patch16_resolution/pilot_seed42/vision_lora_ce_seed42/best_adapter.pt --baseline-logits artifacts/patch16_resolution/pilot_seed42/vision_lora_ce_seed42/best_validation_logits.pt --artifacts-dir artifacts/patch16_resolution/whole_zoom_validation
```

## 7. 已完成的本機檢查

在本機 CPU 以隨機輸入 `[2, 8, 3, 224, 224]` 分別建立 patch32 與 patch16 的 `XCLIPVisionLoRAClassifier`：兩者皆為 6 個 LoRA targets，vision／MIT／classifier 三組參數都有非零 gradient，merge 與 strict reload 的最大誤差分別為 `1.67e-6`（patch32）與 `1.31e-6`（patch16）。兩個 processor 的 `image_processor.to_dict()` 完全相同。

## 8. 結果

所有數字皆為 seed 42、完整 12,960 筆 canonical validation，未讀 test。Artifacts 位於 `artifacts/patch16_resolution/`。

### Stage 0：GPU smoke

RTX 2080 Ti 上 patch32 與 patch16 各跑 100 個 training batches 與 10 個 validation batches，兩者皆正常結束。Trainable 參數同為 193,656；merge 與 merged RGB-only strict reload 最大誤差分別為 `1.43e-6` 與 `1.91e-6`；peak memory 為 1.33 GB 與 2.93 GB。前 100 個 batches 的累計吞吐量為 51.4 與 45.3 examples/s，暖機期間主要受資料讀取限制，因此不能代表穩定速度。

### Stage 1：patch16 pilot

完整 pilot 以 baseline 相同配方（60-epoch cosine、batch 16、seed 42、AMP）跑到第 20 個 epoch 停止，總時間 10.55 小時，穩定吞吐量約 50 examples/s，每個 epoch 約 32 分鐘，約為 baseline 的 3 倍。最佳 checkpoint 為 epoch 20，validation accuracy `74.12%`、macro-F1 `75.45%`、NLL `0.931`；merge 與 merged RGB-only strict reload 最大誤差皆為 `8.58e-6`。

| Epoch | 1 | 2 | 3 | 5 | 9 | 13 | 14 | 17 | 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Patch16 | 60.37% | 64.63% | 67.01% | 70.68% | 72.75% | 73.63% | 73.97% | 73.51% | 74.12% |
| Baseline patch32 | — | 47.72% | 51.09% | 55.57% | 57.10% | 58.92% | 57.85% | 58.45% | 58.02% |

Baseline 數字取自原 run 的 `train.log`（該 run 在 epoch 38 中斷，最佳為 epoch 13 的 `58.92%`；epoch 1 未記錄 validation）。Patch16 在 epoch 1 已超過 baseline 最佳值，epoch 5 的早期門檻（領先至少 `1.0` point）與最終門檻（高出至少 `3.0` points）都大幅通過。Epoch 14 之後曲線大致持平在 73.1%–74.1%。

### Stage 2：whole＋zoom validation

以 epoch 20 best adapter 在完整 validation 上推論；whole view 與訓練時保存的 logits 一致（最大誤差 `1.61e-4`，argmax 零差異）。

| View | Patch32 | Patch16 | 差距（points） |
| --- | ---: | ---: | ---: |
| Whole（單次推論） | 58.92% | 74.12% | +15.20 |
| Center zoom70（單次推論） | 63.91% | 78.97% | +15.06 |
| Whole＋zoom 機率平均（兩次推論） | 66.94% | 80.96% | +14.02 |

| Subgroup（patch16 correct／n） | Whole | Zoom70 | Whole＋zoom |
| --- | ---: | ---: | ---: |
| A001–A060 | 3,457／4,320 | 3,585／4,320 | 3,683／4,320 |
| A061–A120 | 6,149／8,640 | 6,650／8,640 | 6,809／8,640 |
| Previously ignored（17 類） | 1,299／2,448 | 1,579／2,448 | 1,570／2,448 |
| Attention A042–A048 | 358／504 | 383／504 | 385／504 |
| Product excluded（24 類） | 1,700／3,168 | 2,073／3,168 | 2,066／3,168 |

A042–A048 的 top-1 correct 由 patch32 的 whole `303`、ensemble `349` 提升到 patch16 的 whole `358`、ensemble `385`，沒有退步。Patch16 下 zoom 仍再提升約 4.9 points、ensemble 約 6.8 points，代表解析度收益尚未飽和；但 ensemble 需要兩次 forward，中央 zoom 也仍有裁掉倒地人物的風險（見 `docs/crop-aware-v1-plan.md`）。

### 判讀

換成 patch16 是目前所有方法中收益最大的一項，單次推論即超過先前 patch32 ensemble。這支持「錯誤主要來自 RGB 空間解析度」的判斷。下一步候選依序為：(1) 第二個 seed 確認可重現性；(2) 在 patch16 上做 crop-aware 或 person-aware 訓練，把 zoom 的收益學進單次推論；(3) 評估 `xclip-large-patch14`。Test 仍保留到配方鎖定後只評估一次。
