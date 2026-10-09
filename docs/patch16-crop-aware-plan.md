# Patch16 crop-aware 訓練計畫

日期：2026-10-09。狀態：計畫與實作完成、CPU 檢查通過；GPU smoke 與完整訓練尚未啟動。本文完成 `docs/patch16-resolution-plan.md` Stage 3 唯一剩下的項目。只讀 canonical train／validation，不讀 test。

## 1. 目標與範圍

在 `microsoft/xclip-base-patch16`（revision `d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6`）上做完整 budget 的 crop-aware 訓練：每個訓練樣本每個 epoch 只看 whole 或 center_zoom70 其中一個 view，標籤不變，目標是把 zoom 的收益學進模型本身，再和 patch16 pilot 比較。

不在範圍內：seed 2、`xclip-large-patch14`、EmbeddingGemma、VLM verifier、警報與門檻、person-aware crop，以及任何 test split 讀取。

## 2. 比較基準：patch16 pilot

Pilot 配方為 60-epoch cosine、batch 16、seed 42、AMP、upper-2 vision＋MIT q/v LoRA、`--stop-after-epoch 20`，最佳 epoch 為 20。以下數字由 `artifacts/patch16_resolution/whole_zoom_validation/view_predictions.pt` 重新計算確認（完整 12,960 筆 validation）。

| View | Correct | Accuracy | A042–A048 correct／504 |
| --- | ---: | ---: | ---: |
| Whole（單次推論） | 9,606 | 74.12% | 358 |
| Center zoom70（單次推論） | 10,235 | 78.97% | 383 |
| Whole＋zoom 機率平均（兩次推論） | 10,492 | 80.96% | 385 |

Pilot 逐 epoch whole accuracy 見 `docs/patch16-resolution-plan.md` 第 8 節（epoch 5 為 `70.68%`）。Pilot 每個 epoch 約 1,860 秒：訓練約 1,400–1,650 秒（約 50 examples/s），FP32 whole validation 約 470 秒。

## 3. 既有實作盤點

- `patch16_acceleration.py`、`patch16_anchor_cache.py`、`patch16_native_runtime.py`、`patch16_skeleton.py` 都只用 whole-view 的 `ProcessedUInt8Cache`；`patch16_skeleton.py` 的 whole＋zoom validation 透過 `rgb_input_audit.ViewDataset` 每次重新解碼原始影片。目前沒有任何 zoom／crop uint8 cache，需要新增。
- `crop_aware.py`（v1）寫死 patch32、warm start、只跑一個 4,096 筆 pass，且每個樣本現場解碼；只作為 view 混合方式的參考，不放大使用。
- 不修改 `processed_uint8_cache.py`：`patch16_skeleton.py`、`patch16_anchor_cache.py`、`patch16_native_runtime.py`、`patch16_control.py`、`native_attention.py`、`patch16_acceleration.py` 都把它的 source hash 寫進 run identity，修改會讓既有 run 無法 resume。也不修改 `skeletonclip_phase3_gate1.py`，保留 pilot 用 `--resume` 延長的可能。

## 4. 實作

新增單一 runner `src/vlm/training/patch16_crop_aware.py`，沿用 Stage 1 runner 的 `_model`、`_optimizer`、`_train_epoch`、`_snapshot`／`_restore` 與 checkpoint 工具。

- **配方**：直接從 pilot 的 `run_lock.json` 讀入整份 config 作為訓練參數，CLI 只能改 operational 設定（artifacts 目錄、zoom cache 目錄、workers、resume、smoke 用的 batch 上限、`--stop-after-epoch`，預設 20）。另檢查 split 與 winning config 的 SHA-256 與 pilot lock 相同。Fresh LoRA 初始化與 pilot 相同（同 seed、同建構順序）；run lock 記錄第一個 validation batch 的初始 logits hash 是否與 pilot 的 `786a79b8…` 相同，作為資訊而非硬性門檻（不同 driver／library 版本可能讓 hash 不同）。
- **View 選擇**：`sha256("patch16-crop-aware:{seed}:{epoch}:{sample_id}")` 第一個 byte 的最低位元決定 whole 或 center_zoom70，約 50／50，各 epoch 獨立、與標籤無關。每個 epoch 開始時把實際分配數寫進 `train.log`。
- **Shuffle 順序**：`EpochSampler` 用 DataLoader 自己的 generator 包住 `RandomSampler`，產生與 pilot `DataLoader(shuffle=True)` 完全相同的 generator draws，只在 index 上附加 epoch。因此與 pilot 的差別只有訓練 view；dropout、feature noise 的 RNG 消耗也逐 batch 相同。
- **Zoom cache**：`CenterZoomUInt8Cache` 繼承 `ProcessedUInt8Cache`，config hash 由 whole config hash 加上 `view: center_zoom70` 與裁切定義重新雜湊，因此和 whole cache 分開；whole cache 的 hash 不變，沿用既有 `cache/skeletonclip_phase3_rgb_uint8`。裁切與 `rgb_input_audit.ViewDataset` 完全相同：同一組 uniform 8 幀，取邊長為原圖短邊 70%（`round`）的中央正方形，偏移量為 floor，再送進同一個 processor。不預先建立，讀不到時才解碼並以 atomic write 寫入；命中與未命中都從同一個 uint8 陣列重建 pixel。預設目錄 `cache/patch16_crop_aware_zoom70_uint8`。
- **Validation**：每個 epoch 用單一 DataLoader 讀取兩個 view，在完整 12,960 筆 validation 上以 FP32（與 pilot 相同）分別推論，報告 whole、center_zoom70 與兩者機率平均的 accuracy、correct、macro-F1、NLL 與 A042–A048 correct。
- **產物**：`run_lock.json`、`train.log`、`progress.json`、`console.log`、`last_checkpoint.pt`（可 resume）、最後 epoch 的 `final_adapter.pt`、`final_view_predictions.pt`、`final_merged.pt` 與 `run_report.json`。結束時檢查 merge 與 merged RGB-only strict reload parity（`atol=rtol=1e-5`），並對照 pilot 的 `view_predictions.pt` 逐 view 計算 fixes／regressions 與判讀結果。
- 不新增 persistent tests；第 6 節的 CPU 檢查與 GPU smoke 已涵蓋主要契約。

## 5. 事先登記的判讀規則

開跑前登記，跑完後不得更改。判讀一律使用 epoch 20（最後一個 epoch）的 checkpoint，不做 epoch selection；pilot 的 epoch 20 同時是它的最後與最佳 epoch，因此兩邊條件對等。

- **提早停止（runner 自動執行）**：epoch 5 的 whole accuracy 低於 `69.68%`（pilot 同 epoch `70.68%` 減 1.0 point，約為 validation 標準誤的 2.5 倍）時停止，判為 no-go。
- **有希望（promising）**：whole＋zoom 平均 accuracy 至少 `81.16%`（pilot `80.96%`＋0.20 points，即至少 10,518 筆正確），且 A042–A048 的 whole＋zoom correct 至少 385。
- **No-go**：whole＋zoom 平均 accuracy 低於 pilot 的 `80.96%`。
- **Inconclusive**：其餘情況，例如平均提升但未達 0.20 points，或 A042–A048 退步。
- **另外報告（不作為門檻）**：whole 單次推論 accuracy 與 pilot 的 zoom 單次 `78.97%`、whole＋zoom `80.96%` 比較（是否把 zoom 收益學進單次推論）；zoom 單次 accuracy；各 view 相對 pilot 的 fixes／regressions；逐 epoch whole accuracy 與 pilot 曲線比較。

沒有另跑 control group：pilot 本身就是 matched control（相同配方、初始化與 shuffle 順序），差別只有訓練 view。只有 seed 42，結論只能視為單一 seed 的證據。

## 6. CPU 檢查（已通過）

以 `artifacts/patch16_resolution/attention_review/raw/` 中 25 支原始影片，對照已 commit 版本的 `rgb_input_audit.ViewDataset`：

- **裁切一致**：4 支影片的 zoom cache pixel 與 `ViewDataset` center_zoom70 完全相同（`torch.equal`，最大誤差 0）；whole cache 也與 `ViewDataset` whole 完全相同。
- **Cache identity 分開**：whole config hash 與未修改的 `ProcessedUInt8Cache` 相同（`742406b5…`），zoom 為 `88a7eac1…`，兩者路徑不重疊。
- **命中與未命中一致**：第二次讀取（命中）與第一次（未命中、寫入）pixel 完全相同，cache 檔案未被改寫，沒有殘留暫存檔。
- **Dataloader**：fork 模式、2 個 persistent workers、batch 4 跑 2 個 epoch。`EpochSampler` 的順序在 3 個 epoch 都與 `DataLoader(shuffle=True)` 相同；每個樣本的 pixel 等於其 hash 指定 view 的 cache 內容；validation batch shape 為 `[4, 2, 8, 3, 224, 224]`。以 80,640 個合成 ID 檢查，各 epoch zoom 比例為 50.0%–50.2%。

## 7. 指令

以下指令都在 repo 根目錄執行，cache 與 artifacts 使用 repo 內的相對路徑。需要已下載 patch16 與 patch32 的 Hugging Face snapshot（runner 使用 `local_files_only=True`），以及 pilot 的 `run_lock.json` 與 `whole_zoom_validation/view_predictions.pt`。

GPU smoke（50 個 training batches、10 個 validation batches、1 個 epoch）：

```bash
.venv/bin/python -m vlm.training.patch16_crop_aware --artifacts-dir artifacts/patch16_crop_aware/smoke --max-train-batches 50 --max-val-batches 10 --stop-after-epoch 1
```

Smoke 通過條件：`run_report.json` 存在、loss 有限、`run_lock.json` 的 `vlm_package_path` 指向預期的 source、trainable 參數與 pilot 相同（193,656）、merge／reload parity 通過、zoom cache 有新檔案。

完整訓練：

```bash
.venv/bin/python -m vlm.training.patch16_crop_aware --artifacts-dir artifacts/patch16_crop_aware/seed42
```

中斷後以同一指令重新啟動即可從 `last_checkpoint.pt` resume（identity 必須完全相同）。

## 8. 資源估算

- **GPU**：pilot 在 RTX 2080 Ti 上 peak memory 為 3.15 GB；validation 每個 batch 依序推論兩個 view，不增加 peak。需要一張沒有其他大型工作的 GPU，否則吞吐量會下降。
- **磁碟**：每支影片的 zoom cache 為 1.2 MB（`8×3×224×224` uint8）。Validation 12,960 筆在 epoch 1 全部寫入；每支 train 影片每個 epoch 有 50% 機率被分到 zoom，20 個 epoch 後幾乎全部寫入（從未被分到 zoom 的機率為 2⁻²⁰）。預估累計：epoch 1 後約 53,000 支（64 GB）、epoch 2 後約 73,000 支（88 GB）、epoch 4 後約 89,000 支（107 GB），最終約 93,600 支、113 GB（105 GiB）。Checkpoint 只有 LoRA 與 classifier，容量可忽略。啟動前要求 cache 所在檔案系統至少 150 GB 可用。
- **CPU／記憶體**：train 與 validation 各 24 個 persistent workers（與 pilot 相同）；validation batch 含兩個 view，prefetch 佔用的主機記憶體約為 pilot 的兩倍（約 4 GB）。單核（Apple silicon）解碼＋前處理一支 1080p 影片約 0.15 秒，24 workers 預估每秒可解碼數十支，高於 epoch 1 約每秒 25 支的未命中需求，因此訓練預期仍以 GPU 為瓶頸，但 CPU 負載較高時前幾個 epoch 可能變慢。
- **時間**：cache 填滿後每個 epoch 約為訓練 25–28 分鐘＋whole／zoom 兩次 FP32 validation 約 16 分鐘，約 42–44 分鐘；20 個 epoch 約 14–15 小時。Epoch 1–4 與第一次 validation 因 zoom 未命中需要解碼，預估額外 0.5–1.5 小時，總計約 15–16.5 小時。若在 epoch 5 提早停止，約 4 小時結束。

## 9. 查看狀態

```bash
cat artifacts/patch16_crop_aware/seed42/progress.json
grep -E "validation_completed|early_stopped" artifacts/patch16_crop_aware/seed42/train.log | tail -3
find cache/patch16_crop_aware_zoom70_uint8 -name "*.npy" | wc -l
```

## 10. 結果

尚未執行。
