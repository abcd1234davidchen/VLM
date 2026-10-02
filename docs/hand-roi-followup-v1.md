# RGB 局部互動：擴大確認與短訓練

2026-10-01：使用者授權接著進行。先擴大固定評估，若新增樣本的目標收益與保留類別收益方向仍成立，再做獨立局部 expert 的短訓練。

## 固定評估設計

- A002、A003、A013、A019、A033、A037、A039、A041、A045、A047、A049 各使用完整 canonical validation 72 筆，其他 109 類各使用 seed42 sample-ID hash 排序前四筆，共 1,228 筆；其中原先看過 176 筆，新增 1,052 筆。選樣不依模型預測。
- 主要看新增樣本中的 A002／A041／A049 與保留 86 類；同時列出其他保留類別、排除 34 類負例，以及 A042–A048。這是特定類別加密取樣，不能當成 full validation 準確率；新增樣本仍屬開發用 validation，不是獨立 test。
- 使用同一個 MediaPipe lite 模型、相同 confidence0.5、相同手＋頭／肩 interaction 裁切與 torso margin。沿用第一輪可核對來源與幾何的 176 筆定位、局部 logits，其餘新推理；不搜尋裁切比例、融合權重或類別專屬 view。
- 比較 whole＋zoom 與固定 whole＋zoom＋interaction 等權機率平均。Whole／zoom 仍是原 baseline 全量保存結果，所有 120 類輸出都保留。定位失敗回退 whole，樣本仍列入評估。
- 短訓練 gate：新增樣本的三個目標類別 net gain > 0，且新增樣本的保留 86 類 net gain >= 0。這是探索性繼續條件，不是統計顯著或採用門檻；未通過就不啟動本輪訓練。

## 短訓練設計

- 訓練樣本依獨立 `hand-roi-train42` hash 選擇：上述 11 類各 128 筆，其他 109 類各八筆，共 2,280 筆 canonical train；排除類別保留作負例，train／validation IDs 不重疊。
- 另外複製原 baseline，訓練局部 interaction expert 的 upper2 vision q/v LoRA、MIT q/v LoRA 與 120 類分類頭。原 whole／zoom checkpoint 不更新，推理時固定加入局部 expert 的機率。
- 固定 seed42、單 pass、batch4、AdamW lr1e-5／weight_decay5e-4、cosine、CE label_smoothing0.1、gradient_clip1。沿用既有 dropout／feature noise，不使用 skeleton 或新的 attention loss。
- 執行前確認 baseline whole warm-start parity；首步確認 vision／MIT／classifier gradients，訓練後保存局部 adapter 及 merged RGB state，驗證 strict reload parity。這是短迭代，不新增 persistent tests，不讀 test。
- 與同批 expanded audit 的未訓練 interaction 比較，並與原 whole＋zoom 比較。另列出新增 validation 與先前 176 筆，不能用先前小樣本收益掩蓋新樣本退步。

## 執行與產物

Artifacts 位於 `artifacts/hand_roi_followup_v1/`；選樣先固定為 `val_selection.json`／`train_selection.json`。RGB 定位採四個 CPU workers，原 `.venv` 不更動，MediaPipe 仍隔離於 `/tmp/vlm-mediapipe`。既有 patch16 工作保持運作，GPU 評估／訓練 batch 上限四。Runner 為 `src/vlm/training/rgb_hand_roi_localization.py` 與 `src/vlm/training/hand_roi_followup.py`，各 run 保存 source snapshots、hashes、progress、crop metadata 與 predictions。

## 擴大評估結果

| 範圍 | 樣本 | 原 whole＋zoom 正確 | 加入 interaction 正確 | 修正／退步 |
| --- | --- | --- | --- | --- |
| 全部選樣：三個目標類 | 216 | 105 | 124 | 25 / 6 |
| 全部選樣：保留類別 | 1092 | 743 | 806 | 88 / 25 |
| 全部選樣：其他保留類別 | 300 | 236 | 245 | 15 / 6 |
| 全部選樣：A042–A048 | 164 | 87 | 85 | 8 / 10 |
| 新增樣本：三個目標類 | 168 | 86 | 101 | 19 / 4 |
| 新增樣本：保留類別 | 916 | 639 | 685 | 67 / 21 |
| 新增樣本：其他保留類別 | 300 | 236 | 245 | 15 / 6 |
| 新增樣本：A042–A048 | 132 | 73 | 70 | 6 / 9 |

| 類別 | 原 whole＋zoom | 加入 interaction |
| --- | --- | --- |
| A002 | 35 / 72 | 44 / 72 |
| A041 | 37 / 72 | 46 / 72 |
| A049 | 33 / 72 | 34 / 72 |
| A047 | 24 / 72 | 20 / 72 |
| A045 | 47 / 72 | 49 / 72 |

新增目標類別 net +15、保留類別 net +46，通過預先固定的短訓練 gate。其他保留類別 net +9，但 A042–A048 新增樣本 net −3，其中摸頸完整 72 筆由 24→20；不能以整體收益掩蓋這個退步。當前不替換 baseline。

Validation RGB 定位 fallback 18／1,228；全部仍計入評估。Warm-start whole parity max error 0。176 筆既有 interaction logits 在確認原圖 SHA256、幀索引、裁切座標與 fallback 均一致後重用。

短訓練將檢查局部 expert 能否進一步改善吃東西／咳嗽，同時列出摸頸及其他保留動作的退步；不會依 validation ground truth 選擇 view。

## 功能 smoke

先使用 16 筆 train 裁切、16 筆 validation 執行四個 optimizer steps；smoke 只確認 warm-start、梯度、更新與 merged reload，不據其分類數字選模型。Vision／MIT／classifier gradients 均非零，whole parity max error 為 0.0，merged strict reload max error 為 3.0994415283203125e-06，`status=smoke_pass`。Smoke 使用獨立 artifact 目錄，不將其更新接續到 pilot；pilot 重新從 baseline 開始。

## 比較訓練是否增值

短 pilot 的主要比較是「訓練後局部 expert 的固定合併」對「未訓練局部 view 的固定合併」，而非只對 whole baseline。列出三個目標類、所有保留類與 A042–A048 的 before／after、修正／退步，也保留與原 whole＋zoom 的比較。若訓練沒有增加收益，就保留未訓練 RGB 局部 view 的研究結果，不延長這個 recipe；即使短 pilot 改善，也需完整產品評估才能決定是否採用。

[新增樣本修正／退步影格對照](../artifacts/hand_roi_followup_v1/case_review/index.html) 包含每個目標類與摸頸各一筆修正、一筆退步；事後依 sample-ID 排序選第一筆，只用於診斷，不估計準確率。摸頸退步例子的手與頸接觸在局部八幀仍可見，模型卻由摸頸轉成擦臉；僅能說明取樣沒有完全漏掉接觸，尚不能推論 attention 或分類器的因果問題。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_followup --mode audit --artifacts-dir artifacts/hand_roi_followup_v1/expanded_audit
PYTHONPATH=/tmp/vlm-mediapipe MPLCONFIGDIR=/tmp/vlm-native-matplotlib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vlm.training.rgb_hand_roi_localization --selection-lock artifacts/hand_roi_followup_v1/train_selection.json --artifacts-dir artifacts/hand_roi_followup_v1/train_localization --workers 4
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_followup --mode smoke --train-selection artifacts/hand_roi_followup_v1/smoke_train_selection.json --train-crops artifacts/hand_roi_followup_v1/smoke_train_crops.json --workers 2 --artifacts-dir artifacts/hand_roi_followup_v1/smoke
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_followup --mode pilot --smoke-report artifacts/hand_roi_followup_v1/smoke_v2/report.json --artifacts-dir artifacts/hand_roi_followup_v1/pilot
```

修正 checkpoint identity 後的同版 smoke 位於 `smoke_v2`，再次通過 gradients 與 merged reload；pilot 使用該 report 驗證 runner hash。新 local adapter 的 identity 記錄實際 train／validation 數量與 init adapter hash，原 baseline identity 另存，不沿用其 80,640 筆資料數作為本輪 train 數。

## 短 pilot 結果

| 範圍 | 原 whole＋zoom | 未訓練 interaction 合併 | 訓練後 interaction 合併 |
| --- | --- | --- | --- |
| 全部選樣：三個目標類（n=216） | 105 | 124 | 128 |
| 全部選樣：保留類別（n=1092） | 743 | 806 | 827 |
| 全部選樣：其他保留類別（n=300） | 236 | 245 | 249 |
| 全部選樣：A042–A048（n=164） | 87 | 85 | 86 |
| 新增樣本：三個目標類（n=168） | 86 | 101 | 105 |
| 新增樣本：保留類別（n=916） | 639 | 685 | 703 |
| 新增樣本：其他保留類別（n=300） | 236 | 245 | 249 |
| 新增樣本：A042–A048（n=132） | 73 | 70 | 71 |

| 類別 | 原 whole＋zoom | 未訓練局部合併 | 訓練後局部合併 |
| --- | --- | --- | --- |
| A002 | 35 / 72 | 44 / 72 | 45 / 72 |
| A041 | 37 / 72 | 46 / 72 | 49 / 72 |
| A049 | 33 / 72 | 34 / 72 | 34 / 72 |
| A047 | 24 / 72 | 20 / 72 | 20 / 72 |
| A045 | 47 / 72 | 49 / 72 | 51 / 72 |
| A075 | 0 / 4 | 2 / 4 | 3 / 4 |

### 訓練相對於未訓練局部分支的增值

| 範圍 | 訓練前正確 | 訓練後正確 | 修正／退步 |
| --- | --- | --- | --- |
| all（n=1228） | 868 | 894 | 30 / 4 |
| new_samples（n=1052） | 747 | 770 | 27 / 4 |
| retained86（n=1092） | 806 | 827 | 25 / 4 |
| targets（n=216） | 124 | 128 | 5 / 1 |
| attention_A042_A048（n=164） | 85 | 86 | 3 / 2 |

Pilot 實際更新 2,280 筆 train、570 steps、單 pass；首步 gradient families 為 `{'vision': 4890.933258056641, 'mit': 888.1595764160156, 'classifier': 194.78873765468597}`，merged strict reload max error 為 3.1292439e-06。Baseline whole／zoom checkpoint 不更新。局部 adapter 在 `artifacts/hand_roi_followup_v1/pilot/local_adapter.pt`，merged state 在 `merged_local_rgb_state.pt`；沒有讀 test。

這份選樣在難類上加密，其他類別只有四筆，不能據其總分宣稱保留 86 類 full validation 的準確率。先前 176 筆與新增 1,052 筆均分開報告；新增仍是開發 validation，沒有把它當獨立 test。

## 最終判讀（2026-10-02 整理）

擴大評估的局部輸入收益重現，短訓練也有額外收益：保留類別同批 1,092 筆由原 whole＋zoom 743 correct，到未訓練局部合併 806，再到訓練後 827；訓練本身相對未訓練局部合併修正 25、退步 4，net +21。全 1,228 筆相同比較為 868→894，修正 30、退步 4，net +26。新增樣本中的保留類別由 639→685→703／916，短訓練額外 net +18，收益沒有只集中在先前 176 筆。

完整 72 筆的吃東西由 35→44→45、咳嗽由 37→46→49，支持繼續驗證這兩類；搧風由 33→34→34，收益有限。摸頸由 24→20→20，短訓練沒有救回退步。A042–A048 的選樣由 87→85→86／164，仍比原 baseline 少一筆；新增部分由 73→70→71／132，仍少兩筆。不能把這輪當成所有關注動作都改善。

目前保留原 baseline，不採用這個候選替換。下一個採用前的步驟是完整 validation 的固定 RGB interaction 比較，確認 86 類產品範圍的整體收益與各類退步；摸頸應獨立處理其接觸位置／分類邊界，而非延長本次單 pass recipe 或繼續只增加手部 attention。這輪沒有驗證任何 attention supervision 的增益，也沒有啟用警報或讀 test。

## 完整 86 類確認（2026-10-02 啟動）

依使用者授權，啟動 `artifacts/hand_roi_full86_v1/`：固定使用本輪 pilot 的 local adapter，對 canonical validation 中所有保留 86 類的 8,640 支影片評估。沿用 RGB interaction 定位、原 whole／zoom logits 與等權機率融合，不追加訓練、不調權重。快取的定位與 local predictions 需核對 source hash／crop geometry；另外列出先前 1,228 筆診斷樣本以外的結果。全部 120 logits 保留，偵測失敗 fallback 不剔除。

Runner 為 `src/vlm/training/hand_roi_full_validation.py`，定位完成後自動接續 GPU inference；完成時寫出 `evaluation/report.json`、`view_predictions.pt` 與 `docs/hand-roi-full86-v1-results.md`。評估不自動替換模型。

完整評估已完成：保留 86 類 79.9769% → 84.2245%（+4.2477 個百分點），修正 505、退步 138、淨增 367；先前診斷之外的 7,548 支由 81.7038% → 85.4531%，淨增 283。吃東西／咳嗽收益保留，摸頸仍退步。詳見 [完整結果](hand-roi-full86-v1-results.md)。
