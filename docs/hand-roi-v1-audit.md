# 手部與互動區域局部推理：v1 診斷

## 要驗證的問題

2026-10-01：使用者授權嘗試局部畫面方向。這輪先驗證：當手與互動部位在模型輸入中占更大範圍，既有 patch32 baseline 是否更容易分辨吃東西 A002、打噴嚏／咳嗽 A041、搧風 A049。摸頸 A047 作為次要觀察類別；剪指甲 A075 不在現有 paired skeleton manifest 的 A001–A059 範圍，尚未進行其局部診斷。

## 固定設計

- Checkpoint 使用原 baseline 的 upper2 vision q/v LoRA、MIT q/v LoRA 與 120 類分類頭；不使用 native-attention 訓練結果，不更新權重。
- 使用 canonical validation 與 paired manifest 的交集，選 A002、A003、A013、A019、A033、A037、A039、A041、A045、A047、A049，各取 sample-ID seed42 hash 排序前 16 筆，共 176 筆。選樣不依預測正確與否；其他八類用來觀察混淆與退步。這不是保留 86 類的代表性估計。
- 同一模型比較 whole、hands、interaction；原 center_zoom70 predictions 用於固定 whole＋zoom 基準。所有 views 都使用 full-video uniform8。
- Hands 由 wrist、hand、hand tip、thumb 的有效 color joints 定位；interaction 再包含頭、頸與上脊椎。取八幀合併的固定方形區域，避免逐幀追蹤裁切抵消相對移動；邊長為最大座標跨度加 0.6 倍 torso 長度，至少 1.2 倍 torso 或 128 原圖像素，並限制在畫面內。相同規則用於所有類別，不依 ground truth 決定裁切。
- 有效手部幀少於四幀或缺少 torso 尺度時，兩種局部 view 回退到原圖，並記錄原因；不能默默移除難例。
- 除局部單獨推理外，分別以 whole、zoom、hands 或 interaction 三個 views 等權機率平均；不搜尋混合權重，不逐樣本挑最佳 view。
- Whole 重新推理必須與 saved baseline top1 一致，logit 最大差異不超過 1e-3；記錄每筆索引、裁切座標、來源 SHA256 與 parity 誤差。保留所有 120 個輸出；排除類別仍作混淆負例。

## 定位資訊與限制

這輪使用 skeleton 座標作為診斷用的定位資訊，目的是先測「位置已知時局部畫面有沒有幫助」。它不能直接作為 RGB-only 部署方案，也不是該方案可達準確率的嚴格上界。若有效，下一步需用 RGB 定位重現並評估定位誤差；若無效，僅能否定這套裁切配合既有分類器的直接收益，不能否定經訓練的局部分支。

執行時發現既有 `patch16-pilot` 正在訓練，因此本輪使用 batch1、torch／OMP／MKL threads2、逐樣本推理與解碼，不停止或修改既有工作。11 筆 smoke 先驗證執行與 baseline parity，再跑 176 筆診斷；不新增 persistent tests，不讀 test split。

## 執行

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_audit --per-class 1 --artifacts-dir artifacts/hand_roi_smoke_v1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_audit --per-class 16 --artifacts-dir artifacts/hand_roi_audit_v1
```

Runner：`src/vlm/training/hand_roi_audit.py`。Artifacts 包含 `experiment_lock.json`、`runner_snapshot.py`、`crop_manifest.json`、`view_predictions.pt`、`report.json`，以及事先選定每類第一筆的實際輸入影格拼圖。

## 176 筆診斷結果

| View | 目標三類正確／48 | 混淆類別正確／128 | 全部正確／176 | 目標修正／退步 |
| --- | --- | --- | --- | --- |
| whole_zoom_mean | 19 | 85 | 104 | 0 / 0 |
| hands | 24 | 86 | 110 | 13 / 8 |
| interaction | 27 | 85 | 112 | 14 / 6 |
| whole_zoom_hands_mean | 22 | 95 | 117 | 6 / 3 |
| whole_zoom_interaction_mean | 25 | 100 | 125 | 8 / 2 |

| 類別 | Whole＋zoom | Hands | Interaction | Whole＋zoom＋hands | Whole＋zoom＋interaction |
| --- | --- | --- | --- | --- | --- |
| A002 eat meal/snack | 8 / 16 | 10 / 16 | 11 / 16 | 8 / 16 | 10 / 16 |
| A041 sneeze/cough | 5 / 16 | 7 / 16 | 9 / 16 | 8 / 16 | 9 / 16 |
| A049 use a fan (with hand or paper)/feeling warm | 6 / 16 | 7 / 16 | 7 / 16 | 6 / 16 | 6 / 16 |
| A047 touch neck (neckache) | 4 / 16 | 3 / 16 | 2 / 16 | 4 / 16 | 4 / 16 |

Fallback 共 1 筆；whole logit 最大 parity 誤差為 2.8073788e-05。診斷推理耗時 360.4 秒（不含模型載入）；GPU 與既有 patch16 訓練共用。

### 混淆類別流向目標類的數量

| View | 非目標類被預測為吃東西／咳嗽／搧風 |
| --- | --- |
| whole_zoom_mean | 5 / 128 |
| hands | 3 / 128 |
| interaction | 3 / 128 |
| whole_zoom_hands_mean | 2 / 128 |
| whole_zoom_interaction_mean | 4 / 128 |

這是 top1 的負例混淆數，不是部署警報誤報率。

[實際輸入對照頁](../artifacts/hand_roi_audit_v1/index.html)。

## 第一階段判讀與 RGB 定位確認

使用 privileged 定位時，固定 whole＋zoom＋interaction 的目標三類由 19／48 提升至 25／48（修正 8、退步 2）；混淆類別由 85／128 提升至 100／128（修正 17、退步 2）。收益主要來自吃東西 8→10、咳嗽 5→9；搧風 6→6、摸頸 4→4。這支持繼續驗證局部互動資訊，但不是手部 attention 已有效的證據，也不能宣稱四類都改善。

第二階段以相同 176 筆與相同幀索引，使用 [MediaPipe Pose Landmarker 官方 Python API](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) 的 lite 模型，僅由 RGB 估計 wrist／finger、頭部、肩與髖座標。採 IMAGE mode、num_poses1、detection／presence0.5、landmark visibility／presence0.5；以肩中點到髖中點作 torso 尺度，其餘方形區域與 margin 規則相同。這是診斷的定位替代，沒有 threshold／margin 搜尋，沒有使用 NTU skeleton 座標作裁切。配對 manifest 只延用第一階段的樣本選擇與標籤。

定位工具隔離安裝於 `/tmp/vlm-mediapipe`，不修改原 `.venv` 或既有 patch16 訓練環境；模型下載自 Google 官方 storage，SHA256 鎖在 localization manifest。依賴為 `mediapipe==0.10.32`、`absl-py==2.5.0`、`flatbuffers==25.12.19`、`sounddevice==0.5.6`、`cffi==2.1.1`、`pycparser==3.0`、`protobuf==7.36.2`，沿用原環境的 numpy／cv2／matplotlib。

```bash
PYTHONPATH=/tmp/vlm-mediapipe MPLCONFIGDIR=/tmp/vlm-native-matplotlib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.rgb_hand_roi_localization --artifacts-dir artifacts/rgb_hand_roi_localization_v1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_audit --per-class 16 --rgb-crop-manifest artifacts/rgb_hand_roi_localization_v1/crop_manifest.json --artifacts-dir artifacts/rgb_hand_roi_audit_v1
```

CPU 定位結果與有效座標保存在 `artifacts/rgb_hand_roi_localization_v1/`，分類推理結果保存在 `artifacts/rgb_hand_roi_audit_v1/`。定位失敗回退到 whole，不排除失敗樣本；原圖 hash 或索引不一致則終止，不當成定位失敗處理。
