# RGB accuracy improvement series

2026-10-02使用者授權依序實作並執行全部方向。以upper4的7,305／8,640（84.5486%）為基準，固定RGB-only、120 outputs、canonical retained86與原whole／zoom checkpoint，不讀test、不自動部署。

- Validation的18位participants依 `SHA256("roi-series42:Pnnn")` 排序，前9位calibration（3,096筆），後9位confirmation（5,544筆），共用固定名單。設定與候選只按calibration選擇；兩組都曾用於開發，不視為獨立test。比較accuracy、NLL、逐筆修正／退步、86類、targets、A042–A048、摸頸與診斷之外7,548筆。
- 第一輪沿用upper4 logits，逐視角搜尋temperature `{0.5,0.75,1,1.5,2}` 的最低calibration NLL，再搜尋間隔0.1的simplex融合權重，另包含等權配置。候選依calibration答對數、NLL與較低forward成本選擇，只有accuracy嚴格提升才取代原等權。
- 第二輪從Phase3 baseline重新初始化upper4 blocks8–11，9,120筆／單pass／batch4／2,280steps。保留原2,280筆，11 focus classes各512筆、其餘109類各32筆；新增影片依 `SHA256("roi-series-train42:<sample_id>")` 補足，selection不使用predictions。因舊manifest未記錄原hash salt，直接保留所有舊IDs與其類別內順序，將新增salt明確記錄，不臆測舊排序。Rank16、291,960參數、lr1e-5、cosine、weight_decay5e-4、smoothing0.1、clip1與dropout／noise維持原recipe；新增RGB localization與舊2,280筆crop共用原幾何。
- 第三輪用calibration選出的local expert，獨立測temporal及context。Temporal取16個uniform positions，偶數／奇數各八幀，沿用原clip-static crop，先平均兩次local softmax機率再融合。Context使用原八幀，interaction square邊長1.25倍，上限影片短邊並在邊界內平移；fallback保留whole frames。兩項各自提升才另測組合。
- 融合設定在第一輪後鎖定，第二／三輪另報等權分數。最後只按calibration選input，再作一次融合refit；沒有嚴格accuracy收益就保留原設定。Confirmation也提升才記為confirmed gain；完整分數不參與搜尋。某方向沒有收益仍繼續下一方向，只有執行錯誤停止。

獨立入口保留既有570-step runners與歷史產物。Source snapshots、鎖定SHA256、selection、input geometry、逐輪logits、模型與stage完成記錄保存於 `artifacts/hand_roi_improvement_series_v1/`；`--resume` 驗證來源與已完成產物hash，未完成階段按同一鎖定設定重跑。GPU工作在tmux `roi-improve` 中循序執行，batch4、workers4、CPU threads2；smoke使用16筆／4steps，不接續到正式訓練。

```bash
PYTHONPATH=/tmp/vlm-mediapipe OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/vlm-series-matplotlib .venv/bin/python -m vlm.training.hand_roi_improvement_series --artifacts-dir artifacts/hand_roi_improvement_series_v1
```

七項新合約tests覆蓋participant隔離、confirmation不可影響選擇、temporal機率融合、非法設定、短影片／crop邊界、精確train quotas與完成產物變更拒絕；既有14項LoRA tests通過。實際影片的原輸入／temporal／context／combined四種dataset shape與geometry已確認。完成後runner在CPU重新計算metrics與候選選擇，稽核訓練budget、精確blocks、gradients、初始化／merged reload、尾端更新、fallback與所有120 outputs，再生成結果報告；可用 `--audit-only --resume` 重跑稽核。MediaPipe沿用舊試驗的隔離環境 `/tmp/vlm-mediapipe`（0.10.32），透過PYTHONPATH載入，不修改主 `.venv`。


## 已完成

融合→新9,120筆expert→temporal→context→combined→final refit依序完成，所有GPU stages與117項CPU產物稽核通過，最終exit code 0。另以獨立NumPy重建融合，93個群組metrics全部一致。正式calibration選出的final為85.4630%（7,384／8,640；對upper4 net +79，confirmation +25）；組合固定融合的全量最高分為85.5440%，沒有用它反向修改選擇規則。逐輪表、群組退步與refit限制見 [完整結果](hand-roi-improvement-series-v1-results.md)。
