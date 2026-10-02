# RGB interaction middle4：blocks6–9 位置比較

2026-10-02，使用者授權「try it out」。比較相同四個 vision blocks 的可訓練容量，將 upper4 的 blocks8–11 改為 blocks6–9；rank16、MIT／classifier、資料與570-step預算保持原設定。從 Phase 3 epoch13 baseline 初始化，不接續 upper2 或 upper4 local expert 的訓練更新。

- 訓練 blocks6、7、8、9 的 self-attention q/v LoRA；既有 blocks10、11 的 baseline adapters 保留並凍結，不納入 optimizer。新增八個 q/v targets 的 B 初始化為0，初始 eval 函數與 baseline 一致。
- Adapter checkpoint 因保留 baseline blocks10／11，包含 blocks6–11 共六個 blocks，但實際可訓練 vision blocks 只有四個。MIT q/v LoRA 與 classifier 仍訓練；runner 核對 trainable parameter count 等於上一輪 upper4，並在訓練結束核對 blocks10／11 adapters 完全不變。
- Train selection／RGB interaction crops 沿用2,280支；11 focus classes各128支，其他109類各8支。Seed42、batch4、workers4、單pass／570 steps，AdamW lr1e-5、weight_decay5e-4、cosine、CE smoothing0.1、clip1；crop、rank、融合權重沿用原設定。
- 固定1,228支診斷完成後，接續完整retained86的8,640支validation。保留120 outputs、全部fallback、whole／zoom baseline與三視角等權softmax機率平均；重用RGB定位metadata，不讀test。
- 分別與 upper2（7,277／8,640＝84.2245%）及 upper4（7,305／8,640＝84.5486%）逐筆比較，列出各86類、三個目標類、A042–A048、摸頸，以及診斷之外7,548支。高於7,305才記為相較前輪配置提升；若沒有額外收益，不延長這個設定。不自動替換部署模型。
- 新增 adapters 的初始化與 dropout 會改變 RNG 消耗；相同seed不代表逐步相同的dropout／feature-noise realization。記錄這項差異，不把結果全部解釋為 block位置的因果效果。

CPU 臨時檢查已通過：exact warm-start parity、可訓練 vision blocks 恰為6–9、新增八個q/v gradients、optimizer step後blocks10／11全部權重不變，以及 merged RGB-only strict reload。既有14項LoRA tests通過，沒有新增persistent tests。

GPU主機為RTX2080Ti，啟動前利用率0%、memory332MiB。Tmux session `roi-middle4` 依序執行短GPU smoke、pilot與full evaluation；任何階段失敗立即停止，smoke更新不接續到pilot。Artifacts：`artifacts/hand_roi_middle4_v1/`；source snapshots、進度與stage logs獨立保留。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_upper4 --vision-layout middle4 --artifacts-dir artifacts/hand_roi_middle4_v1
```

需使用fresh root；原upper4入口與預設配置維持可用。新配置為shared runners的 `--vision-layout middle4 --upper-blocks 4`；此處的4代表可訓練block數，實際checkpoint另保留兩個凍結baseline blocks。完整結果輸出為 `docs/hand-roi-middle4-v1-results.md`，不覆蓋前輪報告。

GPU smoke已通過：16 train／16 validation、四steps，warm-start max error 0，新增八個q/v gradients全部非零，凍結adapters parity通過，merged strict reload max error `6.4373016e-06`。可訓練參數291,960，與upper4相同；pilot已重新從Phase 3 baseline接續啟動。

## Pilot 結果與全量評估

正式pilot完成2,280筆／570steps；warm-start max error 0，merged strict reload max error `6.3180923e-06`。保存checkpoint中的blocks10／11八個adapter tensors與Phase 3 baseline逐值完全相同。相較upper2，診斷retained86為827→825／1,092，修正5／退步7、net −2；相較upper4為840→825，修正1／退步16、net −15。

三個目標類130／216（upper2為128、upper4為131）；A042–A048為86／164（upper2為86、upper4為92）；摸頸20／72（upper2為20、upper4為22）。診斷方向較差，不據小樣本決定完整產品範圍分數；完整8,640支evaluation已按固定規格完成，退出碼0，不延長訓練或改crop／融合。

比較也包含既有尾端adapters是否繼續更新的差異：upper4會更新已由Phase 3學到的blocks10／11，middle4的四個訓練blocks全部由新增零B開始，原blocks10／11則凍結。相同參數量、初始eval函數與570steps，不保證相同的短訓練學習效率；結果應解讀為這兩個具體配置的比較。

使用者追問middle2＋upper2的配置。若要更直接比較新增LoRA的位置，訓練blocks6／7／10／11、保留10／11繼續更新，會比本輪凍結10／11的設計更接近upper4（8／9／10／11）的條件。後續已按此配置完成mixed4，見 [mixed4結果](hand-roi-mixed4-v1-results.md)；本輪結果仍按原middle4配置解讀。

## 完整結果與結論

Middle4完整retained86為7,291／8,640（84.3866%）。相對upper2的7,277，修正44／退步30、net +14（+0.1620 percentage points）；相對upper4的7,305，修正34／退步48、net −14（−0.1620 points）。未達「高於7,305」的前輪配置提升判準，不推薦採用middle4、不延長本輪設定，保留現有候選。

診斷之外7,548支為6,466 correct：對upper2的6,450，修正39／退步23、net +16；對upper4的6,465，修正33／退步32、net +1。對upper4的整體−14由先前診斷−15與診斷之外+1組成。A042–A048為375／504（upper2為371、upper4為372），但摸頸仍20／72，低於upper4的22與原whole＋zoom的24。

原whole＋zoom為6,910，本輪7,291，修正518／退步137、net +381（+4.4097 points）；這個累積收益主要已由既有local expert提供，不能當成middle4新增收益。相對upper4，86類中18類改善、40類持平、28類退步；相對upper2，23類改善、44類持平、19類退步。

全部197個localization fallbacks計分，checkpoint／cached prediction parity max error 0，來源hashes在完成時核對。獨立稽核核對所有86類及兩組逐筆比較、三個目標類、A042–A048、摸頸、診斷之外7,548支、固定三視角平均、120 outputs、baseline來源、同資料／recipe／參數量，以及blocks10／11精確凍結；產物為 `artifacts/hand_roi_middle4_v1/requirement_audit.json` 與可重跑的 `requirement_audit_snapshot.py`。

完整逐類表與重現入口見 [middle4結果](hand-roi-middle4-v1-results.md)。本輪只驗證指定的middle4／freeze設定；middle2＋upper2（6／7／10／11）後續已另行完成mixed4，沒有以middle4結果推論其效果。
