# RGB interaction mixed4：middle2＋upper2

2026-10-02，使用者授權「try it」。訓練vision blocks6、7、10、11的self-attention q/v LoRA，保留既有10／11 adapters繼續更新；與upper4（8／9／10／11）相比，新增零B adapters的位置由8／9改為6／7。這輪從Phase 3 epoch13 baseline初始化，不接續任何已訓練local expert的更新。

- 明確選擇非連續blocks6／7／10／11；blocks8／9維持原pretrained線性層，不注入額外adapters。保留baseline blocks10／11、MIT q/v與classifier權重，新blocks6／7的B初始化為0，初始eval logits需與baseline一致。
- 可訓練vision blocks恰為6／7／10／11，MIT與classifier照原recipe訓練。可訓練參數數量需等於upper4的291,960；GPU smoke核對新增6／7及既有10／11的全部q/v gradients與tail adapter實際更新，避免重現middle4的尾端凍結。
- 同2,280筆canonical train／570steps／單pass、seed42、batch4、workers4，train selection與RGB interaction crops沿用既有鎖定資料；11 focus classes各128支，其他109類各8支。AdamW lr1e-5、weight_decay5e-4、cosine、CE smoothing0.1、gradient_clip1，dropout／feature noise沿用原recipe。
- 固定1,228筆診斷後接續完整retained86的8,640支validation；whole／zoom權重不更新，RGB定位metadata重用，所有120 outputs與197個fallback保留，三視角softmax機率各占三分之一。不讀test、不調crop或融合權重。
- 逐筆比較upper2（7,277／8,640）、upper4（7,305）及middle4（7,291），列出各86類、A042–A048、三個目標類、摸頸、診斷之外7,548支。高於7,305才記為相較upper4提升；沒有額外收益就不延長此設定，不自動替換部署模型。
- 相同seed、資料順序與recipe；同為兩個新增零B blocks加兩個既有tail blocks，較middle4更直接比較新增位置。Dropout作用位置會改變，不將單seed結果解釋為位置效果的定論。

CPU臨時檢查通過exact warm-start parity、train blocks6／7／10／11、尾端adapters更新、q/v gradients與merged RGB-only strict reload；既有14項LoRA tests通過，未新增persistent tests。GPU啟動前利用率0%、memory332MiB；既有tmux工作保留。

Tmux session `roi-mixed4` 依序執行GPU smoke、fresh pilot與完整evaluation，任一階段失敗停止；smoke更新不接續到pilot。Artifacts：`artifacts/hand_roi_mixed4_v1/`；source snapshots、init／selection／crop／reference hashes、target-module清單、實際step數、各階段log與progress獨立保存。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_upper4 --vision-layout mixed4 --artifacts-dir artifacts/hand_roi_mixed4_v1
```

重跑需fresh root；shared runners使用 `--vision-layout mixed4 --upper-blocks 4`。完整結果已寫入 `docs/hand-roi-mixed4-v1-results.md`，保留前輪所有報告。

GPU smoke完成16 train／16 validation、四steps，`status=smoke_pass`；whole warm-start max error 0，八個可訓練vision q/v gradients全部非零，blocks10／11八個adapter tensors全部更新（max-abs更新量最小者為 `2.5011599e-05`）。Merged strict reload max error `6.3180923e-06`；fresh pilot已自動接續。已核對train IDs順序與baseline／selection／crop／split／reference hashes均與upper4相同。

## Pilot結果

Pilot完成2,280筆／570steps，初始whole parity max error 0，merged strict reload max error `8.7618828e-06`；blocks10／11八個adapter tensors皆更新，最大絕對更新量的最小者為 `0.0010228343`。固定診斷retained86為832／1,092：相對upper2的827，修正6／退步1、net +5；相對upper4的840，修正1／退步9、net −8；相對middle4的825，修正10／退步3、net +7。

三個目標類130／216（upper2為128、upper4為131、middle4為130）；A042–A048為87／164（upper2為86、upper4為92、middle4為86）；摸頸20／72（upper2／middle4為20、upper4為22）。完整8,640支validation已完成，最終決策見下節。


## 完整結果與稽核

完整retained86為7,292／8,640（84.3981%），相對upper2修正31／退步16、net +15；相對upper4修正16／退步29、net −13；相對middle4修正31／退步30、net +1。三個目標類130／216、摸頸20／72、A042–A048為370／504；診斷之外7,548支為6,460，對upper4 net −5、對middle4 net −6。

獨立產物稽核325項全部通過：精確train blocks與291,960參數、尾端實際更新、初始化／資料／source hashes、570steps、完整8,640支及所有120 outputs、197個fallback、固定等權融合、三組逐筆比較、merged reload與成功exit code 0；沒有test inference。稽核入口與結果保存在 `artifacts/hand_roi_mixed4_v1/requirement_audit_snapshot.py` 與 `requirement_audit.json`。

依預先約定，相對upper4為 `no_additional_gain_stop`，不採用或延長此recipe。這是固定短訓練、seed42的一次比較，不能據此斷言中層位置普遍無效。完整逐類結果見 [mixed4結果](hand-roi-mixed4-v1-results.md)。
