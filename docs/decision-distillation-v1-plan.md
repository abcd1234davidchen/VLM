# Direct decision distillation with RGB retention

## 依據與範圍

使用者在2026-09-30分析SKI-style semantic pilot後同意嘗試改版。Semantic pilot是supervised120／partial59的本地改版，不是官方SKI zero-shot reproduction。它更新的RGB features對paired有收益，卻傷害部分unpaired；語意head並未直接用於推論，因此不能以該pilot否定論文。

唯讀診斷將既有features與classifier交叉搭配，在全部12,960 validation samples上的net correct counts如下。Baseline與完整新模型的argmax均與原saved predictions完全一致；batch-size FP32 logits的小差異已核對，不影響prediction counts。

| 搭配 | Paired net | Unpaired net | Overall net |
| --- | ---: | ---: | ---: |
| Original features + updated classifier | +1 | +2 | +3 |
| Updated features + original classifier | +10 | -17 | -7 |
| Updated features + updated classifier | +19 | -11 | +8 |

不能因此凍結classifier：在updated features下它補回部分損失。A061–A120的201個regressions中，192個仍是這60類彼此混淆，只有9個流向A001–A059。相同472-sample held-out subset上，current skeleton teacher為84.75%、main RGB120 classifier為67.16%、semantic59 head為61.23%。Final checkpoint的32個train samples局部feature-gradient檢查顯示，加權skeleton MSE norm約為主CE的8–20%，額外semantic CE則約80–145%；這不是整段訓練的gradient audit，不能作因果歸因。

## 固定實驗設計

- 不再訓練或部署額外semantic head；直接使用main120 logits的A001–A059 slice作conditional59 distribution。GT的120-class CE仍涵蓋所有RGB樣本，不把全局120 distribution誤當成teacher59 distribution。
- Skeleton teacher沿用`artifacts/skeletonclip_gate3/xclip_seed42/best_checkpoint.pt`，全部固定並eval。這是frozen direct-decision KD，不是SKI官方online collaborative co-training，也不是其text-conditioned MSE的重現。
- Frozen RGB anchor是original Phase3 warm-start model的獨立deep copy，只在unpaired train samples上提供120-class軟分布。沒有第二個可訓練RGB模型、沒有額外control group。
- Loss為`CE120(label smoothing .1) + .5*T^2*KL(skeleton59 || main_logits[:59])`（paired）加`T^2*KL(original_RGB120 || main_logits120)`（unpaired），固定`T=2`。KD與anchor分別在各自非空subset取sample mean；empty subset提供differentiable zero。Target scores全部detach。
- Student的vision／MIT LoRA與120-class classifier更新，base weights不解凍。Student使用eval mode停用dropout／feature noise，但保留梯度與optimizer更新，使anchor約束對應deterministic inference行為。這也是相對semantic pilot的改動，不能把差異只歸因於KD。
- 共享original baseline、seed42、seeded sample-ID hash與120-class equal-class round robin。固定1 epoch／8,192 train／batch4／fp32／workers4且hard cap24。AdamW LR`2.25e-5`、weight decay`5e-4`、cosine到0、gradient clip1；不搜尋weights或epochs，不暖啟動自上一個inconclusive pilot。
- 全部12,960 canonical validation，比對original baseline，另報paired/unpaired、per-class confusion與unpaired validation KL at T2。Promising須overall accuracy至少`+0.20` percentage points，且unpaired accuracy不下降；否則no-go或inconclusive，不自動放大budget。單一pilot不是winner確認，也不能隔離CE、KD、anchor各自貢獻。
- 只評估120類、不訓練106類、不讀test。Inference與merged artifact僅保留原RGB120 model，不需要skeleton或anchor，沒有新增推論分支。

## Smoke與dispatch

Smoke使用128 train／120 val，驗證initial RGB logit parity、initial anchor KL接近0、pure teacher KD確實到達classifier／vision／MIT、finite gradients、frozen teacher與anchor權重／梯度不變、RGB-only merged strict reload。Smoke只certify機制，不用其accuracy聲稱收益。Pilot必須通過相同source hashes的smoke。

Runner為`src/vlm/training/decision_distillation.py`；immutable lock與runner snapshot、predictions、checkpoint、merged RGB state均放在獨立`artifacts/decision_distillation_v1/`。Dispatcher用tmux並保存`dispatch.json`、`train.log`、`console.log`與terminal `dispatch_status.json`。Existing run directory不可覆寫。

```bash
.venv/bin/python -m vlm.training.spatial_dispatch --artifacts-dir artifacts/decision_distillation_v1/smoke --tmux-session decision-kd-smoke -- .venv/bin/python -m vlm.training.decision_distillation --mode smoke --artifacts-dir artifacts/decision_distillation_v1/smoke
.venv/bin/python -m vlm.training.spatial_dispatch --artifacts-dir artifacts/decision_distillation_v1/pilot_seed42 --tmux-session decision-kd-pilot -- .venv/bin/python -m vlm.training.decision_distillation --mode pilot --artifacts-dir artifacts/decision_distillation_v1/pilot_seed42
```

## 執行紀錄

Smoke已通過：pure KD的classifier／vision／MIT梯度皆非零且finite，frozen teacher／anchor unchanged為true，initial baseline logit error為`1.86e-5`，RGB-only merged reload error為`4.29e-6`。128 train／120 val smoke的主分類gain為0，只certify實作與梯度機制，不聲稱分類收益。KL target detach／identity／empty mask與tiny-model KD梯度inline檢查通過，既有vision／skeleton model tests共7個通過；未新增persistent tests。

Pilot已在tmux session `decision-kd-pilot`啟動，training PID `521201`、logger PID `521204`，詳見run的`dispatch.json`。Train／validation IDs與上一個semantic pilot完全相同，runner snapshot hash符合immutable lock，source hashes符合passing smoke。所有terminal結果以`run_report.json`與`dispatch_status.json`為準；不能以training loss下降預告勝出。

## Pilot結果：inconclusive

Tmux pilot已完成，dispatcher terminal為`complete`。Raw120 accuracy由baseline `58.9198%`升至`58.9583%`，gain僅`+0.0386` percentage points：7,636→7,641 correct，300 fixes／295 regressions。MacroF1由`60.4554%`降至`60.3522%`，NLL由`1.3923`升至`1.4607`。未達事前overall `+0.20` percentage points門檻，不選winner、不自動增加epochs或搜尋weights。

Paired的4,180 samples由2,774→2,759 correct，141 fixes／156 regressions，net -15；accuracy由`66.3636%`降至`66.0048%`。Unpaired的8,780 samples由4,862→4,882 correct，159 fixes／139 regressions，net +20；accuracy由`55.3759%`升至`55.6036%`。因此本輪沒有unpaired accuracy退步，卻未建立skeleton對paired分類的正收益。

同一unpaired validation與original RGB baseline之間的`T^2 KL`，前輪semantic pilot為`0.06703`，本輪為`0.03393`，約減半；這與較小的output drift相容，但因loss、teacher policy與student dropout policy同時改變，不能將差異全歸因於anchor。沒有matched CE-only control，也不能認定paired退步必然由KD造成。

Frozen teacher／anchor unchanged為true，KD梯度確實到達classifier／vision／MIT。Initial baseline logit error為`1.36e-5`，RGB-only merged reload error為`4.17e-6`。12,960 validation IDs唯一、labels與original baseline一致且logits finite，correct counts與fix/regression counts已獨立核對，runner snapshot hash與immutable lock相符。所有logs、predictions、checkpoint與merged RGB artifact已保留，未讀test。

目前能支持的結論是direct-decision蒸餾與retention流程可運作、unpaired漂移較小，但沒有足夠的主分類收益；這仍不是官方SKI的有效性測試。若繼續，應先檢查teacher軟分布對RGB錯誤是否提供可轉移的訊號，而不是直接放大此設定的train budget。
