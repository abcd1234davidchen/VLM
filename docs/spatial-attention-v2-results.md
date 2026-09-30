# Motion-guided spatial attention：結果與停止決策

日期：2026-09-30。依使用者指示停止新的實驗，保留baseline與全部artifacts。這輪完成兩次pilot，均只訓練120-class模型；沒有完整CE control group，也沒有訓練106-class模型。第二輪為mechanism-only，未達到winner confirmation門檻；不以這些結果否定SKI或所有skeleton supervision方法。

## Canonical validation結果

兩次pilot皆為seed 42、1 epoch、相同hash-stratified 8,192個training samples，完整12,960個canonical validation samples。模型選擇不讀test。Baseline為既有upper-two vision LoRA＋MIT LoRA＋120-class classifier。

| 指標 | Baseline | Joint adaptation | Frozen baseline、attention-only |
| --- | ---: | ---: | ---: |
| Raw 120-class accuracy | 58.9198% | 58.1096% | 58.9429% |
| Accuracy差異（percentage points） | — | −0.8102 | +0.0232 |
| Macro-F1 | 0.604554 | 0.598620 | 0.604798 |
| 正確samples／12,960 | 7,636 | 7,531 | 7,639 |
| 修正baseline錯誤／新增錯誤 | — | 598／703 | 8／5 |
| Paired accuracy | 66.3636% | 65.8852% | 66.4354% |
| Unpaired accuracy | 55.3759% | 54.4077% | 55.3759% |
| Attention CE（較低較好） | 不適用 | 1.9673 | 1.9834 |
| 同frames的center-prior CE | 不適用 | 3.6112 | 3.6112 |
| Motion target mass recall@5 | 不適用 | 0.8496 | 0.8467 |
| Pooling alpha | 不適用 | 0.007623 | 0.002413 |
| RGB-only merged reload max error | 不適用 | 2.38e-6 | 3.34e-6 |
| 決策 | 維持 | no-go | mechanism-only；不選為winner |

Spatial metrics涵蓋held-out paired validation的33,320個有效motion frames。Causal masking固定前32個有效paired samples，每frame遮住5個patches；highest／random／lowest attention masking的平均correct-class logit下降，joint版本為`5.2535／1.1891／0.1015`，frozen版本為`6.4317／1.6138／0.2634`。這支持attention與分類相關，但CLS本身也依賴image patches，因此masking不能單独證明skeleton帶來accuracy收益。

第二輪只比baseline多答對3個samples，未達事前設定的`+0.20` accuracy points確認門檻。Macro-F1與paired metrics的小幅差異同樣沒有第二seed支持；不將其描述為可重現或具實用價值的提升。沒有追加epoch、搜尋weights或另跑完整control group。

## 為何改成frozen baseline

第一輪退步後，使用同一checkpoint做480-sample、每類4個examples的post-hoc inference診斷，沒有parameter updates。保留pooling或將alpha設為零，accuracy均為`60.8333%`，該subset的baseline為`61.2500%`。這只支持「退步出現在更新過的representation／MIT／head，而不是pooling當下的輸出」的局部觀察，不能將退步單獨歸因於spatial loss，也不能取代canonical validation結果。

第二輪因此固定所有baseline vision／MIT LoRA、projection與classifier，只學attention head與alpha，並停用frozen modules的dropout／feature noise。Initial logits與保存的baseline predictions max error為`1.36e-5`；baseline LoRA與classifier tensors在訓練後逐項相等，相關回歸測試也檢查所有frozen state保持不變。凍結避免了第一輪的明顯退步，但仍沒有達到實用分類收益。

## 與SKI paper的區別

[SKI Models，§3.2、Figure 2、Table 1](https://arxiv.org/html/2502.03459v1)的SKI-VLM讓SkeletonCLIP與VideoCLIP共同訓練，以兩個modality與text的class-similarity scores做MSE distillation，並保留雙方CE。其body-focused activation map是結果展示，不能當作本文motion-patch loss的直接實驗依據。作者在NTU120 `110/10` zero-shot split報告XCLIP `57.8% → SKI-XCLIP 66.1%`，其task與本地supervised 120-class validation不同，不能直接比較數字。

| 項目 | SKI-VLM paper | 本輪motion attention |
| --- | --- | --- |
| Skeleton signal | Learned action semantics，經text grounding | Root-relative joint displacement的patch mass |
| Objective | Video-text CE＋skeleton-text CE＋semantic-score MSE | 120-class classifier CE＋patch attention CE |
| Teacher／student | Online collaborative training | 沒有SkeletonCLIP teacher；第二輪固定baseline |
| 本地評估 | 不適用 | Supervised 120-class、participant-disjoint validation |

本repo較早的frozen BlockGCN linear-logit KL也不是原版SCD：當時vision encoder固定、teacher只有A001–A059，而且沒有使用video-text similarity head；這些差異原先記錄於[distillation plan](skeletonclip-distillation-plan.md)。本輪失敗與舊KL no-go都不能當作SKI重現失敗的證據。若日後恢復工作，應先釐清是否要實作semantic distillation與其資料／teacher前提，而不是繼續調motion-pooling weights；目前不啟動這項新實驗。

## Artifacts與verification

- `artifacts/spatial_attention_v2/smoke/`：8-sample motion-target audit、entropy／center／body-part mass、RGB overlays、head-only correct／shuffled screen、adapter與merged RGB-only smoke checkpoint。
- `artifacts/spatial_attention_v2/pilot_seed42/`：第一次pilot的immutable lock、baseline report、full validation predictions、per-action confusion、checkpoint、dispatch metadata與logs；`runner_snapshot.py`的SHA256符合原lock。這輪使用exec session加背景logger，不是tmux。
- `artifacts/spatial_attention_v2/diagnostic_pooling_seed42/`：480-sample inference-only診斷，tmux `spatial-pooling-diagnostic`；不是第三次pilot，也不是trained control group。
- `artifacts/spatial_attention_v2/frozen_attention_smoke/`：attention-only短smoke，correct／shuffled held-out CE為`3.66825／3.69742`，兩個arm的baseline LoRA與classifier tensors皆保持原值。
- `artifacts/spatial_attention_v2/frozen_attention_pilot_seed42/`：第二次pilot，tmux `spatial-frozen-pilot`；`console.log`保存完整輸出，`train.log`保存training／validation進度及terminal結果，`dispatch_status.json`為complete。完整predictions、per-action metrics與RGB-only merged checkpoint均保留。
- 兩次canonical prediction payload皆為12,960個unique identities，與baseline labels一致，logits全部finite；重新計算的正確sample counts符合reports。
- 17項相關測試通過，含motion translation／scale invariance、static masking、identity parity、CE-to-attention gradient、frozen baseline preservation與merged RGB-only strict reload。`git diff --check`通過。未選winner、未評估test、未部署新checkpoint。

目前結論是「motion-guided attention可以學會定位，這兩種pooling策略沒有已證實的實用分類收益」。保留原baseline，依使用者要求停止新實驗；accuracy改善的原目標尚未達成。
