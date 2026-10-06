# XCLIP 原生 attention supervision v1

日期：2026-10-01。使用者要求研究、完成計畫與文件，之後核准實作。目標是改善完整120-class分類，skeleton只用於訓練，推理保持八幀RGB-only、單次forward。使用者選定先做短smoke，再用全部80,640 train跑固定2 epochs。這是本專案的native-attention改版，不是SkeletonCLIP／SKI論文重現。

## 1. 研究依據與取捨

| 研究 | 可採用的依據 | 限制 |
| --- | --- | --- |
| [Seeing the Pose in the Pixels](https://arxiv.org/html/2306.09331) | Pose priors可改善video representations；PAAT可在推理移除pose prediction head | PAAB需要推理時pose；作者分析指出representation收益不只來自attention sparsity，不能據此保證直接attention loss有效 |
| [Attending Self-Attention](https://aclanthology.org/2021.acl-srw.8/) | 比較原生attention直接監督、rollout監督與額外prediction head | Grounding變得更清楚，downstream成績仍沒有勝過原模型；漂亮maps不等於分類收益 |
| [Quantifying Attention Flow](https://aclanthology.org/2020.acl-main.385/) | 多層資訊混合使raw attention不足以解釋模型決策，需配合介入診斷 | Rollout／flow是post-hoc分析，不保證實際分類改善；普通ViT rollout也不能忽略XCLIP message token |
| [官方XCLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP) | Cross-frame communication在vision encoder內進行 | 不能把模型當成獨立per-frame image ViT；需保留時間資訊通道 |

選擇training-time soft supervision；hard mask與inference-time pose bias留作其他研究方向。Hard restriction可能移除手持物件、背景與互動資訊；pose-guided inference也會改變RGB-only部署成本。本輪保留所有native attention運算，只改訓練objective。

現有occupancy probe監督額外spatial head，motion-pooling版本也另外學patch pooling。這次loss直接作用於模型用來聚合values的原生CLS attention，不新增learned attention／pooling branch。只監督固定部分heads，讓其他heads保有不同功能；但Q/V projections共用參數，因此不能保證其餘heads完全不變。

## 2. 本地架構與可行性

本地Transformers為5.3.0，模型為`microsoft/xclip-base-patch32`，revision沿用既有Phase3。Vision有12 layers、12 heads，224×224輸入對應49 patches。每層先產生跨八幀message，然後以`CLS → 49 patches → message`共51 tokens做spatial self-attention；message在該層運算後移除。

現有最後兩層vision與MIT q/v LoRA可以沿用。唯讀tiny-model probe已證明原生attention loss可回傳到vision Q LoRA。Attention probabilities直接取決於Q/K，同一層V不直接接收該loss梯度；V由分類CE更新，較早層V也可能透過下一層的Q/K收到attention梯度。不能將最後一層V的attention-only梯度為零判為接線失敗。

`XCLIPVisionClassifier.forward(..., return_native_attentions=True)`回傳`(logits, attention)`，attention shape為`[B, T, 2, 51]`，是最後兩層各自heads0–3的CLS row平均，包含CLS與message原始mass。預設仍只回傳logits；features與native attention輸出互斥。Runner使用eager backend與原生`output_attentions`，不重寫Transformers attention。Tiny tests可使用兩層／四heads模型，正式runner固定檢查12／12與patch32、frame8。

## 3. 固定targets與loss

以既有joint-to-RGB對齊、原processor幾何與top-two-body occupancy建立49-patch target。有效frame需至少10個state2、in-crop tracked joints。多人occupancy合併；crop外joint不填入target。靜止frame仍可提供位置supervision，不用motion magnitude。缺skeleton或無效frame只算120-class CE。

Occupancy經3×3 Gaussian smoothing（σ=0.5 patch，zero padding），再normalize成`s`；target為`t = 0.8s + 0.2/49`，invalid frame整列為零。保留uniform floor以降低完全排除context的壓力。版本為`native-occupancy-gaussian05-uniform20-v1`。

固定監督vision layers10、11（zero-based），各取heads0–3平均的CLS→patch attention。這四個heads是事前固定的工程heuristic，未依validation選head，也不宣稱最佳。Patch attention除以patch總mass形成conditional distribution`p`；CLS與message不加入空間target、不施加hard mask。

```text
L = CE120(all RGB, label_smoothing=0.1)
    + 0.1 * mean_valid_frames_and_layers KL(t || p)
```

Loss使用FP32、`1e-8`數值保護；all-invalid batch回傳可微分零loss。Conditional KL可能改善patch間分布，卻不增加實際分給patches的mass，所以另報未normalize joint-region mass、conditional joint-region mass、patch／CLS／message mass與conditional entropy。Joint region指有tracked joints的patch occupancy，不是精確人體segmentation。

## 4. Audit、smoke與full pilot

Runner為`src/vlm/training/native_attention.py`，`--mode audit|smoke|pilot`；batch4／seed42固定，workers預設4且hard cap24。沿用原Phase3 epoch13 adapter、canonical manifests、原whole preprocessing及uint8 RGB cache；不暖啟動自後續KD／crop-aware checkpoint。Skeleton targets沿用有source metadata檢查的Phase4 occupancy cache。所有120類保留，test不參與開發。

Audit固定每個paired class8筆、共472 validation，依seed42 sample-ID hash與equal-class round robin選取，不依prediction correctness選樣。保存原始CLS attention、occupancy、smoothed targets、valid mask、logits、metrics與首個樣本的八幀overlay。這只是小subset map診斷，不能代替full validation accuracy或全量對齊審核。

Smoke固定128 train／120 validation，最多8 optimizer steps（實際32 training sample exposures），另做相同472筆before／after attention audit。檢查saved-baseline及attention-output parity、pure attention-to-Q backward、vision／MIT／classifier CE gradients、all-invalid loss與merged RGB-only strict reload；量測peak VRAM與median optimizer step time。Smoke只證明機制與資源可用，不用其accuracy選winner。Smoke另保存前32個有效canonical paired validation的masking診斷。

Pilot要求相同source hashes與input hashes的passing smoke、completed audit。使用全部80,640 train、2 epochs、每epoch完整12,960 validation。固定epoch2為主要结果，epoch1另外保存，不事後挑best epoch。AdamW LR2.25e-5、weight decay5e-4、跨兩epochs的cosine decay、gradient clip1、FP32。沿用rank16／alpha32／LoRA dropout0.05、head dropout0.25、feature noise0.01、CE label smoothing0.1；train mode保留這些既有regularizers。只有upper-two vision q/v LoRA、MIT q/v LoRA與classifier可更新，base參數凍結。

Merged export直接strict reload到現有RGB classifier，不需要skeleton、target loss或attention輸出。原模型單次whole完整validation baseline為58.9198%；zoom ensemble、taxonomy aggregation、threshold與product filtering都不參與這輪主要比較。沒有matched CE-only control，不能把所有續訓收益歸因於attention loss。

每run使用獨立immutable目錄，保存source／input hashes、sample IDs、model/target/runner snapshots、trainable audit、各epoch checkpoint／predictions、history、merged checkpoint與persistent logs。GPU／parity／finite-gradient／coverage／strict-reload契約失敗或OOM則停止並記錄，不自動改recipe或覆寫lock。

```bash
.venv/bin/python -m vlm.training.spatial_dispatch --artifacts-dir artifacts/native_attention_v1/audit --tmux-session native-attention-audit -- .venv/bin/python -m vlm.training.native_attention --mode audit --artifacts-dir artifacts/native_attention_v1/audit
.venv/bin/python -m vlm.training.spatial_dispatch --artifacts-dir artifacts/native_attention_v1/smoke --tmux-session native-attention-smoke -- .venv/bin/python -m vlm.training.native_attention --mode smoke --artifacts-dir artifacts/native_attention_v1/smoke
.venv/bin/python -m vlm.training.spatial_dispatch --artifacts-dir artifacts/native_attention_v1/pilot_seed42 --tmux-session native-attention-pilot -- .venv/bin/python -m vlm.training.native_attention --mode pilot --artifacts-dir artifacts/native_attention_v1/pilot_seed42
```

命令供查核；已有lock的目錄不可重跑。執行結果以`run_report.json`及dispatcher terminal state為準，lock是啟動契約而非完成狀態。

## 5. Verification與判讀

關鍵tests涵蓋attention-output logit identity、51-token indexing、native loss到Q與frozen-base preservation、border／多人target normalization、all-invalid differentiable zero、CLS／message排除及RGB-only merged strict reload。Decision contract另外檢查event gate與負gain判讀；不新增大量重複現有coverage的tests。

Full classification報raw120 accuracy、macro-F1、paired／unpaired、A001–A059／A060–A120、每action confusion與fixes／regressions，以及A042–A048 top1 correct counts。固定472 audit比較before／after target KL與unconditioned joint-region mass；兩者方向皆改善才標記mechanism direction pass，它不是classification winner條件或統計顯著性。

Masking使用canonical validation order前32個有有效paired frames的樣本，每幀遮5個patches，分別採最高、最低native attention與seed42 random；replacement為normalized RGB零值。保存sample IDs、valid masks及correct-class logit下降。輸入遮擋有distribution shift，不能單獨證明skeleton監督的因果貢獻；本輪也不使用忽略message路徑的普通ViT rollout。

Promising須raw120至少比baseline增加0.20 percentage points、macro-F1不下降、unpaired accuracy下降不超過0.20 points且A042–A048合計correct不下降。Raw120負gain為no-go，其餘inconclusive。單seed不足以選winner／核准部署；promising後另行規劃matched CE-only confirmation，不在這輪預設完整多arm控制組。一次失敗只限制此固定recipe與partial59資料，不否定所有attention supervision。

## 6. 執行紀錄

5項native-attention contract tests與既有vision／spatial tests共10項通過，`git diff --check`通過。472筆GPU audit已完成，IDs唯一、包含每個paired class8筆；3,766個有效frames。開啟attention輸出的logit差異為0，與保存baseline的472筆argmax完全一致，logit最大誤差`4.51e-5`。Overlay已目視查看，綠框為tracked-joint occupancy，紅色為conditional native patch attention；不把首個代表樣本當成全量grounding證據。

GPU smoke完成8 optimizer steps，status為`pass`；vision／MIT／classifier CE gradients皆finite且非零，attention-only loss到vision Q LoRA的gradients非零，all-invalid loss為0。Median optimizer step約`0.0675秒`（不含data loading與validation），峰值allocated VRAM約`1.07 GiB`；merged RGB-only strict reload最大誤差`3.58e-6`。不能把這個小budget的optimizer timing當成完整訓練wall time。

Smoke固定472 audit的target KL由`0.75975`降至`0.75576`，unconditioned joint-region mass由`0.24924`到`0.24919`，mechanism direction未通過；smoke只驗證功能，並未建立分類收益。全部logs／snapshots／checkpoints保留於`artifacts/native_attention_v1/audit/`與`smoke/`，未讀test。

Full pilot已透過tmux `native-attention-pilot` dispatch，目錄為`artifacts/native_attention_v1/pilot_seed42/`，source hashes與passing smoke／audit一致。固定80,640 train、2 epochs及每輪12,960 validation；現已完成，終局結果如下。`console.log`保存stdout／stderr，`train.log`保存persistent progress，`dispatch.json`保存PIDs及命令，`run_report.json`產生後才判定promising／no-go／inconclusive。

## 7. Full pilot完成結果與決策

Dispatcher terminal為`complete`，固定2 epochs、40,320 optimizer steps／161,280 training sample exposures及每輪12,960 validation完成，總run wall time約61.1分鐘。事前raw120 gate決策為`inconclusive`，不選winner、不推薦替換baseline，也不自動延長epochs或搜尋weights。未讀test。

| 指標 | Baseline | Epoch1 | 最終Epoch2 |
| --- | ---: | ---: | ---: |
| Raw120 accuracy | 58.9198% | 58.8503% | 58.9429% |
| Raw120 correct | 7,636 | 7,627 | 7,639 |
| Macro-F1 | 60.4554% | 60.4892% | 60.5930% |
| A042–A048 correct | 303 | 309 | 313 |

Epoch2相對baseline僅多答對3筆（+0.0232 percentage points），未達+0.20 points門檻。關注七類top1多答對10筆不能取代逐事件recall／false alarms分析，也不能當成threshold或部署收益。

依使用者指定排除的24個source actions作額外product-scope分析，保留96類的9,792筆validation由6,671降至6,638 correct：68.1270%→67.7900%，淨減33筆（−0.3370 points）；334 fixes／367 regressions。被排除24類的3,168筆則965→1,001 correct：30.4609%→31.5972%，淨增36筆。Raw120淨增3筆主要由不要求独立產品輸出的類別抵銷保留類別退步；不符合使用者希望改善保留類別的目標。

這份scope分析只移除ground-truth屬24類的評估樣本；模型、120個logits與argmax不變，預測成排除類別仍算錯，不移除logits後renormalize。結果與明確source IDs保存在`product_scope_analysis.json`；它是後續分析，不改寫immutable lock或原定raw120 gate。

固定472 audit的target KL由0.75975降至0.39499，未normalize joint-region mass由0.24924增至0.27420，mechanism direction通過。Patch mass由0.38661升至0.46971，但conditional joint-region mass由0.63757降至0.57763，entropy由2.47635升至2.91522；attention確實改變且更符合soft target，不等同於更尖銳的joint focus或已帶來分類收益。

結論：此partial59／最後兩層固定四heads／smoothed occupancy target的recipe學到了attention引導，但沒有實用overall收益，保留產品類別還退步。維持原baseline，停止放大這個recipe；單一seed且沒有matched CE-only續訓control，不能將退步單獨歸因於attention loss，也不能外推成所有native attention或SkeletonCLIP方法無效。全部snapshots、epoch predictions／checkpoints、merged RGB-only export與logs保留。
