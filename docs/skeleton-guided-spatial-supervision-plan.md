# Skeleton-guided spatial supervision plan

日期：2026-09-30。狀態：joint-adaptation pilot為no-go，frozen-baseline pilot為mechanism-only（accuracy僅增加`0.0232` points）；依使用者指示停止新實驗、保留baseline。完整結果與SKI paper差異見[results](spatial-attention-v2-results.md)。第一版旁支occupancy probe為no-go。現階段只以120-class模型判定結果，106-output aggregation不列為門檻或必要交付，也不訓練第二個106-class模型。

## 1. 研究問題

對具有可靠RGB–skeleton alignment的training samples，將2D joint位置與motion轉成XCLIP patch-level attention targets，並讓model預測的attention直接參與frame embedding與classification，是否能在維持RGB-only inference的前提下改善action classification，尤其是目前容易混淆的body-contact與大幅肢體動作？

這不是BlockGCN logit distillation。Teacher signal是「哪些image patches包含正在產生判別性運動的joints或body parts」，不是BlockGCN對action classes的機率分布。Ground-truth 120-class CE仍是主要目標；skeleton target只在training提供，但由RGB預測的attention與pooling module會保留在inference path。

## 2. 現有證據與失敗原因

Phase 4 Gate 2已完成第一版occupied-joint patch supervision。第20 epoch的joint arm相對CE arm只提升`0.001911` patch AP、`0.002867` top-k recall與`0.0772` classification accuracy points；只有單一seed，未達可重現或具實際意義的改善，因此舊實作是no-go。

舊probe的CE arm已達`0.954959` patch AP，代表「預測任何joint-occupied patch」太容易，也主要反映人體通常位於畫面中央。Classification讀取原始vision `pooler_output`，spatial head則從patch tokens獨立預測occupancy；分類路徑可以完全忽略spatial head。Shuffled-joint的patch AP降到`0.759423`，證明正確target含有可學signal，但「能定位人體」沒有自動轉成「分類決策使用該位置」。因此舊結果否定的是可繞過的旁支auxiliary設計，不足以否定decision-coupled spatial supervision。

2026-09-30完成的direct 106-output upper-vision run達到`61.9830%` exact accuracy，證明upper vision adaptation本身有效；但仍低於既有120-class upper-vision logits經probability aggregation的`65.8565%`。下一版因此保留120-class head與細粒度CE，避免在representation learning階段把視覺差異大的17類硬合併；aggregation只作部署後處理與106-output evaluation baseline。

## 3. Motion-aware target

每個sampled RGB frame沿用已驗證的skeleton-to-RGB alignment與XCLIP patch grid。有效joints先依Kinect confidence與frame validity遮罩，再建立三層資料：

1. `joint occupancy`只保留作最低限度localization diagnostic，不作主要learning target。
2. `motion-weighted joints`以相鄰skeleton frames的normalized joint displacement分配target mass；先移除root translation並正規化body scale，避免走路時所有joints因整體位移一起取得高權重。
3. `body-part distribution`保留head、torso、left/right arm與left/right leg，以及interaction samples的兩個主要body；第一輪只作diagnostic與可視化，不立即增加multi-head complexity。

主要spatial objective使用normalized soft spatial distribution，而不是大量negative patches主導的binary BCE。第一版固定per-frame cross-entropy between predicted patch attention and motion-weighted target；所有invalid frames與joints完全mask。除非單一motion target smoke的失敗原因明確指向part collapse，否則不搜尋多種target、blur或loss weights。

## 4. Decision-coupled model path

輸入仍為deterministic whole-video uniform 8 frames。從已驗證的upper-two vision-block LoRA + MIT LoRA + 120-class classifier開始，新增由RGB patch tokens預測的attention與identity-initialized residual pooling：

```text
RGB frames → XCLIP vision encoder → CLS token ──────────────────────────────┐
                              └→ patch tokens → attention logits → softmax │
Skeleton joints ── training only ─────────→ motion-weighted target         │
                                                                              ↓
frame = CLS + α × (attention-pooled patches − CLS) → projection → MIT → 120-class head
                                                                              ↓
                                                        probability aggregation → 106 outputs
```

總loss固定為（第11節的frozen-baseline候選使用相同loss，但baseline weights不更新）：

```text
L = L_CE_120 + λ_spatial × L_attention
```

`α`初始化為零，使新增path在第一個forward與既有classifier完全一致；attention head先由skeleton loss取得gradient，classification loss同時學習是否及多大程度開啟residual。Attention與classification gradient均可更新事前鎖定的upper vision LoRA，frozen base XCLIP不可更新；MIT LoRA與classifier只接收classification gradient。Checkpoint保存attention module、target版本與provenance，但不包含skeleton payload。部署保留RGB attention pooling，移除skeleton loader與spatial loss。

現階段只以raw 120-class metrics衡量representation是否真正改善。Probability aggregation若日後恢復，只是同一模型的後處理，不能把label merging本身的提升歸因給skeleton；本階段不要求106-output或public taxonomy結果。

## 5. 快速迭代與comparison策略

- 第一輪只跑identity parity、短smoke與單一`correct spatial` pilot，直接對照已鎖定的既有120-class upper-vision checkpoint；不強制先重跑完整CE arm。
- Smoke包含固定shuffled target，確認attention objective依賴sample alignment；shuffled只跑到足以判斷mechanism，不必與每個候選使用相同長度。
- 候選若未改善raw 120-class validation accuracy，停止，不以額外control擴大成本；macro-F1一併報告以檢查class-level影響。
- 候選若超過既有baseline，才補同初始化與budget的CE comparison或第二seed，以區分新architecture、額外參數與skeleton supervision的貢獻。
- `static center/person prior`與舊`joint occupancy`只作diagnostic；不得以高occupancy AP替代classification或causal evidence。

## 6. Evaluation

Model selection只看canonical validation，不讀test。Classification報告raw 120-class accuracy／macro-F1，另報A001–A059／A060–A120、paired／unpaired與action-level confusion changes。106-output與public taxonomy metrics依使用者指示暫緩；既有runner若順帶生成aggregation欄位，不參與本階段判定，也不代表訓練了第二個模型。

Spatial evaluation只用held-out paired validation，報告attention-target cross-entropy、motion-joint top-k recall與相對center prior的gain。Causal diagnostic遮住最高attention patches，並和matched random及lowest-attention patches比較correct-class logit下降；只改善correlation或localization不足以證明分類真正使用spatial signal。

Action-level analysis優先檢視低accuracy但產品仍需要的類別。NTU 25-joint skeleton沒有每根手指，也不描述瓶子、剪刀或紙張，因此fine hand sign與small-object actions只作限制分析，不把它們設定為本階段必須改善的主要gate。

## 7. Gates

### Gate 0：alignment and target audit

從不同setup、camera、single-person與interaction actions抽樣可視化motion-weighted targets，確認frame mapping、左右joints、兩人場景與invalid masking。要求identity/hash fail closed，並量化target entropy、center bias、valid frame coverage與各body part coverage。

### Gate 1：mechanism smoke

小型deterministic subset驗證finite loss／gradients、`α=0`初始化logit parity、只有whitelist parameters更新、all-invalid batch退化成CE、shuffled mapping可重現、checkpoint strict reload與RGB-only parity。Correct target必須比shuffled target更快降低attention loss；joint-adaptation版本須證明開啟`α`後CE gradient通過attention-pooled tokens進入vision LoRA，frozen-baseline版本則須證明只更新attention／alpha且baseline tensors保持不變。未通過對應adaptation policy的smoke則不進pilot。

### Gate 2：single-seed fast pilot

以事前鎖定的小型budget跑單一correct arm，和既有upper-vision baseline的相同canonical validation metrics比較。Correct arm必須同時滿足：(a) attention metric優於center prior，shuffled smoke僅作同subset的mechanism evidence；(b) masking causal diagnostic方向一致；(c) raw 120-class accuracy改善。Macro-F1及各subgroup一併檢查，106-output結果不參與本階段gate。未達成則no-go，不延長epoch或搜尋多個weights。

### Gate 3：confirmation

只有Gate 2達到至少`+0.20` accuracy points或事前指定的macro-F1 gain，才補matched CE comparison或第二seed。確認收益不是新pooling架構單獨造成，且A060–A120、unpaired與retained actions沒有明顯退步後，才鎖定winner並考慮test-once。

## 8. Stop conditions

若correct與shuffled在attention metric近似、`α`維持接近零、masking沒有causal effect、收益只存在於center-heavy localization，或後補comparison顯示improvement只來自新pooling參數，停止此路線。不得用反覆搜尋`λ_spatial`、target blur、epoch或解凍層數把單一validation差異調成正值。

## 9. Deliverables

- Immutable experiment lock與target schema/version。
- Alignment audit與representative overlays。
- Pilot config、checkpoint、完整validation predictions與per-action metrics；只有通過pilot才補matched comparison artifacts。
- Raw 120-class accuracy／macro-F1、subgroup／per-action metrics與causal masking結果；106-output及public relevance暫緩。
- RGB-only merged checkpoint parity與明確的go／mechanism-only／no-go decision。

## 10. 第二版實作與鎖定的快速pilot

`src/vlm/spatial_joint_targets.py`新增`root-scale-motion-v1`：以每個sampled frame的前／後相鄰skeleton frame計算motion，必須是同一body ID且joint、root與spine shoulder皆為tracked；XYZ先減去spine-base root，再除以root–spine-shoulder長度。Motion mass投影到目前frame的color-image patch位置，保留兩個body與六個body parts的mass。至少10個有鄰幀motion觀測的in-crop joints且總motion大於`1e-6`才supervise；完全靜止或沒有有效鄰幀的frame遮罩為零，不以occupancy prior填補。Motion只用於分配target，inference完全不讀skeleton。

`XCLIPAttentionLoRAClassifier`以RGB patch tokens預測softmax attention，對pooled patch representation套用與CLS相同的frozen vision post-layernorm，再以zero-initialized `pooling_alpha`混合CLS與patch pooling。Export以`XCLIPAttentionClassifier`strict reload，保留attention與alpha、移除LoRA wrappers；不使用旧版會移除spatial head的RGB export路徑。

2026-09-30的8-sample target audit包括不同setup/camera、single-person與A054/A050/A059 interaction，64/64 frames有效；entropy、center mass與body-part mass及representative overlays在`artifacts/spatial_attention_v2/smoke/`。這是representative motion-target audit，既有full RGB–skeleton alignment audit仍是對齊依據，不能把8-sample結果稱為full motion coverage。Head-only correct/shuffled screen的train attention CE分別由`3.8976 → 2.1875`及`3.9698 → 2.9116`，held-out CE為`2.0202`及`3.0439`；這只是固定RGB tokens的可學性screen。

`artifacts/spatial_attention_v2/joint_smoke_v2/run_report.json`記錄同初始化、同budget的real-data joint-training smoke（16 train／16 val，2 epochs）。Correct/shuffled held-out attention CE為`3.6575`／`3.6915`，motion mass recall@5為`0.4460`／`0.4027`，correct alpha為`0.000423`；correct尚未超過center prior CE `3.5343`。最高／random／最低attention patch masking的平均correct-class logit下降為`3.8311`／`1.0568`／`0.5710`，merged RGB-only reload max error為`2.38e-6`。這些結果只支持進入pilot，不能以16-sample accuracy差異聲稱分類提升，也不能將masking本身視為已證明skeleton的因果收益。

Fast pilot事前固定seed 42、1 epoch、8,192 train samples、batch 4、fp32與`λ_spatial=0.1`。Train以seeded sample-identity hash排序，再依120類equal-class round robin抽樣；所有12,960 canonical validation samples皆評估，不抽validation subset。Vision／MIT／classifier／attention-and-alpha LR固定`1e-4`／`2e-5`／`8e-5`／`1e-4`，不搜尋weights或epochs。Pilot只跑correct arm，對照warm-start既有120-class validation logits；不訓練106-class模型。Causal diagnostic固定canonical order的前32個有效paired validation samples，每frame遮住5個patches。Pilot若滿足改善gate才考慮second-seed confirmation，不強制完整control group；若未通過則停止，不延長epoch。

Head-only shuffled screen只有4個held-out samples，joint shuffled smoke只有16個validation samples，因此不能直接把它們的CE與canonical pilot CE作numeric threshold比較。Pilot先以相同canonical frames的center prior、classification與masking方向作candidate screen；有收益後confirmation必須補相同held-out frames上的shuffled診斷與matched CE／second seed，才能對supervision收益作go決策。這項修改避免以不同evaluation subset形成錯誤的Gate 2證據。

```bash
.venv/bin/python -m vlm.training.spatial_attention_smoke
.venv/bin/python -m vlm.training.spatial_attention --mode smoke --artifacts-dir artifacts/spatial_attention_v2/joint_smoke_v2
.venv/bin/python -m vlm.training.spatial_attention --mode pilot --epochs 1 --train-samples 8192 --causal-samples 32 --artifacts-dir artifacts/spatial_attention_v2/pilot_seed42
```

Run directories有immutable lock且不會覆寫既有run；上面的smoke commands已執行，不應在同目錄重跑不同實作。所有runner保持test embargo，pilot不把較小train budget或未完成confirmation當成full objective完成。

Joint-adaptation pilot已完成，只有correct arm，沒有重跑完整CE control group。`artifacts/spatial_attention_v2/pilot_seed42/dispatch.json`保存training／logger PID與exact command，獨立背景logger將進度與terminal結果寫入`train.log`及`dispatch_status.json`；epoch metrics、per-action confusion、validation predictions與checkpoint各自保存在correct目錄。這一輪使用existing exec session加logger，不是tmux；後續dispatcher預設使用tmux，完整stdout／stderr保存到`console.log`，session名稱記錄於`dispatch.json`。舊runner原始碼已保存為`pilot_seed42/runner_snapshot.py`，SHA256與原experiment lock一致；新版runner遵照使用者指示只生成120-class結果。

```bash
.venv/bin/python -m vlm.training.spatial_dispatch --artifacts-dir artifacts/spatial_attention_v2/NEW_RUN -- .venv/bin/python -m vlm.training.spatial_attention --mode pilot --epochs 1 --train-samples 8192 --causal-samples 32 --artifacts-dir artifacts/spatial_attention_v2/NEW_RUN
```

## 11. Joint-adaptation no-go與frozen-baseline候選

完整canonical validation（12,960 samples）的joint-adaptation pilot raw accuracy由baseline `58.9198%`降至`58.1096%`（`−0.8102` points）；macro-F1由`0.604554`降至`0.598620`。修正598個baseline錯誤，同時新增703個錯誤。Paired／unpaired accuracy均下降，unpaired退步較多。Attention CE `1.9673`優於center `3.6112`，motion mass recall@5為`0.8496`，highest／random／lowest masking平均logit下降為`5.2535`／`1.1891`／`0.1015`；alpha為`0.007623`。因此attention mechanism成立，但沒有分類收益，第二版joint-adaptation路徑停止，不延長epoch或搜尋weight。

使用既有checkpoint、沒有任何parameter updates的480-sample post-hoc診斷已在tmux `spatial-pooling-diagnostic`執行完成，artifacts位於`diagnostic_pooling_seed42/`。同樣的trained weights保留pooling或將alpha設為零，accuracy均為`60.8333%`，原baseline為`61.2500%`；不能把這個subset代替canonical metrics，也不能將退步單獨歸因於spatial gradient而排除CE更新。結果支持先檢驗是否能在不改動baseline representation的情況下取得pooling收益，而不是加長失敗run。

下一個候選固定所有baseline vision／MIT LoRA、projection與classifier，只讓RGB attention head與alpha更新；frozen modules維持eval mode以停用dropout／feature noise。Motion target、lambda、attention architecture、train subset、seed與budget都維持不變。這是由失敗診斷支持的一次adaptation-policy修改，不是weight／epoch搜尋。若它仍不能改善120-class accuracy，停止此attention-pooling家族，不再用同budget連續嘗試變體。`frozen_attention_smoke/`已完成16-train／16-val、2-epoch smoke；correct／shuffled CE為`3.66825`／`3.69742`，classification parity保持baseline，兩個arm的baseline LoRA與classifier checkpoint tensors皆逐項相等。

Frozen-baseline pilot仍只跑correct arm，1 epoch／8,192 train／全部12,960 val／32個paired causal samples，保留相同baseline及raw-accuracy門檻，不強制control group。Runner檢查initial logit parity、baseline權重未變與merged RGB-only strict reload，並持續寫出validation sample進度。

```bash
.venv/bin/python -m vlm.training.spatial_dispatch --artifacts-dir artifacts/spatial_attention_v2/frozen_attention_pilot_seed42 --tmux-session spatial-frozen-pilot -- .venv/bin/python -m vlm.training.spatial_attention --mode pilot --adaptation attention-only --epochs 1 --train-samples 8192 --causal-samples 32 --joint-smoke-report artifacts/spatial_attention_v2/frozen_attention_smoke/run_report.json --artifacts-dir artifacts/spatial_attention_v2/frozen_attention_pilot_seed42
```

## 12. 轉向SKI-style semantic distillation

使用者在暫停motion-pooling並整理成果後，要求嘗試SKI的核心方法。新的實作與budget見[SKI-style semantic pilot](ski-semantic-v1-plan.md)：先prealign RGB對文字的語意branch，再以online skeleton/video text-conditioned scores進行MSE distillation，保留120-class主任務與RGB-only部署。這不是另一個spatial target／pooling變體；motion-pooling的兩次結果不變，也不證明SKI不可行。現有teacher只覆蓋59類，因此明確標示supervised120／partial-pair59改版，不宣稱完整論文重現。
