# Skeleton-guided spatial supervision plan

日期：2026-09-30。狀態：第二版架構提案；第一版旁支occupancy probe為no-go，下一版改測decision-coupled spatial attention。Training維持120-class CE，產品taxonomy只在evaluation與inference作probability aggregation，不與direct 106-output training混用。

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

總loss固定為：

```text
L = L_CE_120 + λ_spatial × L_attention
```

`α`初始化為零，使新增path在第一個forward與既有classifier完全一致；attention head先由skeleton loss取得gradient，classification loss同時學習是否及多大程度開啟residual。Attention與classification gradient均可更新事前鎖定的upper vision LoRA，frozen base XCLIP不可更新；MIT LoRA與classifier只接收classification gradient。Checkpoint保存attention module、target版本與provenance，但不包含skeleton payload。部署保留RGB attention pooling，移除skeleton loader與spatial loss。

Probability aggregation與spatial learning是正交操作：raw 120-class metrics衡量representation是否真正改善，aggregated 106-output與public metrics衡量產品taxonomy。任何spatial gain必須先在相同aggregation下相對既有120-class upper-vision baseline成立，不能把label merging本身的提升歸因給skeleton。

## 5. 快速迭代與comparison策略

- 第一輪只跑identity parity、短smoke與單一`correct spatial` pilot，直接對照已鎖定的既有120-class upper-vision checkpoint；不強制先重跑完整CE arm。
- Smoke包含固定shuffled target，確認attention objective依賴sample alignment；shuffled只跑到足以判斷mechanism，不必與每個候選使用相同長度。
- 候選若未改善raw 120-class或相同aggregation後的106-output validation metrics，停止，不以額外control擴大成本。
- 候選若超過既有baseline，才補同初始化與budget的CE comparison或第二seed，以區分新architecture、額外參數與skeleton supervision的貢獻。
- `static center/person prior`與舊`joint occupancy`只作diagnostic；不得以高occupancy AP替代classification或causal evidence。

## 6. Evaluation

Model selection只看canonical validation，不讀test。Classification同時報告raw 120-class accuracy／macro-F1，以及套用完全相同mapping後的106-output exact、public accuracy、relevant named accuracy、irrelevant recall與false-irrelevant rate；另報A001–A059／A060–A120、paired／unpaired與action-level confusion changes。

Spatial evaluation只用held-out paired validation，報告attention-target cross-entropy、motion-joint top-k recall與相對center prior的gain。Causal diagnostic遮住最高attention patches，並和matched random及lowest-attention patches比較correct-class logit下降；只改善correlation或localization不足以證明分類真正使用spatial signal。

Action-level analysis優先檢視低accuracy但產品仍需要的類別。NTU 25-joint skeleton沒有每根手指，也不描述瓶子、剪刀或紙張，因此fine hand sign與small-object actions只作限制分析，不把它們設定為本階段必須改善的主要gate。

## 7. Gates

### Gate 0：alignment and target audit

從不同setup、camera、single-person與interaction actions抽樣可視化motion-weighted targets，確認frame mapping、左右joints、兩人場景與invalid masking。要求identity/hash fail closed，並量化target entropy、center bias、valid frame coverage與各body part coverage。

### Gate 1：mechanism smoke

小型deterministic subset驗證finite loss／gradients、`α=0`初始化logit parity、只有whitelist parameters更新、all-invalid batch退化成CE、shuffled mapping可重現、checkpoint strict reload與RGB-only parity。Correct target必須比shuffled target更快降低attention loss，且開啟`α`後classification gradient確實通過attention-pooled tokens進入vision LoRA；否則不進pilot。

### Gate 2：single-seed fast pilot

以事前鎖定的小型budget跑單一correct arm，和既有upper-vision baseline的相同validation metrics比較。Correct arm必須同時滿足：(a) attention metric優於shuffled smoke與center prior；(b) masking causal diagnostic方向一致；(c) raw 120-class或aggregated 106-output主要metric至少一項改善，且另一項不退步超過`0.20` points。未達成則no-go，不延長epoch或搜尋多個weights。

### Gate 3：confirmation

只有Gate 2達到至少`+0.20` accuracy points或事前指定的macro-F1 gain，才補matched CE comparison或第二seed。確認收益不是新pooling架構單獨造成，且A060–A120、unpaired與retained actions沒有明顯退步後，才鎖定winner並考慮test-once。

## 8. Stop conditions

若correct與shuffled在attention metric近似、`α`維持接近零、masking沒有causal effect、收益只存在於center-heavy localization，或後補comparison顯示improvement只來自新pooling參數，停止此路線。不得用反覆搜尋`λ_spatial`、target blur、epoch或解凍層數把單一validation差異調成正值。

## 9. Deliverables

- Immutable experiment lock與target schema/version。
- Alignment audit與representative overlays。
- Pilot config、checkpoint、完整validation predictions與per-action metrics；只有通過pilot才補matched comparison artifacts。
- Raw 120-class、aggregated 106-output、public relevance與causal masking結果。
- RGB-only merged checkpoint parity與明確的go／mechanism-only／no-go decision。
