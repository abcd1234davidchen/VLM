# SkeletonCLIP Phase 2：teacher-logit cache 與 masked-KL distillation

日期：2026-09-16。狀態：Gate 0–2 已完成；identity lock、45,236 筆 frozen teacher raw-logit cache與 masked-KL integration smoke皆通過，尚未啟動 Gate 3 full training。Phase 1 已完成 teacher selection，依據與完整結果見 `docs/skeletonclip-teacher-phase1.md`；整體研究背景見 `docs/skeletonclip-distillation-plan.md`。

目前可續接點：`uv run vlm train skeleton-phase2 --through-gate 2 --batch-size 64 --workers 4` 會依序重驗 Gate 0、fail-closed reuse Gate 1 cache並重跑 Gate 2 smoke。正式 cache位於 `cache/skeleton_teacher_logits/3e4926a1f26529e1/teacher_logits.pt`；machine-readable結果位於 `artifacts/skeletonclip_phase2/`。Gate 3 前不得讀 test split。

## 1. Phase 2 要回答的問題

Phase 2 只回答：「在完全相同的 120-class XCLIP MIT LoRA student recipe、初始化、資料 split 與 training budget 下，對有效 A001–A059 paired training samples 加入 frozen skeleton teacher 的 masked class-logit KL，是否能穩定改善 RGB-only student？」

本階段不解凍 XCLIP vision encoder、不加入 feature alignment、不重現論文的 online SCD、不補 S018–S032 skeleton、不改 student classifier head，也不以 skeleton 作為 inference input。若結果改善，可主張第一版 skeleton-informed temporal／decision adaptation；沒有 spatial supervision 或 localization evidence 時，不得宣稱 joint-focused visual learning。

Phase 2 的完成條件是得到可信且可重現的結論，不是預設 KD 必須勝出。結果可以是 go、conditional 或 no-go；三種結果都必須保留完整 artifacts，且不得在看過 test 後更換 teacher、temperature、loss weight、seed 或 checkpoint。

## 2. 已鎖定的實驗基線

| 項目 | Phase 2 固定值 |
| --- | --- |
| Teacher | BlockGCN linear classifier，`artifacts/skeletonclip_gate3/linear_seed42/best_checkpoint.pt` |
| Teacher classes | ordered `A001`–`A059`；A060 明確排除 |
| Teacher input | Phase 1 deterministic whole-clip preprocessing，`[N, 3, 64, 25, 2]` |
| Teacher output | temperature scaling 前的 59-class raw logits |
| Student | `microsoft/xclip-base-patch32`，MIT `q_proj`／`v_proj` rank-16 LoRA + 120-class head |
| Student input | 既有 deterministic whole-video 8-frame pre-MIT `[8, 512]` cache |
| Student recipe | `artifacts/xclip_lora_tuning/winning_config.json` 的完整內容與 hash |
| Canonical split | `artifacts/xclip/selected_split.json`；80,640 train／12,960 validation／20,880 test |
| Trainable parameters | MIT q/v LoRA 與 classification head only |
| Deployment | RGB-only；不載入 teacher、skeleton parser、teacher cache 或 KD code |

Phase 1 的有效 paired counts 是 train 45,236、validation 4,180，涵蓋 A001–A059。Gate 1 必須從 canonical manifest 與 Phase 1 paired manifest 重新核對這些數字；若數字、membership 或 hash 不一致，不得以「接近」為由繼續。

既有 tuned winner 的 52.12% validation accuracy 與 54.52% test accuracy只作歷史參考，不是本階段公平 control。正式比較必須從同一 pretrained XCLIP state 分別重跑 CE-only 與 CE+KD，不能讓 KD 從已訓練的 winner 接續 fine-tune。

## 3. 不可變的 loss contract

令 `C_T` 為 teacher ordered classes `A001`–`A059` 對應到 student 120-class output 的顯式 index vector，`P` 為 batch 中同時具有效 skeleton 與 cache entry 的 training samples。不得用 label range、row position或補零的 120-class teacher distribution 隱式取代 identity-based pairing。

```text
p_i = softmax(teacher_logits_i / T)
q_i = softmax(student_logits_i[C_T] / T)
L_KD = T² / |P| × Σ(i∈P) KL(p_i || q_i)
L = L_CE(student_logits_120, labels_120) + λ × L_KD
```

`L_CE` 使用全部 120-class training samples；`L_KD` 只使用 `P`。`P` 為空時以 differentiable zero 處理，CE 仍正常 backward。KL 固定以 FP32 `softmax`／`log_softmax` 計算，先對 59 classes 求和，再依整個 optimizer step 的 paired sample總數平均；gradient accumulation 不得讓 paired counts 不同的 micro-batches取得相同權重。Teacher cache必須 detach，training path 不建立 teacher model或 teacher gradient graph。

每個 epoch 至少記錄 CE、unweighted KD、weighted KD、total loss、paired／total sample counts、paired fraction、optimizer steps與 learning rates。`λ=0` 必須走與 CE+KD 相同的 dataloader、model與accounting path，作為 implementation control。

## 4. Gate 0：identity freeze 與 test embargo

### 工作

- 計算並保存 teacher checkpoint、Phase 1 paired manifest、canonical RGB split、student winning config與相關 source revision的 SHA256／Git identity。
- 從 checkpoint strict-load teacher，驗證 architecture、head、ordered class IDs、preprocessing version、excluded classes與 validation metric provenance。
- 建立顯式 `teacher class ID → student class index` mapping，要求長度 59、順序唯一、無 A060、無缺漏或重複。
- 建立新的 Phase 2 run namespace，不覆寫 Phase 1 artifacts、既有 XCLIP caches或既有 tuning winner。
- 在程式介面上讓 search／confirmation runs禁止 test evaluation；只有 Gate 5 鎖定的 final command可讀 test split。

### Pass gate

- 所有 identity 可機器驗證，checkpoint strict reload 無 missing／unexpected keys，59-class mapping 完整且 round-trip 一致。
- Canonical split membership與既有 winner完全一致，train／validation／test identity無交集。
- 在第一個 CE+KD full run 前產生一份 immutable experiment lock，明列本文件版本、搜尋空間、seed policy與決策規則。

### 預期產物

- `artifacts/skeletonclip_phase2/experiment_lock.json`
- `artifacts/skeletonclip_phase2/class_mapping.json`
- 一份 identity audit，清楚列出每個來源 path、hash、class order與 split counts。

任何 hash、class order或 split membership 無法對上時，Phase 2 在此 no-go，先修 provenance，不建立 teacher cache。

## 5. Gate 1：frozen teacher raw-logit cache

### 工作

- 以 selected teacher 的原始 preprocessing對 paired train逐筆 deterministic inference，保存 sample identity與 59-class FP32 raw logits；cache identity不得包含 `T` 或 `λ`。
- Cache與 XCLIP pre-MIT cache分離，建議位於 `cache/skeleton_teacher_logits/<cache_identity>/`；manifest記錄 checkpoint SHA256、source／paired manifest hashes、ordered classes、preprocessing、sampling policy、dtype、tensor shape與每筆 payload fingerprint。
- 預設只建立 paired train logits。Paired validation只保留 pairing metadata供 subgroup diagnostics；不得以 validation teacher logits加入 training。不得建立或讀取 test teacher logits。
- 支援 interruption-safe resume；existing entry只有在 identity、shape、dtype與 payload fingerprint皆相容時才可 reuse。
- 隨機抽查至少 128 筆 cache logits與 online frozen teacher output，另完整掃描 finite values、unique identities、class labels、shape與 coverage。

### Pass gate

- 恰好覆蓋 45,236 個預期 paired-train identities，每個 identity正好一筆 `[59]` FP32 finite raw logits，0 missing、0 extra、0 duplicate、0 class mismatch。
- 128 筆或更多 online parity抽查的 `max_abs_diff ≤ 1e-5`；若 extraction/storage precision使此門檻不合理，必須先改成 FP32，不可放寬門檻來接受 lossy cache。
- Teacher在 `eval()`／no-grad下重跑結果 deterministic；cache validation不依賴 validation或 test labels。
- 修改 checkpoint、preprocessing、class order、paired manifest或 source skeleton fingerprint時，cache compatibility必須 fail closed。

### 預期產物

- Teacher-logit cache與 machine-readable manifest
- `artifacts/skeletonclip_phase2/cache_validation.json`
- Coverage、logit min／max、entropy、argmax distribution與 parity摘要；這些只作 sanity check，不作 student hyperparameter selection。

## 6. Gate 2：masked-KL integration smoke

### 工作

- 在既有 XCLIP LoRA training path加入 optional teacher cache與 masked KL，保持 CE-only預設行為不變。
- 使用小型 deterministic batches覆蓋 all-paired、mixed paired／unpaired、all-unpaired、A060、A061–A120與 shuffled class mapping。
- 比較 single batch與拆成不同 gradient-accumulation micro-batches的 update，驗證 loss normalization依有效 paired samples而非 micro-batch數量。
- 驗證 checkpoint save／strict reload、LoRA merge parity與完整 RGB-only inference path；student checkpoint只能保存 teacher provenance metadata，不得嵌入 teacher weights或 cache payload。

### Pass gate

- `λ=0` 與原 CE implementation在固定 batch、dropout關閉時的 loss、gradients與一次 optimizer update一致，FP32 `max_abs_diff ≤ 1e-6`；AMP end-to-end prediction parity沿用既有 LoRA容許值。
- `λ>0` 時只有 MIT q/v LoRA與 head收到 gradients；base XCLIP、teacher與 cached logits皆無 gradients。
- All-unpaired batch的 KD為零且 CE update正常；mixed batch的 KD只對 paired rows計算；刻意打亂 class order或 sample identity會 fail fast。
- Gradient accumulation切法不改變以 paired-sample normalization計算的 KD update，容許值由 FP32 reference test鎖定。
- Adapter與 merged checkpoint strict reload成功，merge前後 logits符合既有 parity contract，且 inference不需要任何 skeleton相關資源。

### 預期產物

- `src/vlm/skeleton_teacher_cache.py` 或等價的單一 cache責任模組
- 可獨立測試的 masked-KL／loss-accounting helper，以及既有 `xclip_lora` path的最小 optional integration
- 保護 class mapping、cache invalidation、empty-paired batch、accumulation normalization、trainable parameter whitelist與 RGB-only reload的 targeted tests
- `artifacts/skeletonclip_phase2/integration_smoke.json`

### Gate 0–2 執行結果

Gate 0 已 strict-load `artifacts/skeletonclip_gate3/linear_seed42/best_checkpoint.pt`，核對 checkpoint、paired manifest、canonical RGB split、winning student config與本規格的 SHA256，並確認 canonical split為 train 80,640／validation 12,960／test 20,880、split identities互斥、paired train 45,236／validation 4,180。顯式 A001–A059 → student indices 0–58 mapping通過 uniqueness與 round-trip；完整 lock、mapping與 audit分別位於 `experiment_lock.json`、`class_mapping.json`、`identity_audit.json`。

Gate 1 已在 NVIDIA GeForce RTX 2080 Ti 對 45,236 筆 paired train完成 deterministic inference，cache為 FP32 `[45236, 59]`、0 missing／extra／duplicate／non-finite；128 筆 online parity `max_abs_diff=0.0`，重複 forward determinism `max_abs_diff=0.0`。Cache保存 checkpoint／manifest／preprocessing／class-order identity、每筆 source skeleton SHA256與 payload fingerprint；重跑時會先重算全部 source fingerprints，任一 identity、membership、source或payload不一致即 fail closed，未完成 batch shards則可安全 resume。摘要見 `cache_validation.json`。

Gate 2 已將 optional cache與 masked KL接入既有 `xclip_lora` training path；未提供 `--teacher-cache` 時原 CE path不變。Smoke覆蓋 mixed、all-unpaired、A060與 A061–A120 labels、錯誤 mapping、不同 gradient-accumulation切法、frozen base gradient contract與 RGB-only strict reload：`λ=0` update `max_abs_diff=0.0`，accumulation update `max_abs_diff=3.725290298461914e-09`，strict reload `max_abs_diff=0.0`，all-unpaired KD為 0。Checkpoint只保存 teacher provenance，不含 teacher weights或 cache payload；摘要見 `integration_smoke.json`。

## 7. Gate 3：controlled training smoke 與 CE parity

啟動紀錄（2026-09-16）：smoke subset在首次執行前固定為 canonical train各 class依 sample identity SHA256排序後取 2.5%、validation各 class取 4%，test為空；三組皆跑 3 epochs，依序為 `T=2, λ=0`、`T=2, λ=0.1`、`T=4, λ=1.0`。這些設定只用來夾住預定搜尋空間的低／高強度並檢查數值尺度，不參與 Gate 4 winner selection。執行狀態與完整 identity持續寫入 `artifacts/skeletonclip_phase2/gate3_summary.json`；完成的 run可跳過，失敗／中斷狀態會保留。Smoke全數通過後才自動啟動 seed-42、60 epochs、`T=2, λ=0`的完整 control。

### 工作

- 先以固定小型 train subset與獨立 validation subset跑 `λ=0`、一組低強度 KD與一組高強度 KD，確認 loss尺度、throughput、memory與 artifact schema；smoke結果不得用來宣稱模型品質。
- 接著在完整 train／validation split以 seed 42重跑一個正式 `λ=0` CE control，使用完整 winning recipe與 60 epochs，禁止 test。
- 比對新 control與既有 CE-only pipeline的 learning curve、best epoch、validation accuracy、macro-F1、trainable parameter count與 merge parity。

### Pass gate

- 所有 losses、gradients與 logits finite；KD不是恆為零，亦未因少量 paired samples被 micro-batch weighting放大。
- 新 `λ=0` control的 validation accuracy與從相同 seed／recipe重跑的舊 CE path差異不超過 0.10 percentage points；若無法在同一環境直接重跑舊 path，至少要求新 path的 `λ=0` direct-parity run與 CE-only entry point共用相同 core loop且逐步 parity smoke通過。
- 訓練每 epoch恰好消費 80,640 個 CE samples，paired KD count恰好 45,236；augmentation不得改變 sample identity或 teacher lookup。
- Artifact可重建完整 run config、loss history、paired coverage、validation predictions與 selected checkpoint。

### 預期產物

- `artifacts/skeletonclip_phase2/smoke/`
- `artifacts/skeletonclip_phase2/control_seed42/`
- 一份 CE parity與 training-cost摘要，作為 Gate 4所有候選的唯一 control。

若 `λ=0` 無法重現 CE contract，不得開始 KD搜尋；先修 integration，而不是以 KD candidate相對壞掉的 control計算增益。

## 8. Gate 4：validation-only KD search

Dispatch紀錄（2026-09-16）：Gate 3 control以52.1065% validation accuracy、53.5149% macro-F1完成，和歷史baseline差異`-0.0154` percentage points，通過`±0.10` parity gate。Gate 4固定九組完整60-epoch seed-42 runs，順序為temperature `{1, 2, 4}`外圈、KD weight `{0.1, 0.5, 1.0}`內圈；共用一次preloaded train／validation feature cache，每個run逐epoch寫入`train.log`、`progress.json`與可續接`last_checkpoint.pt`。Orchestrator持續更新`artifacts/skeletonclip_phase2/search_summary.json`，保存失敗／中斷狀態，已完成candidate不重跑；test features與test evaluation均禁用。每組完成後保存完整validation predictions與raw logits，並報告accuracy、macro-F1、NLL、ECE及預定subgroups。背景dispatch log為`artifacts/skeletonclip_phase2/gate4_dispatch.log`，PID記錄於`artifacts/skeletonclip_phase2/gate4.pid`。

### 搜尋空間

第一輪固定使用 seed 42、完整 60 epochs、相同 early-selection規則，搜尋 `T ∈ {1, 2, 4}` 與 `λ ∈ {0.1, 0.5, 1.0}` 共九組。`λ=0` 使用 Gate 3 control，不對每個 temperature重複。所有候選從相同 pretrained XCLIP初始化開始，不承接 control或其他候選 optimizer state。

若 Gate 3 顯示 weighted KD／CE ratio在前五個 epochs持續小於 `0.01` 或大於 `10`，視為尺度 implementation review，而不是自動擴張搜尋空間。任何新增 `T`／`λ` 都必須先更新 experiment lock並確認尚未讀取 Phase 2 test結果。

### Ranking 與 advance gate

- Primary metric是完整 12,960-sample RGB validation accuracy；secondary依序為完整 validation macro-F1、A001–A059 macro-F1、NLL與 ECE。Paired subset metrics只作診斷，不可取代完整 validation ranking。
- 最佳 KD candidate只有在 seed-42 validation accuracy至少高於 control `0.20` percentage points，或 accuracy落在 control ±`0.20` points內但 macro-F1至少改善 `0.30` points時，才進入 Gate 5。
- 若多個候選差距小於 `0.10` accuracy points，選較小 `λ`；仍相同時選較低 `T`，避免事後依單一 subgroup挑 winner。
- 必須同時報告 A001–A059、A060–A120，以及 A001–A059內 paired／unpaired validation subgroups。這些 subgroup不作第一輪 winner selection，但用來偵測 supervision bias。

### 預期產物

- 每組 candidate的 config、checkpoint、完整 validation predictions、loss／coverage history與 compute cost
- `artifacts/skeletonclip_phase2/search_summary.json`，包含所有完成／失敗 runs、固定 ranking與 advance理由
- 一個 locked winner recipe，或明確的 validation no-go結論

Gate 4 no-go時 Phase 2仍可完成：記錄「在預先宣告的搜尋空間與單 seed下沒有足夠訊號」，不讀 test、不改門檻，也不挑一個較差 candidate硬進 confirmation。

### Gate 4 執行結果

Gate 4 已於 2026-09-17 完成九組 seed-42、60-epoch validation-only runs；`test_evaluated=false`，所有 candidate均完成且 sample accounting為每 epoch 80,640 CE／45,236 paired。Gate 3 CE control為 validation accuracy `52.1065%`、macro-F1 `53.5149%`。

| Candidate | Accuracy | Macro-F1 | NLL | ECE | Δ accuracy vs CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `T=1, λ=0.1` | 51.9907% | 53.4967% | 1.6450 | 0.0309 | -0.1157 pp |
| `T=1, λ=0.5` | 51.4738% | 52.8098% | 1.6718 | 0.0404 | -0.6327 pp |
| `T=1, λ=1.0` | 50.4861% | 51.8946% | 1.6917 | 0.0436 | -1.6204 pp |
| `T=2, λ=0.1` | 51.9985% | 53.3539% | 1.6659 | 0.0443 | -0.1080 pp |
| `T=2, λ=0.5` | 51.3889% | 52.6361% | 1.7187 | 0.0608 | -0.7176 pp |
| `T=2, λ=1.0` | 50.6867% | 51.9158% | 1.7546 | 0.0734 | -1.4198 pp |
| `T=4, λ=0.1` | 51.9830% | 53.2893% | 1.6708 | 0.0446 | -0.1235 pp |
| `T=4, λ=0.5` | 51.2886% | 52.3626% | 1.7379 | 0.0651 | -0.8179 pp |
| `T=4, λ=1.0` | 50.8333% | 51.8249% | 1.7762 | 0.0744 | -1.2731 pp |

固定 ranking的 winner是 `T=1, λ=0.1`，但其 accuracy比 control低 `0.1157` points、macro-F1低 `0.0183` points，因此未達 `+0.20` accuracy或 `+0.30` macro-F1 advance gate，Phase 2 第一版 class-logit KD 結論為 `no-go`。結果支持「student仍正常學習，但此 teacher／coverage／loss contract沒有提供增益」；不支持讀 test或進行 seed-43 KD confirmation。完整 machine-readable結果見 `artifacts/skeletonclip_phase2/search_summary.json`。

## 9. Gate 5：two-seed confirmation、最終決策與 test-once

### 工作

- 對 Gate 4 locked KD recipe跑 seed 43；對相同 CE recipe也跑 seed 43。Seed 42直接沿用 Gate 3／4正式 runs，不重挑 checkpoint規則。
- 以 seeds `{42, 43}`彙整完整 validation與預先宣告 subgroups；不得以平均後結果回頭換另一組 `T`／`λ`。
- 只有達到 final go或 conditional條件時，才鎖定一個 RGB-only merged checkpoint selection policy並對 CE control與 KD方案各執行一次 test evaluation。Test結果只作最終估計，不改方案。

### 最終決策規則

| 決策 | Validation條件 | 後續動作 |
| --- | --- | --- |
| Go | KD的 two-seed mean accuracy改善至少 `0.20` points，兩個 seeds皆不低於各自 control超過 `0.20` points，且 A060–A120 mean accuracy退步不超過 `0.50` points | 對 locked CE與KD各 test once，報告完整與 subgroup結果；KD成為下一階段候選 baseline |
| Conditional | Mean accuracy改善介於 `0.00`與 `0.20` points，或整體達 go但 A060–A120退步超過 `0.50` points | 可做 test-once作完整報告，但不把 KD升為預設；下一步只能是預先設計的 coverage／loss ablation |
| No-go | Mean accuracy不高於 control，或任一 seed退步超過 `0.50` points | 不執行新 KD test；保留 CE student為預設，結束第一版 class-logit KD |

這些 percentage points是本專案 Phase 2 的工程決策門檻，不宣稱統計顯著性。兩個 seeds不足以支持廣泛統計結論，因此報告必須呈現逐 seed結果與差值，不只呈現平均值。

### 預期產物

- `artifacts/skeletonclip_phase2/confirmation_summary.json`
- 若為 go／conditional：locked CE與KD的 test-once predictions、整體 accuracy／macro-F1／NLL／ECE、A001–A059與A060–A120 metrics、A001–A059 paired／unpaired metrics及 confusion diagnostics
- KD adapter checkpoint與 RGB-only merged checkpoint；merged checkpoint通過 parity並只含 deployment所需 student state
- 一份 Phase 2 final report，明確標示 go／conditional／no-go、可支持的 claim與不可支持的 claim

## 10. Artifact 與 CLI 建議

Phase 2應優先重用既有 `xclip_lora` core loop，不複製第二套 model、evaluation或 checkpoint程式。建議新增 `xclip-lora-distill` mode作為明確 CLI surface，但 CE-only與distill必須共享同一 training implementation；`--kd-weight 0`是可稽核 control，而不是另一套 code path。

```text
cache/skeleton_teacher_logits/<identity>/
artifacts/skeletonclip_phase2/
├── experiment_lock.json
├── class_mapping.json
├── cache_validation.json
├── integration_smoke.json
├── control_seed42/
├── search/<temperature>-<weight>-seed42/
├── confirmation/<control-or-kd>-seed43/
├── search_summary.json
├── confirmation_summary.json
└── final_report.json
```

Machine-readable run artifacts至少保存 Git revision、command、resolved config、all source hashes、teacher/cache identity、student base model revision、class mapping、split membership hash、seed、precision、trainable parameter audit、optimizer／scheduler、best epoch、loss history、paired counts、throughput、peak memory、checkpoint hashes與 predictions。失敗或中斷的 run也要保留 status與原因，resume不得把 partial run誤認為 complete。

## 11. 明確不在 Phase 2 的工作

- S018–S032與完整 120-class skeleton teacher
- XCLIP text-prototype teacher、feature cosine loss或原論文 online SCD
- Bone／motion multi-stream teacher ensemble
- Vision encoder upper-layer LoRA、patch token或 joint heatmap supervision
- 依 paired samples oversampling、重加權 CE或改 canonical split
- 讀取 test來選 `T`、`λ`、seed、epoch、teacher或 checkpoint

其中任何一項若日後啟動，都必須成為獨立 phase／ablation並配有自己的 CE control；不得塞入 Phase 2 winner run後仍把差異歸因於 masked class-logit KD。

## 12. Kickoff 對齊清單

開始實作前只需確認以下三個 policy；其餘 contract可直接依本文件執行。

1. 接受 Gate 4 的固定九組 `T × λ` search，以及 seed 42先篩、seed 43只確認 locked winner的 adaptive compute策略。
2. 接受 `+0.20` accuracy points、per-seed `-0.20` points與 A060–A120 `-0.50` points作為 Phase 2工程門檻。
3. 接受 conditional結果可作 test-once完整報告，但不升格為預設 KD recipe；no-go則不讀新 KD test。

對齊後的第一個 implementation slice是 Gate 0與 Gate 1：先產生 experiment lock、顯式 class mapping與可 fail-closed的 teacher raw-logit cache，完成 45,236 筆 coverage及 online parity驗證後，再碰 student training loop。
