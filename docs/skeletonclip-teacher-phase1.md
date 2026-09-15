# SkeletonCLIP Phase 1：teacher feasibility 與 backbone selection

日期：2026-09-14。狀態：Gate 0、Gate 1 與 Gate 2 已完成；Gate 2 選出BlockGCN作為唯一Gate 3 finalist，Gate 3 linear baseline兩個seeds已完成，XCLIP head尚未啟動。本文件將 `docs/skeletonclip-distillation-plan.md` 的 teacher feasibility 拆成可執行的第一階段，且不把論文使用的 Hyperformer 視為既定答案。

## 1. Phase 1 要回答的問題

Phase 1 只回答：「在不碰 RGB student、不使用 test 選模、也不引入資料洩漏的前提下，哪一個 skeleton backbone 與 output head 最適合成為後續 distillation teacher？」本階段完成後應得到一個 frozen 59-class teacher checkpoint、明確的 preprocessing contract、validation 報告與完整 provenance；teacher-logit cache 和 RGB student KL integration 屬於下一階段。Gate 0 review 發現 A060（walking apart from each other）有大量 non-finite skeleton 欄位，造成 usable coverage 嚴重失衡；使用者於 2026-09-13 決定整類排除，因此後續 class order 固定為 A001–A059，A060 不參與 training、validation 或選模。

現有 skeleton archive 只涵蓋 S001–S017 與 A001–A060；排除 A060 後，Phase 1 是 59-class teacher pilot。訓練只使用 canonical RGB train 中能精確配對且通過 parser validation 的 skeleton；checkpoint selection 只看 paired validation；test 不執行，直到 teacher 與後續 student 實驗規則全部鎖定。

Phase 1 不以「排行榜最高的模型」直接決定 teacher。公開 benchmark 通常使用官方 cross-subject／cross-view split、多 stream ensemble 與不同 preprocessing，不能和本專案的 participant-disjoint paired subset直接比較。選擇依據是本地受控 validation 的 teacher quality、logit quality、重現性、單模型成本與後續 text alignment 可行性。

## 2. 已核對的 reference implementation

官方 SKI-Models Phase 1 使用 Hyperformer 將 `[N, 3, T, 25, 2]` skeleton 映射到 512 維，與 CLIP text features 做 dot-product classification。公開程式以預先產生的 NPZ 載入 skeleton，使用 OpenAI CLIP `ViT-B/16`、prompt `a photo of a {class}.`、100 epochs、AdamW、cosine schedule、label smoothing 0.1 與 gradient accumulation；README 要求 Python 3.7／舊版 PyTorch 並依賴 Apex、mmcv-full 與 decord。[SKI-Models 官方 repository](https://github.com/thearkaprava/SKI-Models)

這份 reference 不能直接複製進本專案。公開 config 的 `TRAINER.SKI_VLM.USE: "both"` 會讓 Hyperformer 與 CLIP text encoder 都可訓練，程式沒有使用 CLIP `logit_scale`、pose embedding 沒有顯式 L2 normalization，而且 `strict=False` 載入 Hyperformer weights；更重要的是，預設 Mixup 只混合未被 teacher forward 使用的 RGB tensor與 labels，沒有同步混合 skeleton tensor，會讓 skeleton input 與 soft label 不一致。這些行為必須在本專案改成明確、可稽核的 contract。論文路線仍保留為比較基準，但不是預設 winner。

## 3. 候選 backbone

| Candidate | Phase 1 角色 | 優點 | 主要風險 | 決定 |
| --- | --- | --- | --- | --- |
| Hyperformer | Paper-faithful reference | 與官方 SkeletonCLIP 接法最接近；2.6M parameters；現成 NTU preprocessing 與 weights | 較舊 stack；reference code 有寬鬆 checkpoint loading；不代表目前最佳 skeleton classifier | 保留 |
| SkateFormer | Research-only modern challenger | ECCV 2024 skeletal-temporal Transformer；官方提供 training、testing 與 pretrained model；單一 architecture 同時建模 skeletal-temporal partitions | Research/education-only license；官方發布包含多個 NTU protocol checkpoints，可能和 canonical split重疊，不可用於正式 clean run；需確認 512 維 adapter 接點 | 不作必要依賴；用途確認為研究時才比較 |
| BlockGCN | License-safe modern primary | CVPR 2024；強調 topology-aware 且 parameter-efficient；Apache-2.0；與 Hyperformer 共用相近 NTU preprocessing lineage | 官方 README 沒有明列可下載 checkpoint；GCN representation 接 text space 的品質需實測 | 建議優先 |
| HD-GCN | Deferred challenger | 強勁的 hierarchical graph baseline，官方有 pretrained weights | 最佳結果偏向多 branch／ensemble，增加訓練與 cache 決策變因；舊 dependency stack | 第一輪不跑 |

SkateFormer 官方程式與 checkpoint 可供研究教育使用，但商業使用需取得作者許可；若本專案可能商用，這一點會直接改變最終選型。[SkateFormer 官方 repository](https://github.com/KAIST-VICLab/SkateFormer) BlockGCN 使用 Apache-2.0，並沿用 Hyperformer／SGN 系列的 NTU 資料處理方式。[BlockGCN 官方 repository](https://github.com/ZhouYuxuanYX/BlockGCN) HD-GCN 保留為後續 challenger，而非把 ensemble benchmark 數字當成單 teacher 的選型依據。[HD-GCN 官方 repository](https://github.com/Jho-Yonsei/HD-GCN)

## 4. Output head 選項

每個通過 smoke gate 的 backbone 先暴露一個固定維度的 pooled skeleton representation，再用兩種 head 做受控比較；backbone 與資料完全相同，避免把「骨架架構」和「語言對齊方式」混成一個變因。

### Option A：linear classifier head

`skeleton_feature → LayerNorm → Linear(59)`，以 ground-truth CE 訓練。這是 teacher-strength control，通常最直接回答 skeleton 本身能否在目前 split 上分類。若它明顯優於 text-aligned head，後續可以把它作為 classifier teacher，但命名必須是 skeleton classifier teacher，不能稱為 SkeletonCLIP。

### Option B：frozen text-prototype head

`skeleton_feature → projection → L2 normalization → cosine similarity with frozen text prototypes × learned/clamped logit_scale`，以 ground-truth CE 訓練，只更新 skeleton backbone、projection 與可選的 scalar logit scale。這是本專案定義的 SkeletonCLIP teacher；text encoder、tokenizer與 class prompt features 固定並寫入 checkpoint provenance。

第一輪 text encoder 建議使用與 RGB student 相同的 `microsoft/xclip-base-patch32` text tower與鎖定 revision，因為後續可能比較 language-aligned features，且可避免額外引入另一套 CLIP implementation。為確認這個選擇沒有犧牲 reference compatibility，僅在勝出 backbone 上比較一次 OpenAI CLIP `ViT-B/16` reference text prototypes；不對每個 backbone 做完整笛卡兒積搜尋。

Prompt pilot 固定為三個預先宣告的模板並對 normalized text features 取平均：`{label}`、`a video of a person {label}`、`a person is {label}`。Action label 必須使用人工審核過的英文語句，而不是直接把 `A001` 或資料夾名稱送入 tokenizer。Template set 只可依 validation 選一次，選定後寫入 teacher identity。

## 5. 實驗漏斗

### Gate 0：data contract

- 安全解壓 S001–S017 archive，保留原 ZIP，拒絕 path traversal、symlink member 與非預期檔名，並讓 ZIP CRC 驗證完成。
- 解析官方 NTU `.skeleton` 格式，套用官方 missing-skeleton list，再檢查 frame、body、joint counts、非有限值、空序列與 tracking validity。
- 以完整 `SsssCcccPpppRrrrAaaa` identity 對 canonical split 配對，輸出 train／validation coverage、排除原因、duplicate、label mismatch 與 split leakage 報告。
- 產生不可變的 paired manifest 與 hash；RGB sample 即使沒有有效 skeleton 也不從原 student manifest 刪除。

Gate 0 是 Phase 1 的硬前置。若有效 train 或 validation coverage 與目前檔名估計差異過大，先回來審查資料，不進行 architecture ranking。

### Gate 1：compatibility smoke

對 Hyperformer、SkateFormer、BlockGCN 各做 CPU shape test、單 GPU forward/backward、16-sample overfit 與 checkpoint round trip。統一輸入語意為 `[N, 3, T, 25, 2]`，但每個 adapter 可使用該官方模型要求的 temporal crop／resample；所有差異都記入 preprocessing identity。

通過條件為 finite loss、可重現 eval logits、16-sample overfit accuracy 至少 95%、strict checkpoint reload 無 missing／unexpected keys，以及單 GPU peak memory 在實際硬體預算內。任何候選若需要把舊 framework 大量搬入主專案、無法 strict reload 或無法清楚取得 pooled feature，便在這一 gate 淘汰。

### Gate 2：short controlled bake-off

通過 Gate 1 的 backbone 全部從 random initialization 開始，先配 Option A linear head，用同一 paired train／validation manifest、joint stream、temporal length、batch-size-equivalent optimizer steps、augmentation policy與固定 seeds `{42, 43}` 訓練 20 epochs。此階段用來淘汰明顯較弱或不穩定的 implementation，不把 20-epoch排名當最終品質排名。

進入下一 gate 的候選最多兩個。主要排序依 paired validation macro-F1，其次為 top-1 accuracy、negative log-likelihood、expected calibration error、seed variance、peak memory 與 examples/sec。若差距在合理波動內，優先保留實作較簡單、license 較合適且成本較低者。

### Gate 3：full teacher comparison

最多兩個 backbone 以 Option A 完整訓練並使用 validation 選 checkpoint；其中勝出者再跑 Option B frozen text-prototype head與 reference text encoder check。完整 run 不承接 short bake-off optimizer state，所有候選從相同 seed policy 重啟，以免 early proxy 帶來不等量訓練。

最終 teacher 不只看 top-1。若 Option B 的 accuracy 略低但 soft target entropy、NLL、ECE與 per-class confusion 更適合 distillation，可選 Option B；如果 Option B 顯著落後，先使用 Option A 做 classifier-KD sanity experiment，並把真正的 language-aligned SkeletonCLIP 延後，而不是為了名稱犧牲 teacher quality。

2026-09-15在任何XCLIP full-run產生結果前，Gate 3計算策略鎖定為adaptive funnel：linear baseline仍以seeds `{42, 43}` 完整確認；XCLIP text-prototype head先只跑seed 42。若XCLIP seed 42的macro-F1明顯低於依linear baseline計算的2 percentage-point門檻，便停止該路線並選linear teacher；只有結果接近或通過門檻時才補XCLIP seed 43確認穩定性。這項節流不移動quality gate，也不以test或已觀察的XCLIP結果作決策。

OpenAI CLIP `ViT-B/16` reference改為診斷性ablation而非Gate 3必跑項目；只有XCLIP結果異常且需要區分「text alignment本身」與「特定text space」時才執行。省略reference不影響linear與XCLIP head的主要teacher選擇，但最終報告必須明記reference是否執行及原因。

### Gate 3 linear baseline執行結果

Gate 3 linear baseline以BlockGCN、固定paired train 45,236筆、paired validation 4,180筆、64 frames、batch size 16、100 epochs與seeds `{42, 43}` 從random initialization完整訓練；兩組run均完成strict checkpoint reload與validation metric reproduction，未讀取test split。Machine-readable彙整位於 `artifacts/skeletonclip_gate3/teacher_comparison_report.json`。

| Seed | Best epoch | Macro-F1 | Accuracy | NLL | ECE | Examples/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 100 | 86.51% | 86.56% | 0.6383 | 0.0754 | 126.26 |
| 43 | 94 | 86.43% | 86.46% | 0.6332 | 0.0729 | 129.12 |
| Mean ± population stddev | — | 86.47% ± 0.04% | 86.51% ± 0.05% | 0.6358 ± 0.0025 | 0.0741 ± 0.0013 | 127.69 ± 1.43 |

兩個seeds的macro-F1差距僅0.08 percentage points，支持linear teacher quality具高度重現性；兩組peak allocated VRAM皆約1.08 GiB。依預先固定的2 percentage-point quality gate，XCLIP seed 42需達到約84.47% macro-F1，且NLL／ECE條件成立，才值得補跑XCLIP seed 43；明顯低於此門檻便選linear teacher並停止主要head comparison。

### Gate 2 執行結果

Gate 2 使用固定 paired train 45,236 筆、paired validation 4,180 筆、A001–A059 class order、64 frames、batch size 16、seeds `{42, 43}` 與每組 20 epochs。四組 run 均從 random initialization 完成，未讀取 test split；完整 machine-readable 結果位於 `artifacts/skeletonclip_gate2/bakeoff_report.json`。

| Backbone | Seed 42 best macro-F1 / accuracy | Seed 43 best macro-F1 / accuracy | Macro-F1 mean ± population stddev | Mean examples/s | Peak allocated VRAM |
| --- | --- | --- | --- | --- | --- |
| BlockGCN | 86.57% / 86.56% | 85.76% / 85.81% | 86.17% ± 0.40% | 157.19 | 1.08 GiB |
| Hyperformer | 76.99% / 77.11% | 68.52% / 68.66% | 72.75% ± 4.23% | 71.77 | 3.39 GiB |

BlockGCN 在兩個 seeds 的最低 macro-F1 仍高於 Hyperformer 兩個 seeds 的最高 macro-F1，同時平均 throughput 約為 2.19 倍且 peak allocated VRAM 較低，因此 Hyperformer 在本 Gate 被明確支配並淘汰。Gate 3 只保留 BlockGCN；Gate 2 的 20-epoch checkpoint 是選型證據，不承接 optimizer state 作為 Gate 3 初始化。

## 6. 公開 pretrained weights 的使用界線

Hyperformer、SkateFormer、HD-GCN 的官方 checkpoint 多半依官方 NTU splits 訓練，可能見過本專案 canonical validation／test participants或 labels。Phase 1 可下載它們以驗證 state-dict mapping、shape、預處理一致性與 throughput，但不得用其 validation performance 選正式 teacher，也不得作正式 teacher 初始化。

正式 teacher 預設從 random initialization 在 canonical paired train 訓練。若後續想測 external-pretraining，必須另立 exploratory track，保存 checkpoint來源與官方 training protocol，並在報告中明確標為可能有 split overlap；它不能支持乾淨的 distillation gain claim。

## 7. 訓練與評估 contract

- Teacher class order固定為 `A001` 至 `A059`，checkpoint 同時保存 ordered class IDs、排除的 A060、英文 labels 與 prompt strings。
- Training只讀 paired train；validation只做 selection／diagnostics；test 在 Phase 1 不讀取、不產生 metrics。
- 第一輪只用 joint stream，不用 joint／bone／motion ensemble；其他 modalities 必須是後續獨立 ablation。
- Temporal resampling、body selection／ordering、root centering、rotation、scale normalization、valid-frame masking與padding皆版本化；padding不可產生假 motion。
- 多人樣本保留兩個 tracked bodies；不能以「第一個 body」取代 identity-consistent ordering。
- Loss與 similarity 在 FP32 計算；mixed precision只用於安全的 backbone operations；任何 NaN／Inf 立即使 run 失敗。
- Checkpoint selection預設以 macro-F1 為主，top-1 為次；同時保存 NLL、ECE、per-class support／F1、confusion matrix與 raw validation logits。
- Run artifact記錄 Git revision、環境、GPU、seed、trainable parameter whitelist、optimizer、scheduler、effective batch size、steps、throughput、memory、manifest hash與source-code provenance。

## 8. 建議的 Phase 1 決策

建議採用「BlockGCN primary + Hyperformer reference + SkateFormer research-only challenger」的漏斗，而不是直接重現官方 Hyperformer。第一個正式比較先用 linear head測出 backbone ceiling，再只對勝出 backbone加入 frozen XCLIP text-prototype head。這樣可以分辨失敗來自 skeleton encoder 還是 language alignment，也把完整訓練數控制在合理範圍。

預設選型順序如下：BlockGCN作 modern primary；Hyperformer一定保留到 short bake-off，提供和 SKI paper最接近的 reference；SkateFormer只有在專案用途確認符合其 research/education-only license時才進入 short bake-off，而且不成為可商用路徑的必要依賴；HD-GCN與 multi-stream ensemble延後。

## 9. Phase 1 產物

審核本文件後的實作應分成可獨立 review 的小切片，且第一個切片只處理資料與介面，不直接啟動完整訓練。

1. `artifacts/skeleton_data/coverage.json`、`paired_manifest.json`、`excluded_samples.json` 與 source hashes。
2. 統一的 skeleton sample／batch contract和三個候選的最小 adapter，舊 repository code不整包 vendoring。
3. Gate 1 compatibility報告，包含 strict reload、overfit、memory與throughput。
4. Gate 2 bake-off報告與最多兩個 finalist決策。
5. 最終 frozen teacher checkpoint、validation predictions、metrics、class／prompt mapping、preprocessing identity與完整 provenance。
6. 一份 go／no-go結論：進入 teacher-logit cache與student distillation，或先修正資料／teacher品質。

## 10. 四項問題的白話說明與採用決策

以下四項已用保守、可重現且不綁死未來用途的方式定案，不再作為 implementation 前的 blocker。若專案用途或研究目標之後改變，可以把替代方案加成獨立 ablation，但不回頭改寫主要 Phase 1 結果。

### 10.1 SkateFormer 的授權能不能接受？

**白話問題：** SkateFormer 的程式和 checkpoint 不是一般「任何用途都可以」的授權。官方只允許 research／education free use，商業使用要另外取得作者正式許可。就算目前只是學校研究，只要未來把模型放進商業產品、收費服務或公司內部產品驗證，都可能需要重新確認權利。如果我們把 SkateFormer 寫成整個 teacher pipeline 的唯一核心，之後用途一變，就可能連架構和 checkpoint 都要重做。

**採用決策：** 正式 primary 改為 Apache-2.0 的 BlockGCN；Hyperformer保留為論文reference。SkateFormer不作必要依賴，只在專案用途明確屬於研究／教育時作 challenger experiment。即使SkateFormer在研究比較中勝出，也要分別保存 `research-only best` 與 `license-safe best`，後續student distillation預設使用後者。這個決策優先確保產物可持續使用，不是假設BlockGCN一定有最高accuracy。

**什麼情況會改：** 若使用者明確確認本專案永遠只作研究／教育，或已取得SkateFormer商業許可，可以把SkateFormer升回完整 finalist，但仍不能使用可能洩漏canonical validation／test的官方weights作正式初始化。

### 10.2 為什麼正式 teacher 要從 random initialization 開始？

**白話問題：** 官方 pretrained model通常已用NTU官方split訓練。那個split和我們目前的participant split不同，所以官方模型很可能已看過我們放在validation甚至test中的人物或影片。即使我們只拿它當「起始權重」，它記住的資訊仍可能讓validation看起來特別好。這樣後續student進步時，我們無法分辨是skeleton knowledge真的有用，還是teacher把不該看過的資料帶進來。

**採用決策：** 所有正式比較都從random initialization開始，只以canonical paired train更新參數，並只以paired validation選checkpoint。官方weights只能用在不產生研究結論的工程smoke，例如確認state-dict key能否對上、tensor shape正不正常、forward能不能跑和大概需要多少GPU memory。Smoke結果不得和正式validation結果放在同一張比較表。

**代價與理由：** 從零訓練會比較慢，也可能低於官方benchmark，但它給我們一個乾淨答案。Architecture本身曾在NTU上被研究過不算sample leakage；真正需要隔離的是已經由其他split labels訓練完成的參數。

### 10.3 第一輪為什麼只用 joint stream，不直接做 ensemble？

**白話問題：** Skeleton模型常把同一份資料轉成joint coordinates、bone vectors、joint motion、bone motion等多種stream，分別訓練模型後再平均prediction。這通常會提高排行榜accuracy，但也把「backbone好不好」、「哪種input有用」和「多花幾倍算力」混在一起。後續每個teacher variant還要產生logits cache，因此一開始用四或六個模型會放大訓練、儲存和除錯成本。

**採用決策：** Phase 1所有候選只使用原始3D joint stream與單一模型，統一比較資料、steps和成本。Bone／motion stream不會被永久排除；只有單joint teacher已通過quality gate、但validation仍顯示明確瓶頸時，才新增一個獨立ablation，且必須同時報告增益、參數、training compute與cache成本。

**好處：** 如果BlockGCN勝過Hyperformer，我們能比較有把握地說差異來自backbone，而不是BlockGCN偷偷用了更多modalities或更多模型。這也讓Phase 1較快產出第一個可蒸餾teacher。

### 10.4 SkeletonCLIP 應該用哪一個 text encoder？

**白話問題：** SkeletonCLIP在本階段不是只做59類分類，它要把skeleton feature投影到文字embedding附近。不同CLIP版本即使輸出都是512維，也不代表它們位於同一個語意空間。原論文code用OpenAI CLIP `ViT-B/16`，目前RGB student則建立在`microsoft/xclip-base-patch32`。如果teacher使用另一套text space，後續想比較或蒸餾language-aligned features時會多一層無法解釋的差異。

**白話理解：** 一般classifier是在學「這組骨架是哪一類」；SkeletonCLIP則是在學「這組骨架最接近哪一句動作文字」。例如一般classifier直接輸出A001，而SkeletonCLIP會比較骨架和`drinking water`、`reading`、`standing up`等60個文字向量的距離。文字方法多保留了一層語意關係，但這層限制也可能讓原本容易區分的骨架動作變得較難分類。

**修正後的採用決策：** 不預先把任何text encoder版本指定為最終teacher。勝出的skeleton backbone先用自由學習的linear classifier head建立「只看骨架最多能做到多好」的基準，再用完全相同的backbone訓練frozen XCLIP text-prototype head。若text-aligned版本保持接近的macro-F1，且NLL、ECE與soft logits品質沒有惡化，才把它選為主要SkeletonCLIP teacher；若它明顯較弱，第一版class-logit distillation先使用較強的普通skeleton classifier teacher，SkeletonCLIP保留為未通過quality gate的獨立實驗。這可避免為了使用SkeletonCLIP名稱而讓student向較差的teacher學習。

**主要text候選：** Language-aligned版本先使用`microsoft/xclip-base-patch32`的frozen text tower並鎖定model revision、tokenizer、60個英文action labels、prompt templates、normalization與logit scale。Text features預先計算後視為常數；訓練不更新text encoder，只更新skeleton backbone、projection與受限制的scalar logit scale。選它是因為它和現有RGB student來自同一個XCLIP checkpoint identity，可讓未來的feature-level alignment少一套語意空間；但在第一版只傳遞class logits時，共用text space不是必要條件，也不能當成它優於linear classifier的理由。

**預先固定的quality gate：** 在開始完整比較前，以linear head的paired-validation macro-F1為基準。XCLIP text head若落後不超過2個percentage points，且NLL／ECE至少一項更好、另一項沒有明顯惡化，視為保留足夠teacher quality，可優先選language-aligned版本；若落後超過2點，或calibration與soft-target品質同時變差，主要teacher改用linear classifier。這個2點是Phase 1的工程門檻，不宣稱是通用科學定律；它必須在看完整validation結果前鎖定，不能事後移動。

**Reference怎麼保留：** OpenAI CLIP `ViT-B/16` frozen-text reference保留為條件式診斷ablation，用來在XCLIP結果異常時判斷問題是「文字對齊本身不適合」還是「選到不合適的text space」；它不再是Gate 3必跑項目，也不為每個backbone與每個prompt做完整組合搜尋。如果因診斷需要而執行且OpenAI reference較好，它仍先作ablation結果，除非validation增益足以抵銷引入第二套runtime／provenance contract的成本，才另提變更主要teacher的決策。

## 11. 定案後的 Phase 1 路線

正式路線現為：先完成Gate 0資料audit；再讓BlockGCN與Hyperformer完成joint-only compatibility smoke與short bake-off；用途確認為研究／教育時可加入SkateFormer challenger；所有正式run從random initialization開始；先以linear head的兩個seeds找出backbone ceiling；最後在勝出backbone上先訓練frozen XCLIP text-prototype head seed 42並套用預先固定的quality gate。明顯未通過便選linear classifier teacher並結束主要比較；接近或通過才補XCLIP seed 43。OpenAI CLIP reference只在需要診斷text-space差異時執行。通過quality gate才把text-aligned版本定為主要SkeletonCLIP teacher，否則第一版distillation使用較強的linear classifier teacher並如實命名。

第一個 implementation 切片是 Gate 0 data contract，而不是 backbone training；該切片已完成資料 parser、有效性檢查、canonical pairing 與 coverage artifacts，並在 review 後依使用者決策排除 A060，再進入 Gate 1。

Gate 1 已在 NVIDIA GeForce RTX 2080 Ti 上完成。BlockGCN 與 Hyperformer 的 project-owned minimal adapters 均通過 CPU shape、單 GPU forward/backward、finite loss/gradients、16 個真實跨類別樣本的 100% overfit、strict state-dict reload 與 eval-logit parity；GPU parity tolerance 預先鎖為 `atol=rtol=1e-5` 並保留 raw maximum difference。BlockGCN 在 50 steps 完成，peak reserved memory 約 1.26 GiB；Hyperformer 在 80 steps 完成，peak reserved memory 約 3.99 GiB；兩者都保留超過 30% VRAM headroom。完整數字與 sample IDs 記錄於 `artifacts/skeletonclip_gate1/compatibility_report.json`，下一個 review 點是是否啟動 Gate 2 short controlled bake-off。

## 12. 已確認的用途與共用主機執行政策

### 12.1 專案用途

目前將專案用途視為未確定，不假設永遠限定研究／教育。因此license-safe路徑固定為BlockGCN primary與Hyperformer reference；SkateFormer暫不納入必要implementation或正式teacher selection。未來若用途確認符合其授權，再以獨立challenger加入，不影響先前結果。

### 12.2 先證明能完整收尾，再花正式算力

每個階段先執行最小但完整的end-to-end run，而不只確認forward能動。小規模run必須完成資料讀取、train step、validation、checkpoint save、strict reload、metrics與predictions寫出、run metadata以及正常結束；產物必須可供review和後續比較。只有這個run完整通過，且估算的正式訓練時間、memory與artifact大小可接受，才進入較長訓練。失敗的smoke也保存failure reason與已消耗時間，避免重複花算力踩同一個問題。

正式升級順序固定為：CPU parser／manifest小樣本檢查；16-sample單GPU overfit；小型train／validation end-to-end run；一個候選的一個short bake-off；確認結果可用後才擴展到第二候選或完整訓練。不會同時啟動多個GPU trials，也不會在smoke尚未能strict reload與產出完整report時排長run。

### 12.3 啟動前的主機負載檢查

每次GPU工作開始前直接查詢實際GPU型號、數量、memory、utilization與現有process，並檢查CPU load、可用memory與目前重工作業。硬體資訊不要求使用者手動提供；每個run把偵測結果寫入metadata。若其他使用者正在大量使用候選GPU或主機CPU，就不啟動新訓練。

等待期間以15分鐘為heartbeat週期重新檢查相同資源狀態並回報，不用密集polling，也不因等待而改用正在忙碌的其他GPU。資源恢復後仍先重新跑preflight，確認預估run能在可用時段完成才啟動。

### 12.4 資源使用上限

Smoke與第一輪bake-off預設只使用一張當下空閒的GPU，不使用DDP或多GPU。Batch size由小往上找，但目標不是填滿VRAM；預設保留至少約30% GPU memory headroom，若實測有其他共享需求則保留更多。GPU utilization可以在單一步驟短暫升高，但若持續接近飽和、影響共享使用，就降低batch／worker或暫停，而不是追求最高throughput。

DataLoader workers預設從2開始，沒有I/O瓶頸證據不增加，第一輪上限為4；不以全部CPU cores做解析、解壓或資料轉換。資料audit採有界worker數與可恢復／可重跑的分段輸出。Host memory同樣保留至少30% headroom，不建立會同時把全部skeleton載入RAM的dataset設計。

### 12.5 正式訓練的go／no-go條件

只有在小規模run正常收尾、checkpoint可strict reload、validation與artifact可信、沒有NaN／OOM、資源峰值符合上述headroom，而且依實測throughput推算能在合理共享時段完成時，才建議正式訓練。開始正式run前先回報預估duration、GPU與CPU配置、預期artifact與停止條件；若條件不滿足，就先縮小或修正，不留下「已花大量算力但結果無法使用」的半成品作為實驗結論。
