# Patch16 skeleton 融合實驗計畫

日期：2026-10-02。狀態：計畫已定稿，尚未實作或執行融合訓練。分支 `patch16-skeleton` 從 `patch16-resolution` 的 `7510c0c` 建立；此次交付只包含本文件，不啟動 GPU jobs。後續實作者應依本文重建實驗，不需要其他工作分支或對話記憶。

## 1. 目標與候選排序

目標是檢查 skeleton training-only supervision 能否在已訓練好的 patch16 RGB model 上改善 raw120 classification，同時保護原本 RGB 能力。訓練只使用 whole view；完整 validation 同時評估 whole、center zoom70、whole＋zoom ensemble。Whole 或 ensemble 可分別成為候選，必須明確區分單次與兩次推論成本。

| 優先順序 | 方法 | 選擇原因 | 主要風險 |
| --- | --- | --- | --- |
| 1 | Semantic distillation＋RGB retention | 既有 semantic pilot 的 paired 決策有正向訊號，且不依賴 patch grid；此次互補性檢查仍找到大量 teacher-only correct | Teacher 只覆蓋59類，盲目模仿會損失 RGB 原本正確的決策 |
| 2 | Skeleton-guided temporal importance | 提供時間訊號，可能補足 patch16 已改善的空間辨識；沒有直接依賴 teacher 最終 class logits | 尚未實測；motion 大不保證有判別性，八幀缺失的事件無法藉權重補回 |
| 3 | Spatial joint supervision／occupancy probe | 舊實驗有 correct／CE／shuffled 比較及小幅定位訊號；196 patches 可提供更細的監督 | 定位提升不等於分類改善；NTU joints 不描述手指與物件 |

排序是探索優先順序，並非量測出的成功機率。目前沒有上述方法在 patch16 上的融合增益證據。三個方法各自從同一 patch16 baseline 開始，不能串接前一候選的 checkpoint；不將 skeleton 作為 inference input，不做 reduced taxonomy 或產品 label filtering，不讀 test。

## 2. 固定 baseline、資料及資源

### 2.1 RGB baseline

| 項目 | 固定值 |
| --- | --- |
| Backbone | `microsoft/xclip-base-patch16` |
| Revision | `d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6` |
| Input | Whole-video uniform8、224×224、14×14＝196 patches |
| Adapter | `artifacts/patch16_resolution/pilot_seed42/vision_lora_ce_seed42/best_adapter.pt`，epoch20 |
| Whole baseline logits | 同目錄 `best_validation_logits.pt` |
| Whole／zoom baseline logits | `artifacts/patch16_resolution/whole_zoom_validation/view_predictions.pt` |
| Baseline run lock | Pilot adapter 同目錄 `run_lock.json` |
| LoRA | Upper-two vision blocks＋MIT q/v，rank16、alpha32、LoRA dropout0.05，共6個target modules |
| Main classifier | 120-class；head dropout0.25、feature noise std0.01 |
| Canonical split | `artifacts/xclip/selected_split.json`；80,640 train／12,960 validation |

| Validation view | Raw120 accuracy | Correct／12,960 | A042–A048 correct／504 |
| --- | ---: | ---: | ---: |
| Whole | 74.1204% | 9,606 | 358 |
| Center zoom70 | 78.9738% | 10,235 | 383 |
| Whole＋zoom probability mean | 80.9568% | 10,492 | 385 |

Whole macro-F1為75.4532%。所有其他baseline macro-F1與subgroup metrics必須從保存的logits、labels及sample IDs重算，與候選使用同一reporting implementation。Accuracy計算保留全部120個logits，沒有class aggregation或移除logits後renormalization。

### 2.2 Skeleton teacher、cache及identity

- Paired manifest：`artifacts/skeleton_data/paired_manifest.json`，45,236 paired train／4,180 paired validation，ordered class IDs固定A001–A059。Class index為action ID減1；A060–A120不受teacher59 supervision。這些paired樣本只是canonical RGB split的子集，不是另一個split。
- Semantic teacher：`artifacts/skeletonclip_gate3/xclip_seed42/best_checkpoint.pt`，BlockGCN、text-aligned59、64 skeleton frames，paired validation accuracy86.2201%。Teacher的text backbone為`microsoft/xclip-base-patch32`、revision `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`；它的text identity與student patch16 identity分開保存及驗證，不能直接替換prototypes或放寬既有provenance checks。
- Skeleton tensor cache：`cache/skeletonclip_gate3/train_45236_4180_64.pt`與`val_45236_4180_64.pt`。驗證既有preprocessing ID、ordered sample IDs、labels及shape `[N,3,64,25,2]`，與paired manifest及raw skeleton抽查一致；teacher固定、eval、不更新BatchNorm statistics。
- RGB cache：`cache/skeletonclip_phase3_rgb_uint8`，使用patch32 cache identity及其revision。先確認patch16／patch32的`image_processor.to_dict()`逐項相等，再沿用processed pixels；新student identity仍記錄patch16。
- Spatial／temporal targets使用原始skeleton資料及canonical RGB frame indices，不以64-frame teacher tensor代替原始timing／color coordinates。Spatial新cache固定16px patches與196-patch shape，不能沿用49-patch cache；temporal新cache保存自己的target version。
- 工作環境沿用本機venv、artifacts及cache資源；它們通常不受Git追蹤。新worktree須明確連接這些資源，或提供相同relative paths；缺資源時先報告缺項，不下載不明teacher、不生成替代split、不默默改model revision。
- 任一model／split／teacher／cache identity、class order或sample pairing不一致時fail closed。Run lock保存baseline adapter、teacher、split、paired manifest、runner與model source hashes；target cache包含source fingerprint、sampling／processor幾何及target version。

### 2.3 已做過的patch32實驗

| 方法 | 訓練及比較契約 | 已記錄結果 | 對此次計畫的意義 |
| --- | --- | --- | --- |
| SKI-style online semantic | 8,192 train、1 epoch；58.9198% baseline；partial59，無matched CE-only續訓control | 58.9815%，+0.0617 pp；paired +19 correct、unpaired −11 | 有paired訊號，但需保護RGB；本次改為frozen teacher及retention |
| Phase4 joint occupancy probe | 完整train、20 epochs；CE／correct／shuffled三組，共同初始化 | Epoch20 CE58.5417%、joint58.6188%，+0.0772 pp；patch AP小幅改善 | 最完整的spatial control證據，分類收益仍很小 |
| Native attention supervision | 完整train、2 epochs；partial59，無matched CE-only續訓control | 58.9429%，+0.0232 pp；A042–A048 +10，但保留產品類別−33 correct | Maps改善不足以選分類winner；此次不移植此方法 |
| Direct decision KD＋unpaired retention | 8,192 train、1 epoch；frozen teacher | 58.9583%，+0.0386 pp；paired −15、unpaired +20 | 保護漂移可運作，未證實teacher帶來paired收益 |
| Motion pooling，baseline固定 | 8,192 train、1 epoch；只訓練attention／alpha | 58.9429%，+0.0232 pp，+3 correct | 沒有實用收益，不重跑此pooling家族 |
| Motion pooling，joint adaptation | 8,192 train、1 epoch | 58.1096%，−0.8102 pp | 定位機制成立但分類退步 |
| Frozen logit masked-KL，MIT-only | 完整train、60 epochs；9組T／weight search，共同CE control約52.1065% | 全部accuracy低於CE，差−0.1080至−1.6204 pp | 不原樣重跑；與本次有vision adaptation的semantic方法不同 |
| Skeleton-guided temporal | Proposal，未實測 | 沒有classification結果 | 本次是新機制screen，不承諾收益 |

相關基底文件為[semantic](ski-semantic-v1-plan.md)、[decision KD](decision-distillation-v1-plan.md)、[Phase4 occupancy](skeletonclip-phase4-gate2-results.md)、[motion pooling](spatial-attention-v2-results.md)、[masked-KL](skeletonclip-distillation-phase2.md)、[temporal proposal](skeleton-guided-temporal-supervision-plan.md)及[patch16 baseline](patch16-resolution-plan.md)。Native attention的實作與文件位於其他分支，本文已保留決策所需結果；本次不依賴它。不同budget及control不能當成公平方法排名。

## 3. Patch16／skeleton互補性檢查

2026-10-02的唯讀分析使用已保存validation logits，沒有新model inference、訓練或test evaluation。主teacher是原始text-aligned `xclip_seed42`，不是semantic pilot共同訓練後的teacher。所有比較以sample ID對齊相同4,180 paired validation；RGB仍在全部120個outputs取argmax，teacher在59類取argmax。

| Patch16 view | Paired RGB accuracy | Teacher對／RGB錯 | RGB對／teacher錯 | 兩者皆錯 | Teacher答對RGB錯誤的比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole | 79.69% | 629 | 356 | 220 | 74.09% |
| Zoom70 | 82.68% | 532 | 384 | 192 | 73.48% |
| Whole＋zoom | 84.93% | 452 | 398 | 178 | 71.75% |

RGB conditional59 diagnostic accuracy分別79.76%、83.23%、84.98%，因此teacher優勢不是主要來自RGB跑到teacher vocabulary外。Teacher自身accuracy86.22%不能與full120的74.12%直接比較，因為樣本與label space不同。

Text-aligned／linear teachers與seeds42／43共四組檢查，whole teacher-only correct範圍614–629，ensemble為437–452；這是teacher sensitivity，不是patch16第二training seed確認。對主teacher而言，patch32 whole互補數1,063減至patch16的629，ensemble773減至452，約41%的舊互補機會已被更強RGB消除。

Teacher仍能補上大量RGB錯誤，但RGB也答對teacher錯誤，因此支持bounded pilot而非盲目模仿。Perfect correctness selector對full120 whole的paired補錯上限為+4.8534 pp，ensemble為+3.4877 pp；這不是可部署fusion或預期KD收益，不能據此設定訓練成績。所有結果來自validation-selected checkpoints，屬exploratory分析；沒有證明unpaired或A060–A120 transfer。

本機分析與完整24組比較位於`artifacts/patch16_skeleton_complementarity_v1/`，包含`analyze.py`、`run_report.json`及`sample_comparisons.csv`。可在資源完整的worktree執行`.venv/bin/python artifacts/patch16_skeleton_complementarity_v1/analyze.py`重現；如果未追蹤的script不可用，最小重現流程如下：

1. 載入paired manifest與canonical validation manifest，確認4,180／12,960 unique IDs及paired為canonical子集；逐ID核對label與action ID減1。
2. 載入`artifacts/skeletonclip_gate3/{xclip,linear}_seed{42,43}/validation_logits.pt`及result／prediction files，驗證finite `[4180,59]` logits、ordered classes、saved argmax及teacher accuracy。
3. 載入patch16 `view_predictions.pt`及patch32 `artifacts/whole_zoom_validation_v1/view_predictions.pt`，驗證finite `[12960,120]` logits、labels與IDs；對whole、zoom各取softmax，再等權平均形成ensemble。
4. 以paired IDs重排RGB與teacher rows，分別計算both-correct、teacher-only-correct、RGB-only-correct、both-wrong；四格總和必須4,180。按action計算相同四格，核對逐類總數等於aggregate。
5. Conditional59 diagnostic只對RGB前59類取argmax；oracle upper bound為teacher-only-correct／12,960。這些是分析結果，不修改模型或label space。

## 4. 共用loss、初始化與訓練契約

每個方法初始化original epoch20 patch16 adapter及main120 classifier，另持有完全凍結的original patch16 anchor。Retention在所有train樣本使用原RGB120軟分布；teacher／anchor target detach，不使用validation saved logits作訓練target。

```text
L_CE = CE(student_logits120, ground_truth120, label_smoothing=0.05)
L_retention = T² * mean_samples KL(softmax(anchor_logits120/T) || softmax(student_logits120/T))
T = 2; retention_weight = 1
L_total = L_CE + method_auxiliary_loss + L_retention
```

- Student只更新upper-two vision q/v LoRA、MIT q/v LoRA、main classifier與新增method head；base weights固定。共用student adaptation policy與舊temporal proposal的vision-fixed設定不同，須如實記錄。
- Student使用eval mode停用LoRA／classifier dropout與feature noise，但保留autograd及optimizer更新；初始化參數仍記錄既有dropout／noise設定。Teacher／anchor固定且eval，所有梯度與state均不得改變。
- 每個auxiliary loss在各自有效samples／frames取mean，避免coverage改變loss scale。Empty subset回傳可微分零；unpaired batch的CE與retention仍正常更新。
- 三種方法各自先做smoke，再使用全部80,640 train，最多10 epochs、seed42、batch4、FP32、不使用gradient accumulation。每epoch20,160 optimizer steps，完整十輪201,600 steps。
- AdamW對所有trainable groups固定LR`2.25e-5`、weight decay`5e-4`、betas`(0.9,0.999)`、eps`1e-8`；gradient clipping1。Cosine從第一輪鎖定十輪，以optimizer-step計數衰減至0，沒有warmup；提前停止不壓縮schedule。
- Workers預設24、hard cap24；CPU intra-op threads固定8，seeded DataLoader shuffle及worker initialization沿用既有helpers。Full epoch恰好80,640 sample exposures，不因paired與否重採樣。Model flags、seed、sample IDs與schedule進immutable run lock。

## 5. 方法一：Semantic distillation＋RGB retention

### 5.1 語意空間及prealignment

保留main120 head；MIT pooled512-dimensional features經identity-initialized `Linear(512,512)`、L2 normalization及teacher保存的normalized59-class text prototypes得到video semantic scores。Student scale固定為teacher初始learned scale，teacher、prototypes、scale完全固定。本次是frozen semantic adaptation，與舊online SKI-style pilot不同，不宣稱官方SKI／zero-shot重現。

- 先以全部paired train抽取凍結RGB pooled features，只訓練semantic projection。固定12 epochs、batch128、AdamW LR`1e-3`、weight decay`5e-4`，每epochseeded shuffle，沒有validation-dependent epoch selection。
- Held-out prealignment audit固定每個paired class8筆，共472 validation，使用seed42 sample-ID hash排序，不能挑teacher／RGB正確的樣本。Prealignment pass需CE低於before且accuracy高於`1/59`；main RGB logits與predictions不變。
- Smoke則以128 balanced RGB train內的paired subset prealign，使用同一472 audit，最多12個projection epochs。這個額外projection訓練不算八步model smoke或十輪主訓練，但另報耗時。

### 5.2 主訓練loss及部署

```text
L_aux = 0.1 * mean_paired CE(video_scores59, label59)
      + 0.01 * mean_paired MSE(video_scores59, frozen_skeleton_scores59)
L_total = L_CE + L_aux + L_retention
```

MSE沿用score矩陣元素mean定義，非KL或feature cosine loss。不加入無效的frozen teacher CE。額外video CE由舊設定1降至0.1，是避免auxiliary CE過度支配的本次固定選擇；不搜尋weight，不宣稱與paper等價。Pure semantic auxiliary loss須回傳到projection與vision／MIT LoRA；teacher無梯度。部署discard semantic branch、teacher及anchor，匯出原RGB120 classifier。

## 6. 方法二：Skeleton-guided temporal importance

### 6.1 Target construction

- 使用原始skeleton sequence，計算整段影片內state2 tracked joint observations最多的兩個body；同分按body ID遞增，缺body時不跨identity補值。
- RGB／skeleton frame counts須一致且RGB至少8frames，才提供temporal loss；不一致樣本仍計算CE與retention，保存排除原因。不默默使用normalized-duration映射或64-frame teacher indices。
- 使用canonical uniform8 frame indices及相鄰indices中點劃分八個nearest-frame時間cells，tie歸較早cell，首尾涵蓋整段影片。Raw XYZ減去root joint0，除以該body有效frames中joint0到joint20距離的median；距離須finite且大於`1e-6`。
- 只使用state2、finite joints及有效root。相鄰frames至少10個共同有效joints才計算velocity；三個連續frames至少10個共同有效joints才計算acceleration。Velocity為正規化位置一階差分norm的joint平均，acceleration為二階差分norm平均；多body以有效joint數加權合併。
- Velocity及acceleration各用該影片有效值95th percentile縮放並clip至`[0,1]`；percentile為0時該signal視為0，不除以0。Frame energy為`0.7*velocity + 0.3*acceleration`；只有缺acceleration時用0，不借用其他frame。對energy做三幀valid-aware moving average，再取每個cell有效energy平均。
- 八個cells都需至少一個可靠velocity觀測、總cell energy大於`1e-6`才監督。Target為`0.8*normalized_cell_energy + 0.2/8`；全靜態、缺cells或不可靠影片不提供temporal loss，不強制它們學uniform target。保存coverage、invalid reasons、entropy與motion curve。

### 6.2 RGB gate、loss及部署

```text
Projected frame features [B,8,512]
  → Linear(512,128) → GELU → Linear(128,1) → temporal softmax w
MIT input: x_t * (1 + w_t - 1/8)
L_aux = 0.1 * mean_valid_samples KL(target_time || w)
L_total = L_CE + L_aux + L_retention
```

最後scoring linear的weight／bias初始化0，因此初始w為uniform、MIT輸入與原模型相同；first linear使用seed42預設初始化。KL在FP32計算、使用`1e-8`數值保護。Trainable policy沿用第4節，不凍結既有vision LoRA。部署保留RGB temporal gate，仍八幀、RGB-only、一次backbone forward，但多小型branch；報新增parameters及latency，不能宣稱gate由LoRA merge移除。Whole／zoom各使用自己的RGB features產生w，不讀skeleton、不複製whole的gate到zoom。

## 7. 方法三：Spatial joint supervision／occupancy probe

- 沿用`XCLIPSpatialLoRAClassifier`的training-only occupancy head：最後vision patch tokens `[B,8,196,768]`經`Linear(768,1)`預測occupancy；main RGB path不新增pooling。Spatial head預設初始化固定seed42，不改main logits。
- 原始color coordinates經實際processor resize／center-crop幾何映射到224輸入。使用canonical八幀、既有top-two-body規則，合併多人occupancy；有效frame至少10個state2、finite、in-crop joints，crop外不夾回邊界。
- 固定patch size16、14×14 grid及binary occupancy `[8,196]`，不平滑target、不加motion weighting。Static有效frames仍可監督。Raw alignment須沿用既有audit契約，不能只因indices未超出範圍就假設alignment正確。
- 新建`cache/patch16_skeleton_targets_v1/spatial/`，保存196-patch shape、body selection、tracked validity、canonical indices、processor幾何、source fingerprints及target version，不讀寫舊49-patch target namespace。

```text
L_aux = 0.1 * spatial_region_loss(spatial_logits, occupancy, valid_frame_mask)
L_total = L_CE + L_aux + L_retention
```

Balanced BCE沿用現有`spatial_region_loss`：每frame positive weight為`clip((196-positive_count)/positive_count,1,16)`，先mean196個patch losses，再mean有效frames。Loss梯度進入spatial head及vision LoRA；CE也更新原RGB分類路徑。Invalid frames的occupancy全零且不參與auxiliary loss；all-invalid提供可微分零。部署移除spatial head、teacher及anchor，匯出原RGB120架構。

## 8. Whole／zoom／ensemble、selection及停止政策

### 8.1 每epoch的完整view評估

每個epoch結束後使用同一checkpoint，在全部12,960 canonical validation評估whole及center zoom70。Zoom沿用既有固定70%-short-side中央crop政策及whole-video八幀，不使用person detector。保持sample IDs及labels完全一致，保存兩組raw logits，以各自softmax形成第三組ensemble：

```text
p_whole = softmax(logits_whole120)
p_zoom = softmax(logits_zoom120)
p_ensemble = (p_whole + p_zoom) / 2
prediction = argmax(p_view)
```

不得改成logits平均、搜尋view weights、移除outputs或套用taxonomy；三種view各自對照第2節自己的baseline。Whole／zoom各一次forward，ensemble兩次forward；兩個view的logits重用，ensemble不額外forward。Model一律eval，Temporal保留自己的gate。這與LoRA weight merge是兩種不同操作。

每個method分開保存`best_whole`及`best_ensemble`，依各自raw120 accuracy取best，同分取較早epoch，保留該epoch所有view／subgroup metrics。另維護各自通過guardrails的best promising checkpoint，避免最高accuracy但違反保護條件的checkpoint掩蓋較早合格候選；zoom只作診斷，不單獨通過方法。

### 8.2 Early kill與十輪上限

- Epoch2早期止損：epoch1及epoch2的whole、zoom、ensemble六個accuracy結果，全部各自比baseline低至少`0.50 pp`才立即kill，保存`early_regression`原因，不跑epoch3。任一view／epoch未達此退步幅度，就繼續。
- 其他正常run最多10epochs。十輪是固定compute cap，不是patience；出現新best也不重設上限。Scheduler從開始就是十輪，不先用兩輪cosine再延長。
- 技術失敗，包括OOM、non-finite、identity mismatch或reload／export不一致，立即停止並記錄；不能改batch、AMP、loss或資料繼續當成同一run。
- 到epoch10結束，若沒有任何whole／ensemble checkpoint通過promising，就停止此方法，不自動延長epochs、搜尋weights或重跑更多seeds。Best仍保存供分析。

### 8.3 Promising及terminal classification

Whole或ensemble任一view可通過，但同一checkpoint須同時符合該view的全部條件；不能把不同epochs／views的優點拼成一次pass：

1. Raw120 accuracy相對該view自己的baseline至少`+0.20 pp`；12,960筆上至少淨增26correct。Whole需至少9,632 correct，ensemble至少10,518 correct。
2. 該view的macro-F1不下降；unpaired及A060–A120 accuracy各不下降超過`0.20 pp`，按其實際subgroup denominator計算。
3. 該view的A042–A048 correct不下降：whole至少358／504，ensemble至少385／504。

有符合條件的checkpoint記為single-seed `promising`，分別標示whole／ensemble可用性與成本，但第一輪仍受十輪上限約束。沒有pass且最佳whole及最佳ensemble raw gains都負值，記為`no-go`；其餘小幅增益、持平或guardrail未通過記為`inconclusive`。Early kill屬`no-go`，技術中斷另記`failed`，不能當成方法無效。

第一輪沒有matched完整CE＋retention續訓control，因此即使pass也不能將全部收益歸因於skeleton。Promising後才另行規劃matched comparison與第二seed；Temporal confirmation須包含相同gate但只用CE／retention的RGB-only gate對照。未完成confirmation不選deployment winner、不讀test。

## 9. Audit、smoke、輸出及驗證

### 9.1 Minimum checks

- 每method smoke固定128 balanced train／120 balanced validation，seed42 sample-ID hash及equal-class round robin，最多8optimizer steps；同一audit為每paired class8筆、共472validation。Smoke成績不當成classification winner。
- Initial main logits在120-class smoke及472 paired audit上與saved baseline argmax完全一致，FP32 `allclose(atol=2e-4,rtol=1e-5)`；核對pixel cache parity、targets／teacher pairing及finite tensors。
- 檢查pure auxiliary gradients進入預期head／vision／MIT，CE／retention gradients及gradient whitelist；teacher／anchor參數、buffers與gradients不變；empty paired／all-invalid loss可微分零，原CE繼續更新。
- Semantic prealignment驗證held-out CE下降、accuracy高於chance，main120 logits不變；不得依validation挑prealignmentepoch。Temporal identity gate及Spatial新增probe皆需證明初始化main logits不變。
- Temporal／Spatial各做correct／shuffled target短screen：固定相同32有效train samples與32有效audit samples、8steps、相同initialization，train targets使用seed42固定derangement，held-out維持correct target。Correct held-out auxiliary loss須低於shuffled才進pilot；沒有足夠有效samples或不通過時記錄並停止，不自動加steps。這只檢查機制，不是完整classification control。
- Temporal保存固定472 motion curves、target／predicted weights及entropy；在前32個有效audit samples替換最高／最低／seeded random權重frame，報correct-class logit下降與time-order perturbation。Static／invalid樣本另列，不排除於full classification。
- Spatial保存固定472 overlays、patch AP、occupancy-count top-k recall、center prior及valid coverage。定位數值不能替代raw120 pass。
- Semantic／Spatial discard auxiliary heads後，LoRA merge與clean RGB-only strict reload在FP32使用既有`atol=1e-5,rtol=1e-5`契約，另報max error。Temporal匯出包含gate的RGB-only架構並驗證相同parity，不能把plain classifier strict reload當成完整temporal部署驗證。

### 9.2 Interface、artifacts與運行

後續新增統一入口`python -m vlm.training.patch16_skeleton --method semantic|temporal|spatial --mode audit|smoke|pilot`；方法及model／teacher／cache identity為獨立設定，不依賴patch32 module-level constants。預設artifacts root為`artifacts/patch16_skeleton_v1/`，按method、mode、seed建立獨立immutable namespace；已有lock不得覆寫。此次沒有新增這個runner，以下命令是後續實作介面規格，現在不可當成可執行入口。

```bash
.venv/bin/python -m vlm.training.patch16_skeleton --method semantic --mode audit
.venv/bin/python -m vlm.training.patch16_skeleton --method semantic --mode smoke
.venv/bin/python -m vlm.training.patch16_skeleton --method semantic --mode pilot
# 同樣依序執行 temporal，再執行 spatial；各用自己的passing audit/smoke。
```

- 從基底已有的semantic、spatial、dispatch及cache能力擴充；新增temporal gate／target builder。找出並移除新runner中的patch32 revision、49 patches、7×7等硬編碼；不要改寫已完成歷史run或無關training入口。
- 每epoch保存兩組raw120 logits、sample IDs、labels、derived ensemble及各view predictions／metrics。另保存per-action confusion、fixes／regressions、paired／unpaired、A001–A059／A060–A120、A042–A048 metrics；paired由manifest membership決定，不能只以class range推斷。
- 每view報accuracy、macro-F1、CE／NLL及樣本數，ensemble的NLL由其平均probabilities計算。預測payload標示`test_evaluated=false`，數值必須可由保存prediction rows獨立重算。
- 每run保存immutable config/source hashes、target version、model/runner snapshot、trainable audit、per-step loss breakdown、各epoch metrics、best／last checkpoints、optimizer／scheduler及RNG states、停止原因及finite／parity結果。Checkpoint以原子方式寫入，不把teacher／anchor weights打包到部署export。
- GPU長任務使用tmux及既有dispatcher，保存console／train log、progress及terminal status。記錄本次執行hostname、GPU name／UUID、開始／結束時間、train／validation／prealignment耗時、examples/s、peak allocated VRAM及deploymentparameter／latency差異，不使用舊machine snapshot當成當次測量。
- Full-view validation每epoch有額外成本，須實測並記錄；不能把八步smoke的model-step time當成完整epoch吞吐。Workers hard cap24，資源不足先記錄原因，不自行改recipe。

### 9.3 文件及後續implementation驗收

此次文件驗收只做source／metric核對、Markdown單行paragraph規則、relative links及`git diff --check`，不新增persistent tests或啟動GPU jobs。後續實作只增加保護重要契約的tests：partial59 identity與empty masks、196-patch targets、temporal timing及identity gate、frozen anchor preservation、view probability mean、early-kill／十輪上限／同checkpoint guardrails，以及各架構RGB-only export／reload。使用既有model／cachechecks，避免複製實作細節的測試。

## 附錄A：互補性逐類結果與輸入指紋

以下表格及SHA256直接取自已完成的唯讀分析report，讓本分支即使沒有其他分支記憶仍可核對證據。逐類使用原始`xclip_seed42` teacher與patch16 whole；net為teacher-only correct減RGB-only correct，不是蒸餾後的收益。

| Action | n | RGB correct | Teacher correct | Teacher-only correct | RGB-only correct | Net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A034 | 72 | 19 | 65 | 48 | 2 | +46 |
| A047 | 72 | 22 | 49 | 30 | 3 | +27 |
| A045 | 71 | 39 | 64 | 27 | 2 | +25 |
| A049 | 72 | 38 | 63 | 26 | 1 | +25 |
| A037 | 71 | 44 | 63 | 22 | 3 | +19 |
| A004 | 70 | 39 | 57 | 20 | 2 | +18 |
| A010 | 70 | 42 | 60 | 24 | 6 | +18 |
| A027 | 72 | 55 | 71 | 16 | 0 | +16 |
| A046 | 72 | 54 | 70 | 18 | 2 | +16 |
| A042 | 71 | 54 | 69 | 16 | 1 | +15 |
| A011 | 69 | 34 | 47 | 21 | 8 | +13 |
| A025 | 72 | 53 | 65 | 16 | 4 | +12 |
| A035 | 72 | 59 | 71 | 13 | 1 | +12 |
| A038 | 72 | 52 | 64 | 16 | 4 | +12 |
| A056 | 72 | 51 | 63 | 17 | 5 | +12 |
| A022 | 71 | 60 | 71 | 11 | 0 | +11 |
| A023 | 71 | 43 | 54 | 17 | 6 | +11 |
| A039 | 72 | 59 | 70 | 13 | 2 | +11 |
| A001 | 68 | 50 | 60 | 15 | 5 | +10 |
| A040 | 69 | 57 | 67 | 12 | 2 | +10 |
| A002 | 71 | 52 | 59 | 12 | 5 | +7 |
| A015 | 71 | 62 | 69 | 9 | 2 | +7 |
| A018 | 71 | 59 | 66 | 10 | 3 | +7 |
| A026 | 72 | 64 | 71 | 8 | 1 | +7 |
| A028 | 72 | 52 | 59 | 14 | 7 | +7 |
| A033 | 72 | 63 | 69 | 8 | 2 | +6 |
| A008 | 71 | 66 | 71 | 5 | 0 | +5 |
| A007 | 71 | 65 | 69 | 5 | 1 | +4 |
| A031 | 72 | 54 | 58 | 11 | 7 | +4 |
| A054 | 72 | 61 | 65 | 11 | 7 | +4 |
| A006 | 70 | 66 | 69 | 4 | 1 | +3 |
| A036 | 72 | 69 | 72 | 3 | 0 | +3 |
| A041 | 72 | 53 | 55 | 12 | 10 | +2 |
| A003 | 71 | 54 | 55 | 9 | 8 | +1 |
| A021 | 72 | 68 | 69 | 4 | 3 | +1 |
| A009 | 69 | 68 | 68 | 1 | 1 | +0 |
| A019 | 71 | 63 | 63 | 6 | 6 | +0 |
| A052 | 67 | 62 | 62 | 4 | 4 | +0 |
| A058 | 72 | 65 | 65 | 6 | 6 | +0 |
| A014 | 71 | 70 | 69 | 0 | 1 | -1 |
| A020 | 72 | 68 | 67 | 4 | 5 | -1 |
| A044 | 71 | 61 | 60 | 7 | 8 | -1 |
| A050 | 70 | 56 | 55 | 10 | 11 | -1 |
| A051 | 70 | 66 | 65 | 3 | 4 | -1 |
| A053 | 72 | 64 | 63 | 4 | 5 | -1 |
| A055 | 69 | 68 | 67 | 1 | 2 | -1 |
| A059 | 63 | 63 | 62 | 0 | 1 | -1 |
| A013 | 69 | 66 | 64 | 3 | 5 | -2 |
| A043 | 71 | 69 | 67 | 2 | 4 | -2 |
| A048 | 72 | 56 | 54 | 7 | 9 | -2 |
| A024 | 71 | 70 | 66 | 1 | 5 | -4 |
| A029 | 72 | 57 | 52 | 8 | 13 | -5 |
| A032 | 72 | 48 | 41 | 8 | 15 | -7 |
| A057 | 71 | 63 | 55 | 4 | 12 | -8 |
| A005 | 70 | 62 | 48 | 3 | 17 | -14 |
| A012 | 68 | 51 | 36 | 6 | 21 | -15 |
| A016 | 71 | 64 | 46 | 4 | 22 | -18 |
| A030 | 72 | 57 | 36 | 9 | 30 | -21 |
| A017 | 71 | 62 | 34 | 5 | 33 | -28 |

### Source SHA256

| Source | SHA256 |
| --- | --- |
| `artifacts/xclip/selected_split.json` | `2f36070622c4456a3bac390c8851a499acf17ae2a5e91e363e650cb19324ff16` |
| `artifacts/skeleton_data/paired_manifest.json` | `48ec0d2c52656babae7c9dfccf8ab0705ce89c1f23e6c49b6edb65f6d0e87d3e` |
| `artifacts/patch16_resolution/whole_zoom_validation/view_predictions.pt` | `22709d87c8b1ebdadfd26e88ad7e4f4238eac0aca6a29c634531a3921f753c81` |
| `artifacts/whole_zoom_validation_v1/view_predictions.pt` | `b8e354acbecb2ade80ca47feff95a4b8e9b79372a11af5f9bda7f42c8d3dfc81` |
| `artifacts/skeletonclip_gate3/xclip_seed42/validation_logits.pt` | `0e621f3170a2655f11245403f2e73640988c24b73c54d7e706b447378b2160e7` |
| `artifacts/skeletonclip_gate3/xclip_seed42/result.json` | `3d4146717a48b6368ce389e0cf70d71519249346e75f93dbe6fc8180a0e1448c` |
| `artifacts/skeletonclip_gate3/xclip_seed42/validation_predictions.json` | `ca18f83a9a420eccdc88bbb3380fe1538f635f06bad80c2510ca83d83245c481` |
| `artifacts/skeletonclip_gate3/xclip_seed43/validation_logits.pt` | `f30c84a0ebd4192f6aeeb180123ef00b386ea476f57a1d3b172c9bb0c0e3d82f` |
| `artifacts/skeletonclip_gate3/xclip_seed43/result.json` | `1da75229520278a85502d2f239dc91892b062cf6870bab484750af5cf7c9f427` |
| `artifacts/skeletonclip_gate3/xclip_seed43/validation_predictions.json` | `54921f6848d0fc6c1fb545649443c86bf254bda2d57d6d111283dd5b90a57f40` |
| `artifacts/skeletonclip_gate3/linear_seed42/validation_logits.pt` | `e692fc240749e9fd7f14041e8ba77fd042eea1f1455453e7b7c0053364c39cf5` |
| `artifacts/skeletonclip_gate3/linear_seed42/result.json` | `a92beba05ca9f02cc61542df10d1f79445453907b26b43ecf0cf4b79bfc98b70` |
| `artifacts/skeletonclip_gate3/linear_seed42/validation_predictions.json` | `1dd6bf52718b94da57a3d8042cb5b234b2ef8161417ef04aee6a8449225dd112` |
| `artifacts/skeletonclip_gate3/linear_seed43/validation_logits.pt` | `ac762c72fa1f5d08a1d15ba37abcf0ef4e1aee298b31cba97cf72811a32f4b8b` |
| `artifacts/skeletonclip_gate3/linear_seed43/result.json` | `f7f37f124665553c7480c6944a41cedbfe61177dc8dede2700d6fe976fe12c0d` |
| `artifacts/skeletonclip_gate3/linear_seed43/validation_predictions.json` | `bcd1efb86953beff8f566f8b947958e73569720c3d9ed5448487826fe269b9e9` |
