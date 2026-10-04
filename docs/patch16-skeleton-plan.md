# Patch16 skeleton 融合實驗計畫

日期：2026-10-03。狀態：semantic audit／smoke已通過，pilot進行至epoch8；whole／ensemble已有promising validation成績但尚未完成pilot。依使用者指示暫停後續methods自動啟動，平行加速短測已完成。分支 `patch16-skeleton` 從 `patch16-resolution` 的 `7510c0c` 建立；最初文件提交未啟動 GPU jobs，後續依使用者指示實作及執行。本文保留原定實驗契約，不需要其他工作分支或對話記憶。

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

統一入口為`python -m vlm.training.patch16_skeleton --method semantic|temporal|spatial --mode audit|smoke|pilot`；方法及model／teacher／cache identity分開保存，不依賴patch32 model constants。預設artifacts root為`artifacts/patch16_skeleton_v1/`，按method、mode、seed建立獨立immutable namespace；已有lock不得覆寫。`--artifacts-dir`可指定新namespace，搭配`--audit-dir`／`--smoke-dir`指定相同sources的passing gates。

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

原始文件提交只做source／metric核對、Markdown單行paragraph規則、relative links及`git diff --check`。後續實作只增加保護重要契約的tests：partial59 identity與empty masks、196-patch targets、temporal timing及identity gate、frozen anchor preservation、view probability mean、early-kill／十輪上限／同checkpoint guardrails，以及各架構RGB-only export／reload。使用既有model／cachechecks，避免複製實作細節的測試。

### 9.4 本次實作與執行紀錄

- 依使用者要求於Method 1進行中平行建立共用FP32 anchor cache，新增`patch16_anchor_cache.py`，namespace `shared_anchor_cache_v1/`。Canonical80640筆train以固定batch4、whole-view原processed uint8像素、original epoch20 RGB anchor、eval／frozen／FP32產生raw120 logits，按1024筆原子提交shards與checksum index；`--resume`只接續已核對的prefix，完成後核對所有IDs／labels、finite FP32 logits與source fingerprints，再原子publish combined logits及complete manifest。Consumer拒絕partial cache、identity／order／label／precision／checksum不符；cache identity含model／adapter／split／pixel config／video fingerprints與anchor builder function digest，resume control來源改動另存session snapshots，不能改動model／data依賴。
- 初次1024筆contention probe約118.6秒，cache8.64examples/s，Method 1從11.91降至7.53examples/s，顯示低顯存用量不代表GPU算力閒置。已從該shard續建full cache：CPU threads=2、workers=0、allocator fraction≤0.25，Method 1 PID265549存活時每batch讓出0.35秒，該PID退出後自動取消delay。這只確保提早準備cache，不宣稱總wall time必然縮短；20分鐘背景摘要也納入cache進度，原Method 1 training及pending-method hold未重啟或解除。

- Batch4 AMP＋cache初步驗證於`precision_validation_temporal_v1/`及`precision_validation_spatial_v1/`均pass。兩個method的FP32 cached／online trainable weights、AdamW moments、scheduler、loss traces與held-out logits完全一致；AMP在120筆held-out上的argmax、accuracy及macro-F1均與FP32一致，訓練後FP32 evaluation logits最大差異分別0.01119及0.01398。獨立重算與canonical val IDs／labels核對記錄於`reviews/precision_validation.json`；完整validation與長訓練accuracy等價仍未驗證，且尚未啟動加速pilot。

- 依使用者同意，AMP＋anchor cache的後續驗證固定batch4，先以`patch16_acceleration --method temporal|spatial --validate-precision`跑三個同初始化、同樣本順序、同10-step（2 warmup＋8 measured）更新與cosine排程的arms：FP32 online anchor、FP32 cached anchor、AMP cached anchor。每arm保存trainable weights／optimizer／scheduler／GradScaler及120筆獨立held-out whole-view FP32 evaluation logits、IDs／labels、accuracy／macro-F1／NLL／group metrics。要求FP32 cache arm的loss traces、全部trainable weights及validation logits與online完全相同；AMP記錄差異與finite gradients，不把subset screen當成完整pilot accuracy驗證。後續仍需recipe-specific audit／smoke及每epoch完整12960筆whole／zoom／probability-mean ensemble與原guardrails，正式pilot保持未啟動。

- 三種method共用`--resume <last_checkpoint.pt>`續跑入口，必須使用fresh artifacts namespace及current-source audit／smoke gates。嚴格核對method、data IDs／hashes、baseline／teacher／model依賴、batch／precision／loss／optimizer／scheduler契約及worker設定，只允許已通過當前gates的runner版本更新，不允許續跑時偷偷改batch或precision。續跑載入所有trainable weights、AdamW moments、cosine位置、training generator及Python／NumPy／Torch／CUDA RNG，略過semantic prealignment／mechanism screen，從完成epoch的下一輪接續，保留歷史metrics與earliest-tie best／best-promising checkpoints及RGB exports。
- 新版每個完整validation epoch保留`epochNN/checkpoint.pt`，所有best exports完成後才原子更新`last_checkpoint.pt`，避免export過程消耗RNG後存下過早狀態。若中斷發生在best檔更新與last檔更新之間，從immutable epoch checkpoint回復前一winner並重新export。續跑loader先建立persistent-worker iterator，再還原shared generator，避免額外worker base-seed draw改變下一輪shuffle。現階段續跑粒度是已完成epoch；epoch中途停機會重做該輪，尚未提供batch中途游標或prealignment中途續跑。
- 續跑回歸驗證：11項tests及6個method／worker subtests皆pass；三個真實模型的小型CPU設定在workers=0與persistent workers=1下，續跑與不中斷執行的trainable／frozen weights、AdamW moments、scheduler及下一輪shuffle完全一致，並驗證Python／NumPy／Torch RNG、錯誤recipe／source／scheduler拒絕、earliest ties與selection-update中斷恢復。舊semantic epoch7 checkpoint通過141120-step相容性檢查；尚未執行完整GPU continuation pilot，未重啟原訓練。
- Workflow支援重複`--resume METHOD CHECKPOINT`，僅將checkpoint傳給該method的pilot；audit與smoke仍依新sources先執行。例：`.venv/bin/python -m vlm.training.patch16_skeleton_dispatch --artifacts-root artifacts/patch16_skeleton_v1/continuation_01 --methods temporal spatial --resume temporal /absolute/path/temporal/pilot/seed42/last_checkpoint.pt`。原始training process未重啟，Method 2自動起跑hold保持有效；新版功能於未來啟動的新process生效。

- 2026-10-03使用者要求Method 1繼續、暫停Method 2並平行嘗試加速。`implementation_01/workflow_control.json`由新runner在GPU初始化前檢查，禁止pending temporal／spatial自動起跑；原semantic process繼續且未改recipe。背景tmux `patch16-monitor-20min`每1200秒append `monitor_20min.jsonl`並更新`monitor_20min_status.json`，這是本機摘要紀錄，不是聊天定時推送。
- 平行train-only benchmark位於`artifacts/patch16_skeleton_v1/acceleration_parallel_v1/`，各arm固定32筆有效temporal train samples、2步warmup及8步measurement，workers=0、CPU threads=2、GPU allocator fraction≤0.5，保留至少1GiB預算給原訓練。FP32/b4 online anchor、FP32/b4 cached anchor、AMP/b4、AMP/b8、AMP/b16 cached anchor分別為5.82、10.28、20.82、24.70、27.78examples/s，最快相對baseline為4.78×；五組finite／gradient gates皆pass。固定batch4 cached與online的loss traces完全一致，FP32 cached logits最大誤差0，batch16 parity最大誤差4.05e-5；AMP初始32筆中的各arm argmax checks全一致，但不代表validation accuracy等價。這些是GPU共用且RGB輸入預載的model-step measurements，不包含完整cache preparation／data loading／validation成本，不能推估完整epoch等比例加速；未啟動accelerated pilot、未碰test，後續須獨佔重測與recipe-specific audit／smoke。
- Semantic epochs1–7的saved whole／zoom logits及probability-mean ensemble已獨立重算，canonical IDs／labels、finite、metrics及same-checkpoint guardrails皆pass，見`reviews/semantic_epochs_1_7.json`。Whole best epoch7為74.5910%（baseline74.1204%，+0.4707pp）；ensemble best epoch6為81.4198%（baseline80.9568%，+0.4630pp），皆僅為promising。

- 新增`src/vlm/training/patch16_skeleton.py`、`patch16_skeleton_dispatch.py`、RGB-only temporal model、versioned raw target cache及共用loss／selection contracts。Workflow依semantic→temporal→spatial執行，每method使用自己的audit、smoke及固定十輪上限pilot；gate失敗停止該method，後續method仍從original baseline起跑。
- 八項新增重要契約tests通過：source snapshot namespace／hash、empty temporal masks／KL方向、196-patch occupancy／crop外joints、兩種method各自的top-two body policy、nearest-frame timing／static exclusion／translation、identity gate／frozen anchor／merged temporal reload、probability mean與同checkpoint guardrails／epoch2 early-kill。既有spatial targets兩項及vision LoRA三項tests也通過；這些不能替代GPU smoke。
- 初版semantic GPU audit於`artifacts/patch16_skeleton_v1/semantic/audit/seed42/`通過，包含完整120＋472初始化parity及fresh processor／uint8 cache equality。其後補齊診斷及gradient audits，故初版report不能作為定版sources的pilot gate。
- 定版workflow位於`artifacts/patch16_skeleton_v1/implementation_01/`，tmux session為`patch16-skeleton-v1`。各子run保留自己的`process.json`、`console.log`、`experiment_lock.json`、`progress.json`及terminal report；根目錄保存workflow results與dispatcher logs。查看process／tmux及當前progress再判定是否存活，不能只依既有lock判定。
- 依使用者要求啟動持續監看後，dispatcher logger已改為遞迴記錄各method／mode子run的progress及best／best-promising checkpoint變動，並通過nested-progress／selection／terminal logging smoke。新logger PID記錄於`dispatch.json`，替換過程保留於`monitoring_restart.json`；只替換logger，原workflow及training PID未重啟。監看輸出持續append至root的`train.log`。
- Spatial沿用既有manual landmarks，但對當前raw sources重新投影，要求median error≤16px且motion-informative landmarks的zero offset優於±1–3frames；另保存本次472 overlays。Temporal保存472 motion curves及RGB strips，invalid／static樣本不移除於classification。
- 定版semantic audit與八步GPU smoke均已pass，workflow已啟動semantic full pilot。Smoke prealignment固定12輪，472筆held-out CE由4.37264降至2.23080、semantic59 accuracy由0.8475%升至45.1271%；這只是semantic branch的mechanism gate，不能當成raw120 pilot增益。Pure auxiliary、CE及retention gradients、frozen teacher／anchor preservation與clean RGB-only merged strict reload均通過。
- Semantic pilot執行期間，後續source核對修正spatial body selection為既有sampled-frame top-two規則，temporal維持whole-sequence top-two；補上temporal merged model相對bypass gate的latency delta及mechanism short screen的32筆train／audit sample IDs。已執行的semantic子run繼續使用啟動時保存的source snapshot／hashes；其recipe及執行中的Python module不變，後續temporal／spatial會各自對更新後sources重跑audit／smoke，不能借用semantic gates。
- Source snapshot核對發現absolute path會繞過artifact namespace，已改為project-relative destination及寫入前SHA256驗證，並加入重要regression test。`implementation_01` semantic audit／smoke／pilot的15份Python sources均已補齊到各自`source_snapshot/`；修改過的兩份sources由已知差異反向還原，所有bytes逐一符合原immutable lock SHA256，未改run locks或重啟pilot。初版非定版audit不作為後續gate。
- 尚待semantic pilot及temporal／spatial各自的audit、smoke與pilot結果；未選deployment winner、未讀test。後續完成後應把實際status、compute、各view／subgroup結果及停止原因補入本節。

使用新root啟動整個workflow（existing namespace不可重用）：

```bash
MPLCONFIGDIR=/tmp/vlm-matplotlib .venv/bin/python -m vlm.training.spatial_dispatch \
  --artifacts-dir artifacts/patch16_skeleton_v1/implementation_01 \
  --tmux-session patch16-skeleton-v1 -- \
  .venv/bin/python -m vlm.training.patch16_skeleton_dispatch \
  --artifacts-root artifacts/patch16_skeleton_v1/implementation_01
tmux attach -t patch16-skeleton-v1
```

## 10. 後續候選：Patch16 native attention supervision

2026-10-04依使用者要求續寫。狀態為pending proposal，尚未移植native attention程式、執行audit／smoke或啟動pilot。這是原七方法排名的#3，為本workflow的第四個候選；現有Method 3仍是Spatial occupancy。只有Spatial正式完成或使用者明確停止後，才可另行實作及啟動本候選，不將文件更新解讀為停止Spatial或自動切換訓練的授權。

### 10.1 研究問題、歷史證據及方法差異

Spatial occupancy透過額外prediction head監督patch tokens；native attention直接監督XCLIP在實際聚合values時使用的CLS→patch attention probabilities，不新增learned attention／pooling head。問題是：在196-patch輸入上，直接約束原生attention，能否比額外occupancy probe更有效地把關節位置訊號轉成raw120分類增益？這個機制不同，不能因Spatial成績弱就預告Native會成功。

舊patch32 native pilot使用最後兩層固定四heads、smoothed occupancy target及完整train兩輪，raw120由58.9198%到58.9429%，只淨增3correct；A042–A048多10correct，但產品保留96類的9,792筆validation少33correct。Target KL與unconditioned joint-region mass改善，並未建立實用分類收益。Patch16上的移植只值得bounded screen，不因attention maps好看、smoke accuracy或patch數增加而延長預算。

本方法只使用原始skeleton-derived occupancy與原RGB anchor retention，不載入SkeletonCLIP teacher、不加入semantic MSE／direct decision KD、temporal gate或motion pooling。部署仍是原八幀RGB120 classifier、單次whole或兩次ensemble，不需要skeleton、detector或額外head。

### 10.2 原生attention介面與token indexing

- 新增opt-in `return_native_attentions=True`：回傳`(logits120, cls_attention)`，其中`cls_attention`為`[B,8,2,198]`，兩層各取heads0–3的CLS row平均。預設forward輸出不變，與`return_features=True`互斥；patch數須由input dimensions與model patch size推導，正式patch16 runner再檢查196。
- 正式backbone保留12vision layers／12heads、upper-two vision及MIT q/v LoRA；使用Transformers的eager attention及原生`output_attentions=True`，不改寫QKV計算、不加入hard mask、不移除XCLIP cross-frame message。Layers10／11、heads0–3固定，不依validation挑head。
- Spatial attention順序固定驗證為`CLS[0], patches[1:197], message[197]`，總198tokens。取得的attention須可微分；不能把detached診斷maps當成訓練loss輸入。普通獨立image ViT rollout不能替代XCLIP含message的attention契約。
- Patch probabilities以該CLS row的patch總mass重新normalize，CLS及message不進spatial target。Conditional KL不保證實際patch mass上升，因此必須另報未normalize joint-region mass、patch／CLS／message mass及conditional entropy。
- Pure native loss直接依賴Q／K，不直接依賴同層V。既有Q/V LoRA中，最後一層Q須有非零finite gradient，最後一層V在attention-only backward可能為零；V仍受CE／retention更新，不能將這個差異誤判為接線失敗。

可參考`skeleton-guided-attention`的`ceb95d9`提交及其`native_attention_targets.py`、native runner、model opt-in forward；僅移植所需能力，不整批cherry-pick其他分支。舊helper寫死49patches／51tokens／7×7，不能直接執行；當前patch16 model尚無此opt-in介面，所有新sources都須重跑gates。

### 10.3 196-patch target與固定loss

- 沿用Spatial已驗證的canonical八幀、processor幾何、sampled-frame top-two body selection、state2／finite／in-crop joint occupancy及有效frame門檻10；static frames可監督。先核對現有Spatial cache的source／geometry／target identity與`[8,196]` shape，不能只因檔案存在就reuse。
- 將binary occupancy reshape成14×14，用固定5×5 Gaussian、sigma1patch、zero padding卷積，normalize有效frame的總mass，再形成`target = 0.8*normalized_smoothed_occupancy + 0.2/196`。這保留約16px的Gaussian sigma，與舊patch32的sigma0.5patch物理範圍相近；不是直接把舊3×3kernel放大網格而保持不同物理尺度。
- 新target version為`native-p16-occupancy-gaussian1-uniform20-v1`。保存occupancy來源hash、grid／kernel／sigma／floor、frame mask及body policy；derived cache使用獨立namespace，不改寫Spatial原cache或舊native49-patch資料。
- Invalid frames的target全零，不參與KL；all-invalid batch提供可微分零。Target及attention normalization、KL一律FP32，使用`1e-8`保護；分布不finite、負值或有效target總和不為1則fail closed。

```text
A = mean_heads_0_to_3(native_CLS_attention), layers = [10, 11]
p = A[..., 1:197] / sum(A[..., 1:197])
t = 0.8 * normalize(Gaussian5x5_sigma1(occupancy14x14)) + 0.2/196
L_native = mean_valid_frames_and_two_layers KL(t || p)
L_total = CE120(label_smoothing=0.05) + 0.1*L_native + L_retention(T=2, weight=1)
```

Retention作用於全部train samples，anchor始終為原epoch20 patch16，而非Semantic／Temporal／Spatial winner。固定weight，不在這次pilot搜尋sigma、heads、loss weights或mask策略。Student沿用eval mode保留autograd、upper-two vision／MIT q/v LoRA及classifier更新，base weights凍結；不混入額外occupancy BCE。

### 10.4 Precision、anchor cache與mechanism gates

- 優先採用現行加速recipe：batch4、AMP FP16 student forward、FP32 native loss／CE／retention／validation，GradScaler與optimizer／scheduler／RNG完整保存。Eager attention與198tokens會增加顯存及耗時，須量測，不沿用Spatial smoke的資源數字。
- 原native patch32結果不能當作patch16 precision gate。先做同初始化、同samples／shuffle／optimizer的FP32 online anchor、FP32 cached anchor、AMP cached anchor三arms短screen，固定2warmup＋8measured updates及120筆獨立held-out FP32 whole evaluation；保存attention、loss traces、gradients、parameters／optimizer／scaler及argmax differences。FP32 online／cached須exact agreement；AMP須finite、held-out argmax一致，否則不啟動AMP pilot。
- 共用完整FP32 anchor cache僅在consumer仍能嚴格核對原adapter、pixels、IDs／labels、precision、payload checksum及原builder provenance時reuse。新增attention opt-in若改變source依賴，需明確核對原anchor snapshot與native-disabled輸出契約，不能覆寫cache hash或關閉identity checks。Reuse不相容則記錄阻擋，另以獨立recipe規劃online anchor，不默默fallback。
- 如果AMP precision gate失敗或OOM，停止並保存原因，不自行換batch、heads或backend；可另提出FP32成本方案，但不把不同precision的gates混用。Current native sources／precision／cache依賴必須一致，不能借用Spatial passing smoke。
- Audit使用同一固定472paired validation，保存native-enabled／disabled logits、occupancy／target／attention、raw及conditional mass、entropy與overlays。開啟attention輸出須保持FP32 logits parity及全部472argmax一致；代表圖不能當成全量alignment證據。
- Smoke固定128balanced train／120validation、最多8model optimizer steps；驗證pure loss到vision Q、CE／retention到vision／MIT／classifier、frozen anchor、empty masks、target方向、message exclusion及merged RGB-only strict reload。只在訓練／audit時回傳attention；clean deployment forward不要求輸出attentions。
- 沿用correct／shuffled短screen：32有效train／32有效audit、固定derangement、同初始化八步，held-out均對correct target計算KL。Correct held-out KL須低於shuffled才進pilot；未通過不加steps／epochs。保存樣本IDs與coverage，不以subset分類accuracy選winner。
- Mechanism audit報target KL及unconditioned joint-region mass，並以固定前32有效audit samples每frame遮5patches，對highest／lowest／seed42random比較correct-class logit下降，replacement為normalized RGB零值。14×14下5patches遮擋的實際面積較舊7×7小，不能直接跨patch size比較幅度；masking只作方向診斷，不證明分類因果。

### 10.5 Bounded pilot、whole／zoom／ensemble與停止

先跑完整80,640train的2epoch checkpoint review，從原patch16 baseline開始，使用seed42、batch4、同AdamW／LR`2.25e-5`／weight decay`5e-4`／clip1；schedule仍從第一步鎖定十輪201,600updates，不壓縮成兩輪cosine。這是Native較保守的budget gate，不改寫正在執行的Spatial recipe。

- 每epoch使用全部12,960validation做FP32 whole／zoom70，ensemble為softmax probability等權平均；各自比較原baseline74.1204%／78.9738%／80.9568%。不比較Spatial新best當作新baseline，但另列同epoch／best歷史方法對照與成本，避免混淆增益來源。
- 保留第8節原early-kill：epoch1及epoch2三種views六個結果全部各自低baseline至少0.50pp就停。Native另有epoch2budget gate：截至epoch2沒有任何whole／ensemble同checkpoint通過promising，則停止此固定recipe，按原terminal規則記no-go／inconclusive，不自動跑到十輪。
- Epoch1／2有promising仍只列候選；是否追加epoch3–10由使用者另行決定。若延長，沿用原十輪scheduler及complete-epoch resume，保持source／precision／cache identity，不因延長改weights或reset optimizer。十輪仍為硬上限，沒有patience reset。
- Promising沿用same-view、same-checkpoint的全部guardrails：accuracy至少+0.20pp、macro-F1不下降、unpaired及A060–A120各不下降超過0.20pp、A042–A048 whole至少358或ensemble至少385。完整metrics與per-action fixes／regressions必須保存，不能只看target KL、attention maps或關注七類。
- 若pass，仍不能將增益全歸因於native supervision；下一步confirmation需同budget CE＋retention且相同eager／precision對照，再視收益做第二seed。No matched control／single seed不得選deployment winner或讀test。

### 10.6 後續工程介面、產物與驗收

後續擴充統一runner／dispatcher為`--method native`，保持原semantic／temporal／spatial入口與historical snapshot不變。獨立root為`artifacts/patch16_skeleton_v1/native_attention_01/`，依audit／precision／smoke／pilot區分immutable namespaces；現階段此method尚未實作，以下命令只定義未來介面，不可直接啟動。

```bash
.venv/bin/python -m vlm.training.patch16_skeleton --method native --mode audit
.venv/bin/python -m vlm.training.patch16_skeleton --method native --mode smoke
.venv/bin/python -m vlm.training.patch16_skeleton --method native --mode pilot
```

實作者須保存heads／layers／198-token indexing、Gaussian target version、eager backend／precision、anchor來源、native-enabled parity、finite gradient audit、每epoch三views及same-checkpoint guardrails、mechanism metrics、best／last exports與stop_reason。新增tests只保護dynamic196-patch slicing／CLS-message exclusion、target normalization／all-invalid、attention-to-Q／frozen-base isolation、opt-in logit identity、epoch2budget gate及RGB-only merge／reload。通過audit／precision／smoke後才可獨立tmux啟動，不自動接管Spatial process或其workflow hold。

### 10.7 2026-10-04：使用者要求prepare的核心準備

使用者在Spatial epoch5進行中明確要求prepare Method 4。此授權用於獨立核心元件、CPU contracts與preflight準備，未停止Spatial，也未啟動Native GPU jobs。`models/patch16_native.py`以獨立subclass提供opt-in eager native CLS attention，預設RGB forward和state keys沿用既有RGB model；共用`xclip_vision_lora.py`與`build_model`文字及Spatial當前source dependencies均保持不變。這避免為準備attention介面而使既有anchor cache identity失效；正式native-enabled／anchor parity仍須GPU audit確認。

`patch16_native_targets.py`準備196-patch Gaussian5×5 sigma1 distribution、FP32 KL、CLS/message exclusion、raw／conditional mass metrics、獨立derived cache及兩輪review決策。Derived cache每次重新核對現有Spatial source fingerprint／occupancy／mask與Gaussian distribution，並使用`cache/patch16_native_targets_v1/<target-version>` namespace。Tiny CPU model及cache contracts共5項測試通過，包含attention-only最後一層Q nonzero gradient、同層V允許zero、frozen base無gradient及RGB-only merge／strict reload；這些不是正式12-layer／12-head patch16模型的GPU precision evidence。

準備規格及實際結果存於`artifacts/patch16_skeleton_v1/native_preparation_01/`的`recipe.json`、`cpu_tests.json`、`active_spatial_sources_unchanged.json`與`preflight_report.json`（完成preflight後產生）。統一runner／dispatcher尚未加入`--method native`，因此10.6的未來命令仍不可直接執行；尚待Spatial停止後整合runner、full472 audit、三arms precision screen、correct／shuffled及masking diagnostics、8-step smoke，才可開始2epoch pilot。Pilot兩輪後固定停止review，即使有promising也不自動延長。

### 10.8 2026-10-04：使用者停止Spatial並啟動Native gates

使用者明確要求「kill for 4」，Spatial PID2849345已在epoch6 partial training停止；最後完成的epoch5 checkpoint（100800updates、含GradScaler／optimizer／scheduler／RNG）及whole epoch1／ensemble epoch3最佳RGB exports保留，恢復從epoch6開始。Spatial終止記錄位於`accelerated_spatial_01/spatial/pilot/seed42/manual_stop.json`。

統一runner／dispatcher現已接受`--method native`，Native必須明確使用`--precision amp_fp16 --anchor-cache <complete-cache>`；先跑audit → 三arms precision → smoke，再進入最多兩輪pilot。Source gates和precision recipe identity包含新增native model／targets／runtime helpers及完整cache checksum，不能重用Spatial gates。Native不載入SkeletonCLIP teacher或64-frame tensor cache，paired/raw-source驗證仍保留。Original `build_model`函式文字及共用RGB model／LoRA sources保持不變，完整anchor builder provenance比對通過。

17個CPU regression/contract tests全數通過，包括既有completed-epoch resume和新增native契約。正式Native audit另保存全部472 native-enabled／disabled及anchor logits parity、[8,2,198]原生attention／target／occupancy、raw alignment和前32有效樣本的highest／lowest／seed42random每frame遮5patches診斷。Precision三arms要求FP32 online／cached的weights／losses／validation logits／optimizer／scheduler exact agreement、AMP held-out argmax全同且finite。任何gate失敗停止，不自行改batch、heads、loss weight或precision。

背景workflow位於`artifacts/patch16_skeleton_v1/native_attention_01/`，tmux `patch16-method4-native`。兩輪pilot後依`two_epoch_review_promising`或`two_epoch_review_no_candidate`停止review，不自動延長。正式GPU gates與完整validation結果以各phase run reports為準，啟動audit不等於已通過precision或已開始epoch1。

```bash
.venv/bin/python -m vlm.training.patch16_skeleton_dispatch --artifacts-root artifacts/patch16_skeleton_v1/native_attention_01 --methods native --precision amp_fp16 --anchor-cache artifacts/patch16_skeleton_v1/shared_anchor_cache_v1
```

## 11. 下一步：CE＋RGB retention對照，確認skeleton的額外貢獻

2026-10-04依使用者要求加入。狀態為pending plan，只更新文件，不啟動對照訓練或重啟既有methods。已完成的single-seed結果均有小幅baseline增益，但尚無matched control，不能判定這些增益來自skeleton；在追加其他融合方法或延長budget之前，優先做一次短對照。

### 11.1 目前證據與主要比較對象

| 方法 | 保存的最佳whole | Whole gain | 保存的最佳ensemble | Ensemble gain | 完成狀態 |
| --- | ---: | ---: | ---: | ---: | --- |
| Semantic | Epoch7，74.5910% | +0.4707pp | Epoch6，81.4198% | +0.4630pp | 完成9epochs，epoch10中途手動停止 |
| Temporal | Epoch6，74.4136% | +0.2932pp | Epoch6，81.3349% | +0.3781pp | 完成6epochs，epoch7中途手動停止 |
| Spatial | Epoch1，74.3904% | +0.2701pp | Epoch3，81.3040% | +0.3472pp | 完成5epochs，epoch6中途手動停止 |
| Native attention | Epoch1，74.4830% | +0.3627pp | Epoch1，81.1343% | +0.1775pp | 完成2epochs；whole promising，ensemble未通過 |

這些是不同epoch的各view best，不是同一checkpoint，也不是四次獨立重複。原baseline為whole74.1204%、ensemble80.9568%。完整結果分別在`implementation_01/semantic/pilot/seed42/`、`accelerated_01/temporal/pilot/seed42/`、`accelerated_spatial_01/spatial/pilot/seed42/`及`native_attention_01/native/pilot/seed42/`，以上目錄皆以`artifacts/patch16_skeleton_v1/`為root，以`run_report.json`／`selection.json`及各epoch保存的logits為準，不以可能過時的progress state判定run是否仍在執行。

本次對照的primary comparison固定為已跑滿2epochs的Native attention：它保留原RGB架構，兩個arms可只差native loss，而且正好符合短預算。Semantic的7／9輪best、Temporal的6輪best、Spatial的3／5輪best不能直接拿來對照新的2輪CE run並聲稱公平勝負。

### 11.2 唯一改變：移除native auxiliary loss

```text
Native arm（已完成）：CE120 + L_retention + 0.1*L_native
Control arm（待執行）：CE120 + L_retention
CE label_smoothing = 0.05
L_retention = 4 * mean KL(softmax(original_patch16_logits/2) || softmax(student_logits/2))
```

- 兩個arms均從original epoch20 patch16 adapter／classifier開始，不從Native、Semantic、Temporal或Spatial winner續訓。Control不用skeleton targets決定更新，不做teacher forward、semantic prealignment、temporal gate或occupancy head；paired manifest僅用於evaluation subgroup標記。
- 為隔離loss，Control使用Native相同RGB classifier、eager backend與原生attention-output forward，忽略attention maps、不計算target KL，deployment仍為原RGB120。不能為了加速換成其他attention backend而仍稱為完全matched control；記錄output-attention額外成本。
- Trainable parameters與Native完全相同：upper-two vision q/v LoRA、MIT q/v LoRA及main120 classifier；base固定，student eval保留gradient。共用AMP FP16 student forward、FP32 CE／retention／validation、完整FP32 original anchor cache，嚴格驗證cache身份與checksum。
- 固定seed42、batch4、workers24上限、CPU threads8、全部80,640train、相同shuffle／sampling／processed pixels。AdamW LR`2.25e-5`、weight decay`5e-4`、betas`(0.9,0.999)`、eps`1e-8`、clip1；GradScaler init_scale256、growth_interval1000000；cosine從第1步鎖定201,600steps，不壓縮成2epochs schedule。
- 跑滿2epochs、40,320optimizer updates、161,280sample exposures後停止，不採validation選擇的提前停止或自動延長；技術失敗仍立即停止且另記failed。Primary epoch2比較需要相同exposure，不能Native跑2輪、Control只跑1輪。
- 在訓練前從Native experiment lock／source snapshot逐項核對backend、precision、optimizer／scheduler／scaler、sample order及baseline。保存training-boundary RNG／DataLoader generator狀態與epoch sample-order digest，能取得歷史Native順序時逐一比較。若缺歷史資訊或recipe不符，明確列出差異，只能稱recipe-matched exploratory comparison；不得宣稱bitwise replay或擅自改Native歷史鎖。

### 11.3 Smoke、view評估與公平selection

- 新增獨立control namespace，先做固定128train／120validation、最多8steps的smoke，確認原baseline identity、frozen anchor、CE／retention gradients、無native auxiliary contribution、finite及RGB-only merge／strict reload。沒有新target，不要求skeleton mechanism screen；模型／precision／anchor變更則不能借用舊passing gates。
- 每epoch評估完整12,960validation的FP32 whole、zoom70及equal-softmax ensemble，保存aligned logits／IDs／labels。Baseline、Native及Control全部使用同一reporting implementation重新計算accuracy、macro-F1、paired／unpaired、A001–A059／A060–A120及A042–A048 correct。
- Primary固定epoch2，同view計算`Native epoch2 - Control epoch2`，不拿Native epoch1 best對Control epoch2。Epoch1同預算比較作secondary diagnostic。
- 另報兩個arms各自epoch1–2的best／best-promising，使用同一selection與earliest-tie規則，但標為validation-selected secondary結果。不得比較只挑過best的Native與固定last的Control。
- Semantic／Spatial前2輪可列在附表作描述性比較，但先揭露semantic precision／online anchor與extra branch差異；Temporal改了部署架構，這個plain RGB control不能隔離skeleton target的效益。若日後確認Temporal，需相同temporal gate但關閉target KL的專屬control，不將本次一個control宣稱覆蓋四種方法。

### 11.4 判讀與compute停止決策

需要同時區分兩個問題：`Control - original baseline`回答CE＋retention續訓本身有多少收益；`Native - Control`回答在matched recipe下加入native supervision的額外收益。先前各method相對baseline的+0.2至+0.5pp不是第二個問題的答案。

- 若Native epoch2在whole及ensemble都不高於Control epoch2，則本次沒有native額外分類收益證據，即使兩者都高於原baseline；停止延長Native固定recipe，不因attention maps改善追加budget。
- 若任一view Native相對Control只有小於`+0.20pp`的正gain，或macro-F1／subgroup保護未通過，記`inconclusive`，不自動啟動新融合或延長budget。不得把小正差值稱為可重現收益。
- `incremental_promising`要求同一epoch2、同一view：Native比Control至少淨增26correct（≥0.20pp）、macro-F1不下降、unpaired及A060–A120各不下降超過0.20pp、A042–A048 correct不低於Control。同時Native仍須通過第8節相對original baseline的全部guardrails，不能以傷害Control子群換取aggregate小增益。
- 若Control收益已接近或超過目前融合方法，將「續訓／retention足以解釋主要增益」列為受支持的觀察，但不外推所有skeleton方法無效。保留所有checkpoints，不自動替換部署模型。
- 若Native通過incremental gate，下一步才規劃seed43的Native／Control paired confirmation；不能單獨換seed或拿test確認。這次Control到2epochs即結束，任何更長或其他architecture控制組另行決定。

### 11.5 後續介面與報告

後續runner新增`--method ce_retention`，使用獨立root `artifacts/patch16_skeleton_v1/ce_retention_control_01/`，包含control smoke、pilot及comparison report。這是待實作介面，文件續寫不表示現在可執行或允許自動起跑。新的control不可放進已有four-method dispatcher自動queue，須獨立tmux啟動並維持原run artifacts不變。

比較報告保存兩個arms的recipe／source hashes、epoch1／2／best metrics、同view net correct差異、逐sample Native-only-correct／Control-only-correct／both-correct／both-wrong，以及per-action fixes／regressions與完整guardrails判定；四格須覆蓋全部12,960unique IDs並與accuracy差值相符。記錄FP16 scaler skipped updates／completed exposures，不能只以scheduler步數宣稱實際參數更新完全相同。驗證僅新增保護auxiliary關閉、相同exposure／schedule、matched-epoch comparison及aligned prediction counting的重要tests，文件階段只執行Markdown／diff checks，不啟動訓練。

### 11.6 2026-10-04：使用者授權執行CE＋retention對照

使用者明確要求「new plan came in, do it」，已授權實作與執行第11節。`--method ce_retention`已接入共用runner，使用相同Native RGB model、eager attention-output forward、AMP FP16與FP32 CE／retention／validation、batch4、原epoch20 adapter及完整original FP32 anchor cache；auxiliary固定為無autograd依賴的zero，不載入teacher或skeleton target。Full pilot固定兩輪，不使用validation early-kill，不自動延長；原四方法dispatcher不加入此control。

獨立workflow `python -m vlm.training.patch16_control`依序執行current-source audit → 8-step smoke → 2epoch pilot → comparison，root為`artifacts/patch16_skeleton_v1/ce_retention_control_01/`，tmux `patch16-ce-retention-control`。每phase先核對Native歷史lock及source snapshot、baseline／model／pixel dependencies、AMP／eager／optimizer／scheduler／scaler設定。Original anchor builder source identity保持不變。Full-epoch exposure固定80640、20160updates；optimizer skipped update會立即fail closed並記錄，不能把scheduler step當成功參數更新。

Native沒有保存training-boundary RNG或實際order digest，因此此比較明確標為recipe-matched exploratory，不宣稱bitwise replay。依歷史loader source重建seed42 shuffle後，epoch1／epoch2 generator states皆與Native immutable checkpoints exact一致；historical scaler scale256、growth interval1000000、growth trackers20160／40320可核對。Control另保存training-boundary RNG／trainables／scaler、每epoch ordered IDs及digest，逐一比對重建順序並核對Native completed generator state。

20個CPU regression／contract tests通過；另以完整12960筆Native保存predictions作identical-arm fixture，驗證固定epoch2 primary、兩輪三views共六組aligned四格計數、net correct與完整報告序列化。比較程式重算baseline／Native／Control相同reporting metrics，以Native epoch2−Control epoch2作primary，epoch1及兩輪best／best-promising作secondary，保存per-sample CSV、per-action fixes/regressions及incremental／baseline雙guardrails；兩輪後不自動起跑seed43或其他methods。實際GPU smoke／pilot與comparison status以此root下phase reports為準。

```bash
.venv/bin/python -m vlm.training.patch16_control --artifacts-root artifacts/patch16_skeleton_v1/ce_retention_control_01 --anchor-cache artifacts/patch16_skeleton_v1/shared_anchor_cache_v1
```

### 11.7 2026-10-04：對照完成，未觀察到Native額外分類收益

CE＋retention對照完成audit、smoke與兩輪pilot，於2026-10-04 13:38（Asia/Taipei）到達`two_epoch_control_cap`並自動停止，共40,320次optimizer updates，沒有skipped updates。比較報告位於`artifacts/patch16_skeleton_v1/ce_retention_control_01/comparison/run_report.json`，status為`no_native_incremental_gain`；所有checkpoints與RGB exports保留，沒有評估test或自動延長訓練。

| Primary epoch2 | Native | Control | Native−Control |
| --- | --- | --- | --- |
| Whole accuracy | 74.344134% | 74.359566% | −0.015432pp（−2 correct） |
| Ensemble accuracy | 81.080246% | 81.095678% | −0.015432pp（−2 correct） |

Control epoch2 whole相對original baseline增加0.239193pp，ensemble增加0.138885pp。Epoch1 secondary whole的Native−Control曾為+0.200617pp（+26 correct），但ensemble僅+0.015432pp（+2 correct），且whole優勢在epoch2沒有維持。本次單seed、recipe-matched exploratory結果不支持延長Native固定recipe，亦不代表所有skeleton方法都無效。

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

### 2026-10-03: stop semantic epoch 10 and queue accelerated temporal

User stopped Method 1 during epoch 10 after epoch 9 failed to improve either selected view. The completed epoch 9 `last_checkpoint.pt` and selected whole epoch 7 / ensemble epoch 6 checkpoints remain available; this interruption did not produce a final `last_rgb.pt`. `manual_stop.json` records the intentional interruption. The original pending-method hold stays in place.

The shared FP32 anchor builder automatically removed its yielding delay after semantic PID 265549 ended. Recent throughput rose from about 4.6 to 11.4 samples/s. `accelerated_01` runs in tmux `patch16-method2-accelerated`, waits for complete cache publication, then executes temporal audit → smoke → pilot using AMP FP16 student forward, FP32 losses/anchor/validation, batch 4 and the original scheduler/exposure. Gates require matching sources, precision, cache path and checksum. The cache loader verifies canonical IDs, labels, original adapter, input fingerprints, model builder and complete payload checksum; partial caches are rejected.

Training checkpoints now retain GradScaler state and restore it with optimizer, scheduler, sampling generator and RNG. A short CUDA continuation check confirmed exact subsequent parameters, AdamW state, scheduler and scaler state; a shuffled-loader check confirmed cache row alignment and the original anchor builder fingerprint remained unchanged. The earlier bounded precision screens remain evidence for short-run agreement; the new full temporal validation is pending.

### 2026-10-03: stop temporal after epoch 6 and start spatial

Epoch 6 improved both temporal views: whole 74.41358% (+0.29321 percentage points), ensemble 81.33488% (+0.37808 percentage points). The conditional epoch-6 watcher therefore kept training. The user then explicitly stopped Method 2 during epoch 7 and requested Method 3. All epoch-6 last/best checkpoints and RGB exports were retained, and optimizer/scheduler/GradScaler continuation state was verified (120960 completed updates). Resume starts at epoch 7 and repeats its interrupted partial exposure. `accelerated_spatial_01` now runs spatial audit → smoke → pilot in tmux `patch16-method3-accelerated`, starting from the original RGB baseline with AMP FP16 student forward, the completed shared FP32 anchor cache and batch 4. Full spatial validation is pending.
