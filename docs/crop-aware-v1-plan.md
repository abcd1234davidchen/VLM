# Crop-aware v1：短 RGB120 pilot

2026-09-30使用者核准嘗試crop-aware fine-tuning。本輪只研究whole／zoom輸入適應，不恢復spatial-supervision或KD，不改threshold／產品output filtering，不另訓練106-output模型，不讀test，不自動部署。

## 已有結果與退步檢查

原始raw120 checkpoint在完整12,960筆validation上，whole為58.9198%（7,636筆正確）、zoom為63.9120%（8,283）、whole＋zoom probabilities等權平均為66.9367%（8,675）。Ensemble修正1,456筆、退步417筆，淨增加1,039筆；baseline argmax parity零差異，logit最大誤差8.82e-5。這是完整validation結果，不是先前480筆subset的72.29%。

Ensemble逐類net退步為A069 −14、A067 −5、A084 −4、A072 −2、A053 −1、A074 −1。A043跌倒修正1筆／退步1筆，A047頸部不適修正4筆／退步4筆，overall或group net收益不能排除個別重要事件退步。中央zoom不追蹤人物，也可能丟掉context。

Smoke保存原始validation regression代表樣本的actual processor input strips：`S011C001P038R001A043`、`S006C003P022R002A048`與`S007C001P027R001A047`，whole／zoom各8幀。這些樣本只作裁切風險檢視，不用來選train samples或搜尋crop比例；代表性檢視不等於全部ROI安全保證。

已查看三張input strips：A043樣本倒地後部分body被zoom下緣裁掉，是明確context／body-loss風險；A048與A047代表樣本保留主要人物且放大細節，仍有ensemble退步，因此不能把所有錯誤都歸因於clipping。此pilot不改裁切策略，保留whole比較；即使短fine-tuning改善，也不能修復被裁掉的資訊或宣稱zoom-only安全。後續可研究person-aware ROI，但尚未授權或實作新的detector。

## 固定實驗設定

- Warm-start：Phase3 original best upper-vision／MIT raw120 adapter，沿用原始classifier；不從整合後的merged模型重新初始化LoRA。
- Train：canonical train 80,640筆中，按seed42 sample-ID hash、equal-class round robin固定選4,096筆；只跑一次完整subset，約1,024 steps，batch4、workers4，FP32。
- Mixed views：每sample的seed42 hash固定選whole或zoom，約50／50；單次train forward只看其中一個view，label不變。這輪只跑一pass，不宣稱跨epoch的random-view augmentation。
- Zoom：與已完成診斷完全相同，原圖short side的70%中央正方形、相同whole-video uniform8 frames、同一XCLIPProcessor；不是在224×224 postprocessed cache上再次裁切。
- Parameters：只更新最後兩個vision blocks與MIT的q/v LoRA和classifier；base frozen。沿用先前短pilot的eval-mode gradient策略，停用dropout／feature noise，避免同時加入新的stochastic regularization。
- Loss：raw120 CE、label smoothing0.1；AdamW lr1e-5、weight decay5e-4、cosine decay到單pass末、gradient clip1.0。沒有teacher、KD或新attention head。
- Validation：固定最後checkpoint，全部12,960筆比較whole、zoom、equal probability mean；分別對照原checkpoint同view結果。未搜尋epoch、crop比例或ensemble weights。
- Scope：全部120類的samples／logits保留；24個discard labels只作分組統計，不改模型taxonomy或normalize。Threshold report只作離線診斷，不接runtime alarm。

## Smoke 與判讀

Smoke從全部120類各取1筆train／validation作執行檢查，只做2個optimizer steps；確認兩個warm-start views與保存logits parity、LoRA／classifier trainable範圍、CE gradients到vision／MIT／classifier、finite loss／logits，以及merged RGB-only strict reload parity。其accuracy不是pilot evidence。Pilot必須有同一runner與view helper hash的passing smoke report才允許啟動。

預先判讀規則：ensemble相對原始66.9367%至少增加0.20 percentage points，且A042–A048合計top1 correct不退步，才標為promising；overall下降為no-go，其餘inconclusive。這只是pilot triage，仍須逐事件看漏報／誤報，不能靠group gate核准medical alarm。另一個探索目標是單次view能否接近原始ensemble，但不為此post-hoc更換主要判讀規則。

沒有另跑control group，所以即使改善，也不能將全部收益單獨歸因於crop augmentation而排除額外CE training效果。One-pass 4,096 subset無收益也不能否定充分訓練的crop-aware方法；先保持小成本，再決定是否值得擴大。

## Dispatch 與產物

Runner為`src/vlm/training/crop_aware.py`。Smoke artifact為`artifacts/crop_aware_v1_smoke/`、tmux session `crop-aware-smoke`；pilot已dispatch，artifact為`artifacts/crop_aware_v1_pilot/`、session `crop-aware-pilot`，job PID `2283918`、logger PID `2283921`。啟動時確認state為`running`，whole／zoom warm-start parity均為0；locked train IDs4,096筆與validation IDs12,960筆唯一且互斥，train view assignment為whole1,986筆／zoom2,110筆，runner／helper snapshot hashes符合lock。`experiment_lock.json`保存sample IDs、view assignments、source hashes與固定設定；runner／view helper snapshots保存執行來源，`trainable_parameters.json`保存更新範圍，`train.log`／`console.log`保存progress與process output。Checkpoint、raw predictions、每view confusion／alert reports、merged RGB-only state與`run_report.json`各自保存，不覆蓋原baseline。

Smoke已完成且status為`pass`：2個optimizer steps、8筆實際train（whole4／zoom4）、120筆validation；whole／zoom warm-start logit差異均為0，vision／MIT／classifier皆有finite positive CE gradients，merged strict reload max error為`9.78e-6`。這是重要執行契約檢查，不是accuracy收益證據。

## Completed pilot results and decision

Pilot與完整12,960筆validation已完成，dispatcher state為`complete`；最後checkpoint不經epoch selection，merged RGB-only strict reload max error為`4.83e-6`。實際train4,096筆（whole1,986／zoom2,110），未讀test。最終結果來源為`artifacts/crop_aware_v1_pilot/run_report.json`，原始logits／IDs保存於同目錄`validation_predictions.pt`。

| View | Original checkpoint | Crop-aware checkpoint | Net correct gain | Fixes / regressions | Attention correct / 504, original → new |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole | 58.9198% | 58.9583% | +5 | 189 / 184 | 303 → 303 |
| Center zoom70 | 63.9120% | 64.3133% | +52 | 201 / 149 | 333 → 334 |
| Whole＋zoom mean | 66.9367% | 67.1373% | +26 | 198 / 172 | 349 → 343 |

**Decision：inconclusive，不推薦替換原checkpoint，也不自行放大training budget。** Ensemble雖增加0.2006 percentage points、淨多26筆正確，但A042–A048合計退步6筆，未通過事前鎖定的non-regression gate。單次whole／zoom亦未達原始ensemble的66.9367%；不能宣稱已把雙視角收益學進單次推論。這個短subset結果不是crop-aware充分訓練的否定證明，但不提供有說服力的進一步模型突破。

## Reporting boundary: model gain versus inference or task changes

原模型單次whole的完整raw120 validation仍為58.9198%。同一checkpoint的whole＋zoom提高到66.9367%，是在相同120類、相同validation、不刪難類的實際辨識收益，但需要兩次forward；它是input-view ensemble收益，不是模型參數學習改善。此輪crop-aware訓練後ensemble為67.1373%，沒有任何本輪完整raw120 validation結果超過70%。

先前72.29%來自固定480筆、每類等量的subset，不得作全量成績或與12,960筆不同class權重的accuracy直接混用。合併／忽略labels提高的taxonomy metrics屬不同輸出任務，不能當成原120類辨識能力突破；本whole／zoom及crop-aware runs均保留全部120類。若目標是「單次推論模型全量raw120超過70%」，目前尚未達成。Threshold與discard-output仍是離線分析／產品scope筆記，不是已部署能力。
