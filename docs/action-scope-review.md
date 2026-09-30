# Action scope review

日期：2026-09-29。狀態：`reduced-v1` scope與第一次 direct retrain已完成；本文件記錄產品 scope、實驗結果與 no-go決策。這不把`irrelevant`等同於OOD `unknown`。

## Decision vocabulary

- `candidate`：使用者目前認為 accuracy太低或產品不重要，但尚未決定如何處理。
- `keep`：仍需獨立辨識的 action。
- `irrelevant`：已知且產品不需細分的 action；可能作為一個產品輸出，但需先驗證合併後的class heterogeneity。
- `merge`：和另一個語意接近的 action合併；需明確的新label與mapping。
- `remove`：不進入新任務；部署仍需另外處理模型遇到此動作時的行為。
- `unknown`：輸入超出已知資料分布或證據不足，不應直接由被移除的 NTU labels定義。

## Current user-selected candidates

以下 accuracy與confusions來自 Phase 3 vision LoRA seed-42 best epoch 13的 canonical validation logits，overall accuracy為58.9198%。「Reason」暫時保持未指定，避免把低accuracy誤寫成產品不重要。

| Action | Name | Accuracy | Most common wrong predictions | Status | Reason |
| --- | --- | ---: | --- | --- | --- |
| A073 | staple book | 17/144（11.81%） | A076 cutting paper 41；A075 cutting nails 32；A083 ball up paper 10 | candidate | pending |
| A084 | play magic cube | 10/144（6.94%） | A075 cutting nails 46；A076 cutting paper 26；A078 open bottle 18 | candidate | pending |
| A072 | make victory sign | 5/144（3.47%） | A071 make OK sign 63；A067 hush 15；A069 thumb up 8 | candidate | pending |
| A071 | make OK sign | 73/144（50.69%） | A067 hush 11；A069 thumb up 10；A096 cross arms 8 | candidate | pending |
| A082 | fold paper | 20/144（13.89%） | A076 cutting paper 33；A075 cutting nails 24；A083 ball up paper 18 | candidate | pending |
| A083 | ball up paper | 52/144（36.11%） | A075 cutting nails 19；A076 cutting paper 17；A078 open bottle 11 | candidate | pending |
| A070 | thumb down | 60/144（41.67%） | A069 thumb up 18；A071 make OK sign 11；A068 flick hair 6 | candidate | pending |
| A094 | throw up cap/hat | 97/144（67.36%） | A095 hands up 7；A103 yawn 5；A071 make OK sign 5 | candidate | pending |
| A065 | tennis bat swing | 60/144（41.67%） | A066 juggling table tennis balls 14；A096 cross arms 9；A070 thumb down 8 | candidate | pending |

## Initial analysis

這九類分成兩個明顯的視覺群組。`A071`、`A072`與`A070`屬於細小手勢，主要差異集中在手指形狀；224×224輸入、32×32 patches與8-frame sampling可能先天地弱。`A073`、`A082`、`A083`、`A084`則是桌面／手部小物件操作，彼此以及`cutting paper`、`cutting nails`高度混淆，可能需要object／hand detail而非更多全身joint supervision。`A065`是較大幅動作但仍和其他手臂動作混淆。`A094`已有67.36%，若仍列為candidate，比較可能是產品scope選擇，而不是純粹model無法辨識。

目前不建議把九類直接合成單一`irrelevant`並立即訓練，因為它們橫跨hand gesture、object manipulation與large arm motion，合併class內部差異很大。下一步先由使用者為每類補上`not important`、`too ambiguous`、`merge with ...`或`keep but improve`；完成後再設計mapping與balanced evaluation。

## Assistant audit：top 20 candidates not to keep as separate labels

這份推薦的意思是「優先考慮不再保留為獨立named class」，不是直接丟棄影片，也不是把它們定義成`unknown`。排序綜合目前accuracy、主要confusion是否落在同一視覺群、224×224／8-frame XCLIP是否具備合理辨識條件，以及一般安全／健康語意價值。因為尚未提供最終產品用途，產品價值欄是保守假設，使用者可覆寫。

使用者review override（2026-09-29）：`A086 apply cream on hand back`、`A075 cutting nails`與`A078 open bottle`目前不確定或不希望排除，狀態改為`review／暫時保留`。它們仍留在原始audit table以保存推薦歷史，但不得計入已接受的scope reduction，也不得自動映射到`irrelevant`。

| Priority | Action | Name | Accuracy | Why consider not keeping separately | Suggested disposition | Confidence |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | A074 | counting money | 2.78% | 極低accuracy；主要誤判為cutting nails、cutting paper與stapling，動作差異高度依賴小物件 | 若不屬產品核心，放入fine-object irrelevant pool | high |
| 2 | A072 | make victory sign | 3.47% | 63／144直接誤判為OK sign；32×32 patch與全身取景難解析手指形狀 | 與其他hand signs合成coarse hand-sign label，或irrelevant | high |
| 3 | A084 | play magic cube | 6.94% | 幾乎被各種桌面手部操作吸收；需要物件外觀與手部crop | fine-object irrelevant pool | high |
| 4 | A073 | staple book | 11.81% | 與cutting paper／nails高度混淆，主要辨識訊號是很小的stapler | fine-object irrelevant pool | high |
| 5 | A082 | fold paper | 13.89% | 與cutting／balling paper高度重疊，八幀可能漏掉狀態轉換 | 合成paper manipulation或irrelevant | high |
| 6 | A086 | apply cream on hand back | 15.97% | Cream通常不可見，僅靠手部摩擦很難和其他hand-object actions分開 | review／暫時保留；可評估hand-care grouping | paused by user review |
| 7 | A078 | open bottle | 26.39% | 物件與旋轉細節太小，常被paper／nail actions與cross arms吸收 | review／暫時保留；若要改善可研究hand/object crop | excluded by user |
| 8 | A069 | thumb up | 27.08% | 主要誤判為OK sign與thumb down，差異集中在少數手指pixels | coarse hand-sign label或irrelevant | high |
| 9 | A081 | toss a coin | 27.08% | Coin本身幾乎不可見，短暫拋接可能被8-frame sampling漏掉 | fine-object／short-event irrelevant pool | high |
| 10 | A077 | snapping fingers | 32.64% | 關鍵是細小手指動作與可能的音訊，RGB全身影像證據不足 | hand-sign／short-event irrelevant pool | high |
| 11 | A093 | shake fist | 35.42% | 和OK sign、hush、snapping高度混淆；若不關心威嚇語意，獨立label價值有限 | coarse hand-gesture label；若關心aggression則keep | medium |
| 12 | A083 | ball up paper | 36.11% | 和cutting／nails／bottle操作混淆，紙張狀態與手部細節才是主要線索 | 合成paper manipulation或irrelevant | high |
| 13 | A065 | tennis bat swing | 41.67% | 球拍可能太小或出框，並和一般手臂動作混淆；對通用activity scope較特殊 | sports-action irrelevant pool；sports產品則keep | medium |
| 14 | A070 | thumb down | 41.67% | 和thumb up／OK sign高度重疊，仍依賴手指方向與局部解析度 | coarse hand-sign label或irrelevant | high |
| 15 | A091 | open a box | 41.67% | 和cutting paper、open bottle、fold paper混淆，主要差異來自小物件identity | fine-object irrelevant pool | medium |
| 16 | A076 | cutting paper with scissors | 44.44% | Scissors與paper細節小，且和cutting nails／stapling互相混淆 | 合成tool-based paper manipulation或irrelevant | high |
| 17 | A075 | cutting nails | 47.92% | Nail clipper與指尖不可可靠解析，和多個桌面操作互相混淆 | review／暫時保留；可評估hand-care grouping | paused by user review |
| 18 | A071 | make OK sign | 50.69% | Accuracy雖不算最低，但和hush、thumb up、victory sign共用相似全身姿勢 | coarse hand-sign label或irrelevant | medium-high |
| 19 | A066 | juggling table tennis balls | 51.39% | 小球難見，與balling paper等手部動作混淆；用途偏特殊 | sports／fine-object irrelevant pool | medium |
| 20 | A094 | throw up cap/hat | 67.36% | 模型其實可辨識，不是技術性弱類；列入只因使用者已表示可能不重要且用途較特殊 | 若確認不重要則irrelevant，否則keep | low（技術理由）／high（使用者scope） |

### Explicitly excluded despite low accuracy

下列低accuracy actions沒有列入推薦top 20，因為在未知產品需求下可能具有較高安全、健康或通用語意價值：`A041 sneeze/cough`、`A047 neck pain`、`A110 shooting at another person`、`A106 hitting another person with an object`與`A107 wielding a knife`。`A011 reading`、`A012 writing`、`A034 rubbing hands`與`A049 using a fan`也暫時保留，因為它們比細粒度物件identity更可能是通用活動需求；若產品明確不需要，可在下一輪重排。

### Recommended grouping if accepted

- `coarse_hand_sign_or_irrelevant`：A069、A070、A071、A072、A077、A093；A093若需保留aggression semantics則不得併入。
- `fine_object_or_paper_irrelevant`：A073、A074、A076、A081、A082、A083、A084、A091；A075、A078與A086暫不加入，維持review／保留。
- `specialized_sports_or_event_irrelevant`：A065、A066、A094。

這三群應在內部維持不同negative pools與metrics，不建議直接訓練成單一、語意極度分散的`irrelevant` softmax class。產品輸出可以統一顯示`irrelevant`，但訓練時可使用hierarchical relevance head、grouped auxiliary labels或outlier exposure；實際設計需等使用者確認class scope後鎖定。

## Current product-scope decision (2026-09-30)

使用者明確決定不再保留以下七類作獨立產品輸出：A086 apply cream on hand、A034 rub two hands、A011 reading、A078 open bottle、A012 writing、A110 shoot with gun、A004 brush hair。這次決定覆寫A078與A086先前的「review／暫時保留」產品狀態，但不改寫歷史reduced-v1 taxonomy或實驗結果；A075仍未被排除，亦不自行排除A041、A049或其他未點名類別。

加上先前接受的17類，產品scope目前累計24類不要求獨立named output。這不等於其餘96類都是醫療緊急事件，也不等於啟動新的96／100／106-output訓練。當前模型仍保留raw120 outputs；本次只記錄產品需求，尚未接入runtime output filtering或新的taxonomy CLI。

「Discard」指移出獨立產品辨識要求，不刪除影片、source labels、logits或訓練／評估樣本。這些動作仍是attention-event的已知negative examples；計算事件分數時保留全部120類的softmax probability mass，不移除七類logits後重新normalize。特別是A004 brush hair曾被錯判為A044 head-touch/headache pose，應保留以測試誤報。

A110被排除是因為本輪以醫療／照護注意事件為scope，不是將持槍動作判定為安全。[NTU作者分類](https://rose1.ntu.edu.sg/dataset/actionRecognition/)將它列為two-person interaction；資料集的medical conditions分類也不等同臨床緊急程度。

方向建議為event-centric attention detection：沿用已選的A042–A048候選事件，分開評估falling、nausea/vomiting、失衡與各種不適姿態，不要求它們共用一個高優先警報門檻。主要指標改為event recall／漏報、precision、連續影片的false alarms per hour與偵測延遲；120-class accuracy保留作representation診斷，不再單獨決定照護產品是否成功。姿態辨識不是疼痛或疾病診斷。

近期優先候選是完整validation確認固定whole＋zoom inference及各事件誤報，而非立即再跑spatial-supervision／KD pilot或新增reduced classifier。480-sample改善尚不能外推為full validation accuracy；後續仍需要連續正常活動與事件影片才能驗證警報。本次未啟動新訓練、完整validation或真實警報。

## Implemented reduced-v1 contract (historical; unchanged)

使用者接受的17個source actions被映射為三個internal outputs：`irrelevant_hand_sign`（A069、A070、A071、A072、A077、A093）、`irrelevant_fine_object`（A073、A074、A076、A081、A082、A083、A084、A091）與`irrelevant_specialized`（A065、A066、A094）。A075、A078與A086維持named outputs。模型因此有103個named outputs加3個internal irrelevant outputs，共106 outputs；三個internal outputs在public API均映射為`irrelevant`。OOD `unknown`仍是不同概念，本次未實作。

`vlm experiment action-taxonomy`可對既有120-class logits作probability aggregation；`vlm train xclip-lora --action-taxonomy reduced-v1`可直接訓練106-output MIT LoRA與classifier。Grouped outputs的CE weight分別依來源class數量使用`1/6`、`1/8`與`1/3`，避免17個被合併來源類別因樣本數而壓過每個named class。

## Validation-only results and decision

既有Phase 3 upper-vision checkpoint的120-class validation probabilities經離線aggregation後，106-output exact accuracy為`65.8565%`、public accuracy為`67.8395%`、relevant named accuracy為`63.9555%`、irrelevant public recall為`84.5180%`、relevant false-irrelevant rate為`7.6579%`。這只是同一組logits的後處理，不代表reduced taxonomy retrain或新的視覺能力；產物位於`artifacts/action_taxonomy_audit/`。

第一次direct retrain從pretrained XCLIP開始，只更新cached pre-MIT features之後的MIT q/v LoRA與新106-output classifier，未使用upper-vision LoRA。完整canonical train／validation為80,640／12,960筆，固定60 epochs、seed 42，test未載入或評估。最佳epoch為28，exact validation accuracy為`54.4444%`；最後結果從best checkpoint重算，macro F1為`55.9554%`、NLL為`1.5326`、ECE為`4.0523%`，merge parity最大絕對誤差為`2.3842e-6`。完整log與結果位於`artifacts/action_taxonomy_reduced_v1_seed42/train.log`及同目錄`evaluation_results.json`。

Public metrics為accuracy `55.9645%`、relevant named accuracy `57.4581%`、irrelevant recall `49.5507%`與relevant false-irrelevant rate `3.2344%`。三個internal group的recall分別為hand-sign `44.6759%`、fine-object `50.5208%`、specialized `56.7130%`。暫時保留的A075、A078與A086 exact accuracy分別為`52.7778%`、`25.0000%`與`34.0278%`，其中被錯判為public irrelevant的比例分別為`18.7500%`、`28.4722%`與`22.2222%`。

**Decision：direct MIT-only reduced-v1 retrain為no-go，不作推薦部署checkpoint，也不因本結果直接啟動昂貴的upper-vision reduced retrain。** 它相對原MIT-only 120-class control約只增加2.3 percentage points，且只找回約一半irrelevant samples。下一個低成本候選是直接使用既有upper-vision checkpoint的probability aggregation，並以retained-action與false-irrelevant metrics決定是否足夠；只有其錯誤分析顯示taxonomy-aware optimization有明確修正空間時，才另行鎖定upper-vision retrain。

## Upper-vision reduced-v1 run

使用者於2026-09-29決定繼續upper-vision實驗。`skeleton-phase3-gate1`現支援`--action-taxonomy reduced-v1`：完整RGB forward會更新vision encoder最後兩個blocks與MIT的q/v LoRA以及106-output classifier，沿用group-balanced CE、uint8 processor cache、RGB-only merged checkpoint與test embargo。兩個training batches／兩個validation batches的真實GPU smoke已完成，trainable parameter audit包含預期vision／MIT LoRA與classifier，merge及strict reload最大誤差皆為`1.1176e-6`；產物位於`artifacts/action_taxonomy_upper_vision_smoke/`。

完整seed-42 run使用canonical train／validation 80,640／12,960筆、60 epochs、batch 16、24 workers與`cache/skeletonclip_phase3_rgb_uint8`，輸出位於`artifacts/action_taxonomy_upper_vision_seed42/vision_lora_ce_seed42/`。本次run從pretrained XCLIP與新106-output head開始，不從MIT-only reduced checkpoint續訓；test未載入或評估。Run於2026-09-30完成，accuracy-selected best為epoch 18：exact accuracy `61.9830%`。Epoch 25得到最高macro-F1 `64.4795%`，但不取代事前鎖定的accuracy selection；epoch 60為accuracy `61.4506%`、macro-F1 `64.0145%`、NLL `1.4280`，顯示後半段plateau與calibration overfit。

Best epoch 18的public accuracy為`63.4259%`、relevant named accuracy為`65.6298%`、irrelevant recall為`53.9624%`、relevant false-irrelevant rate為`2.8063%`。A075、A078與A086 exact accuracy分別為`49.3056%`、`25.0000%`與`28.4722%`，被錯判為public irrelevant的比例分別為`17.3611%`、`29.8611%`與`22.9167%`。12,960筆validation predictions的sample identity唯一，106-output／120-source taxonomy metadata與public mapping均通過audit；merged與strict RGB-only reload最大誤差皆為`9.8944e-6`，低於`1e-5`門檻。

**Decision：upper-vision direct reduced-v1 retrain證明vision adaptation有效，但不取代120-class probability aggregation。** 它比MIT-only direct retrain的`54.4444%` exact高`7.5386` points，但仍比既有120-class upper-vision aggregation的`65.8565%` exact低`3.8735` points，public accuracy低`4.4136` points，irrelevant recall更低`30.5556` points。短期部署候選維持既有120-class upper-vision checkpoint加probability aggregation；本106-output checkpoint保留為direct-training control。後續若研究skeleton-guided vision，應保留120-class CE、加入decision-coupled spatial attention，再於inference套用相同aggregation，以分離representation gain與taxonomy post-processing gain。
