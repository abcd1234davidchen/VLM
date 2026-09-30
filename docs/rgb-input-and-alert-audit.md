# RGB input and attention-event audit

## Scope and provenance

2026-09-30，使用者要求RGB輸入診斷、全部120類的20個最容易混淆動作（包含先前忽略的類別），並評估threshold-based警報。使用者選擇先涵蓋跌倒、噁心／嘔吐，再加失衡與疼痛姿態。本輪只做saved-prediction分析與frozen-model inference，不訓練、不修改checkpoint、不啟用live alarm、不讀test。

Checkpoint為Phase3 original best upper-vision/MIT LoRA RGB120，epoch13，validation 12,960 samples、accuracy `58.9198%`。資料、checkpoint、labels、runner與names hashes保存於`artifacts/rgb_input_audit_v1/experiment_lock.json`。Runner為`src/vlm/training/rgb_input_audit.py`，tmux session為`rgb-input-audit`；`train.log`是dispatcher的通用progress log，state明確為`inference_only`，不是新的training pilot。

## Twenty most confusing source actions

「最容易混淆」定義為per-class top1 error rate，由高至低排序，同率依action ID；不是依error count排序，避免A001–A060每類72筆、A061–A120每類144筆的sample-count差異。全部120類皆保留。`†`表示先前reduced-v1中被映射為irrelevant的source action，並未從本ranking過濾。Action semantics沿用[NTU authors class list](https://rose1.ntu.edu.sg/dataset/actionRecognition/)。列出的confusion只取每類最常見的一個錯誤方向，次要方向與全部120類結果在`confusion_alert_report.json`。

| Rank | ID | 動作 | 錯判率 | 最常錯認為（筆數） |
| ---: | --- | --- | ---: | --- |
| 1 | A074† | 數鈔票 | 97.22% | A075 剪指甲（47） |
| 2 | A072† | 比勝利／V手勢 | 96.53% | A071 OK手勢（63） |
| 3 | A084† | 玩魔術方塊 | 93.06% | A075 剪指甲（46） |
| 4 | A073† | 用釘書機釘書 | 88.19% | A076 用剪刀剪紙（41） |
| 5 | A082† | 摺紙 | 86.11% | A076 用剪刀剪紙（33） |
| 6 | A086 | 手背塗乳霜 | 84.03% | A078 開瓶子（22） |
| 7 | A034 | 搓雙手 | 79.17% | A039 雙掌合十（19） |
| 8 | A011 | 閱讀 | 77.78% | A029 玩手機／平板（21） |
| 9 | A078 | 開瓶子 | 73.61% | A083 揉紙成團（17） |
| 10 | A069† | 拇指朝上 | 72.92% | A071 OK手勢（41） |
| 11 | A081† | 拋硬幣 | 72.92% | A077 彈指（19） |
| 12 | A049 | 搧風／覺得熱的姿態 | 69.44% | A037 擦臉（11） |
| 13 | A077† | 彈指 | 67.36% | A071 OK手勢（16） |
| 14 | A047 | 觸碰頸部／頸部不適姿態 | 66.67% | A037 擦臉（10） |
| 15 | A012 | 寫字 | 65.28% | A029 玩手機／平板（28） |
| 16 | A041 | 打噴嚏／咳嗽 | 65.28% | A025 伸手進口袋（6） |
| 17 | A110 | 持槍朝向另一人射擊的動作 | 65.28% | A115 替另一人拍照（33） |
| 18 | A093† | 揮拳示意 | 64.58% | A071 OK手勢（17） |
| 19 | A004 | 梳頭髮 | 63.89% | A044 觸碰頭部／頭痛姿態（13） |
| 20 | A083† | 揉紙成團 | 63.89% | A075 剪指甲（19） |

## Fixed no-training input diagnostic

Validation固定每類4筆、480 samples，依seed42 identity hash／equal-class round robin抽取，獨立於prediction correctness。每個sample均比較whole-video uniform8、early半段8幀、middle半段8幀、late半段8幀，以及whole8的單一中央zoom（原影像short side的70%正方形）。Zoom不是person detector，不使用skeleton／label決定ROI，可能切掉重要context；它只能測試固定輸入變化，不是已核准deployment transform。

另外報固定的三個temporal-window mean probabilities與whole/zoom mean probabilities，不搜尋weights，不以單個sample的最佳view作oracle accuracy。Whole-view logits／argmax必須核對saved baseline。原model維持8-frame架構、不改MIT，不作16／32-frame訓練。Actual post-processor input strips保存於artifact目錄，供查看手、物件、body coverage；不同view的distribution shift與small subset限制都須保留，不能以zoom沒有收益直接否定crop-trained模型。

### Completed diagnostic results

Tmux diagnostic已完成，dispatcher terminal為`complete`，沒有optimizer steps。Whole-view與saved baseline的argmax完全一致，logit max error為`4.71e-5`；480 IDs唯一，labels與baseline一致、所有logits finite，runner snapshot hash符合lock。Actual input strips已查看A043 falling、A048 nausea/vomiting、A072 victory sign、A074 counting money與A089 bag insertion；代表樣本中放大後人物與手部佔用更多pixels，仍保留主要body coverage。這是representative visual audit，不是全部480／12,960的ROI安全保證。

| Fixed view | Correct / 480 | Accuracy | Fixes / regressions vs whole |
| --- | ---: | ---: | ---: |
| Whole uniform8 | 292 | 60.83% | 0 / 0 |
| Early half8 | 160 | 33.33% | 16 / 148 |
| Middle half8 | 244 | 50.83% | 22 / 70 |
| Late half8 | 216 | 45.00% | 13 / 89 |
| Center zoom70 whole8 | 330 | 68.75% | 79 / 41 |
| Three temporal windows mean | 259 | 53.96% | 17 / 50 |
| Whole + zoom mean | 347 | 72.29% | 64 / 9 |

Whole＋zoom mean相對whole gain為`+11.46` percentage points，在48個classes net改善、7個classes net退步。A001–A060為157→190 correct／240，A061–A120為135→157／240；previously ignored類別為18→27／68。Attention A042–A048僅28個subset samples，17→21 correct，不能作event safety確認。Zoom-only的attention correct為22／28，但不能據此post-hoc選alert transform。

這比前幾次小KD pilot提供更明顯的input-view訊號，但480 samples為每類等量，與full validation新60類每類樣本數加倍的權重不同；`72.29%`不能當成完整120-class validation accuracy。Half-clip views也改變原訓練的whole-video temporal分布，表現下降不能證明temporal modeling不重要。

下一個候選是保留固定70% zoom與whole/zoom等權平均，在全部canonical validation確認、再檢查subgroups／每event recall與context clipping，之後才考慮crop-trained RGB-only改版。Equal-mean兩個views約增加一倍model inference成本，需另量測end-to-end latency；不自動替換deployment、不搜尋crop比例、不重訓、不讀test。本輪未自行啟動完整validation確認。

## Full whole-plus-zoom validation dispatch (2026-09-30)

後續工作記錄於[whole-zoom-alert-future-work.md](whole-zoom-alert-future-work.md)，區分目前正在試的whole＋zoom、僅作離線分析的threshold，以及已確認但尚未接入runtime的discard-label產品scope。這是方向筆記，不是可執行設定；本次run沒有刪除類別或啟用警報。

使用者核准完整確認後，runner新增`--whole-zoom-only`與`--full-validation`；固定使用同一checkpoint、whole-video uniform8與70%-short-side center zoom，兩者120-class probabilities等權平均。完整canonical validation 12,960筆全部保留，不增加temporal views、不搜尋crop比例、不重訓、不讀test、不啟用live alarms。產品排除24類仍保留為negative examples，softmax保留全部120類。

短GPU smoke位於`artifacts/whole_zoom_validation_smoke_v1/`，48筆、4 workers、batch4；whole argmax與saved baseline零差異，logit max error為`4.51e-5`，finite logits、labels、unique IDs與保存結果通過檢查。這48筆只覆蓋A001–A048，僅用於執行與parity檢查，不作代表性accuracy估計。另以原始full-validation logits確認新增probability-based report path維持top1結果，calibration／evaluation IDs互斥，24類產品排除不與A042–A048重疊。

完整run已dispatch至tmux session `whole-zoom-validation`，job PID `698015`、logger PID `698018`；啟動時確認state為`running`且已有`inference_only`進度。Artifacts位於`artifacts/whole_zoom_validation_v1/`，`console.log`保存process output，`train.log`是dispatcher的通用progress log而非訓練紀錄；`experiment_lock.json`保存來源hashes、固定policy及sample IDs，`runner_snapshot.py`保存實際runner。完成時寫入`view_predictions.pt`、各view的`*_confusion_alert_report.json`與`run_report.json`，dispatcher更新`dispatch_status.json`。此處記錄dispatch狀態，不預先宣稱結果。

Timing報告包含offline batched decode／preprocessing與兩次forward的pipeline wall time，另記錄各view的forward加host/device transfer時間；不把它當成連續串流event onset latency。每view thresholds仍只作exploratory internal validation；baseline已用全部validation選checkpoint，且related subjects／views可能跨calibration halves，不代表獨立驗證。完成後須檢查各事件收益、誤報與裁切風險，而非只看overall accuracy；代表性input strips不保證全部samples的body coverage。

## Attention-event threshold audit

候選ontology為A042 staggering、A043 falling、A044 head-touch/headache pose、A045 chest-touch/discomfort pose、A046 back-touch/backache pose、A047 neck-touch/neckache pose、A048 nausea/vomiting condition。這些是NTU的acted action labels，不是「已確認疼痛／嘔吐」或「醫療危險」ground truth；暴力／武器類尚未列入這輪警報scope。Softmax sum只能稱為event score，不能等同校準後的danger probability。

Original full validation的A043 top1為70／72（97.22%），A048為45／72（62.50%）。所以120-class overall accuracy約59%不代表每個alert event都只有59%；產品應依event recall、precision、false alarms與latency評估，而不是只追overall120 accuracy。

在全部12,960 validation上，7-class score sum用threshold0.5時，TP326／FN178／FP70／TN12,386，recall64.68%、precision82.32%；threshold0.1時，TP444／FN60／FP402／TN12,054，recall88.10%、precision52.48%。提高門檻減少誤報但增加漏報，不存在通用的0.5安全門檻。

為避免在同一批資料上選門檻再報表現，validation另以每class sample-ID hash切成calibration6,480／evaluation6,480，含各event36個evaluation positives。Baseline此前已用全部validation選checkpoint，而且同subject／related views可跨這兩半，所以這只是exploratory internal evaluation，不是獨立test或prospective臨床／部署驗證。未使用test。

Calibrate各event達約95% recall時，內部evaluation的falling threshold約0.5124，TP35／FN1／FP1／TN6,443，recall與precision均97.22%；nausea/vomiting threshold約0.0479，TP33／FN3／FP39／TN6,405，recall91.67%、precision45.83%。每event僅36個evaluation positives，不能據此保證real-world sensitivity。非該event的其他attention events仍算該event的negatives，event-specific precision與group-warning precision須分開解讀。

7-class group的calibration約95% recall threshold為0.03136，evaluation TP241／FN11／FP459／TN5,769，recall95.63%、precision34.43%、non-event false-positive rate7.37%。因此把所有疼痛／失衡姿態綁成「高優先危險警報」目前會有很多誤報；候選應分層為falling高優先待驗證、其他姿態低優先人工複核，不將觸碰身体直接診斷為疼痛。

## Product direction, not deployment approval

2026-09-30使用者新增排除A086、A034、A011、A078、A012、A110、A004的獨立產品輸出，累計24個source actions；完整決策與歷史reduced-v1的區別見[scope review](action-scope-review.md#current-product-scope-decision-2026-09-30)。本報告的120類ranking、previously ignored標記與已保存metrics保持原樣。排除類別仍保留為attention-event negatives，不刪除樣本或logits，不在事件threshold前移除其probability mass再normalize；runtime尚未套用這次scope變更。

若產品是照護警報，event-centric／multi-label事件偵測加獨立alarm policy比「必須精確辨識120動作」更符合目標；可先用現有scores建立離線baseline，不必立刻重訓。這個任務重定義不是模型能力提升，不能把coarse-group accuracy當成120-class突破。

需要依事件分開選operating point，評估recall／漏報、precision、false alarms per hour、event onset latency、不同場景與人群、多人與遮擋，以及unknown／out-of-distribution輸入。連續stream需event-level annotations、長時間正常活動negative footage與影像quality checks；NTU pretrimmed acted clips不能估計每小時false alarms或連續窗口的timing。可研究hysteresis、debounce與cooldown減少重複警報，但短促falling也可能因連續支持要求而延遲／漏掉，需另外驗證。

高優先警報與低優先「需關注」notification應分層，先human-in-the-loop／shadow mode，不以模型取代照護判斷，也不自動啟動emergency escalation。實際event ontology、可接受漏報／誤報率與response workflow仍需產品使用者／照護專業人員确认。[NIST AI RMF](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)強調false-positive／false-negative、部署情境外部有效性及持續監測；此處只是工程規劃，不宣稱醫療安全認證。
