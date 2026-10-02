# TODO

## 手部與互動區域推理診斷（2026-10-01）

- [x] 完成 11 筆 GPU smoke、176 筆 skeleton 定位診斷，以及同批 176 筆 RGB-only MediaPipe 定位與分類比較；不訓練、不讀 test。
- [x] 固定 whole＋zoom＋RGB interaction 為 121／176（原 104／176），修正 21、退步 4；目標三類 19→23／48。收益主要來自咳嗽、吃東西，搧風與摸頸未有合併增益；不當成 86 類 full validation 準確率、不替換 baseline。詳見 `docs/hand-roi-v1-audit.md` 與 `docs/rgb-hand-roi-v1-results.md`。
- [x] 擴大到 1,228 筆（新增 1,052 筆）並完成 2,280 train 單 pass 局部 expert：相比未訓練局部合併，全選樣淨增 26、保留類別淨增 21；吃東西 35→45／72、咳嗽 37→49／72，摸頸仍 24→20／72，不替換 baseline。詳見 `docs/hand-roi-followup-v1.md`。
- [x] 完整保留 86 類 validation（8,640 支）：whole＋zoom 79.9769% → 加入 trained RGB interaction 84.2245%，修正 505、退步 138、淨增 367；先前診斷之外 7,548 支淨增 283。詳見 `docs/hand-roi-full86-v1-results.md`。
- [ ] 摸頸需另處理（仍 24→20／72），剪指甲另建固定 RGB 診斷；候選整合需評估 RGB 定位與第三次 forward 的推理成本。

## Native attention supervision（2026-10-01）

- [x] 完成原生attention研究與實作：最後兩層heads0–3的CLS→patch soft supervision，保留message token、RGB-only merged inference與獨立audit／smoke／pilot runner；詳見`docs/native-attention-supervision-v1-plan.md`。
- [x] 10項相關tests、472筆attention audit與8-step GPU smoke通過；native attention到Q gradients與merged RGB-only strict reload通過。Smoke沒有分類增益，attention mechanism direction也未通過，不當成winner。
- [x] 全量80,640 train、2 epochs pilot完成：raw120為58.9429%（baseline58.9198%，net+3），`inconclusive`；保留96類68.1270%→67.7900%（net−33），不推薦替換、不延長recipe。Attention audit mechanism direction通過但分類收益不足；未讀test。完整結果見`docs/native-attention-supervision-v1-plan.md`，artifacts位於`artifacts/native_attention_v1/pilot_seed42/`。

## Crop-aware 短 pilot（2026-09-30）

- [x] Whole＋zoom完整validation已完成：58.92%→66.94%，無重訓；檢查代表性退步樣本，發現中央zoom可能裁掉倒地body。
- [x] Crop-aware GPU smoke通過：warm-start parity、vision／MIT／classifier gradients與merged RGB-only strict reload通過。
- [x] 短pilot與12,960筆validation已完成：whole58.96%、zoom64.31%、ensemble67.14%，ensemble net +26但attention349→343，inconclusive、不推薦替換。4,096筆train單pass，log與產物位於`artifacts/crop_aware_v1_pilot/`；未改threshold或discard-output、未讀test。詳見`docs/crop-aware-v1-plan.md`；尚未達到單次推論全量raw120超過70%。

## 資料切分與評估

- [x] 依參與者（participant）切分 train／validation／test，而不是逐部影片
      隨機切分，確保同一位參與者不會出現在多個 split。
- [x] 為 multiclass 與 prototype 管線加入 validation split，讓三條訓練
      命令都使用 train／validation／test。
- [x] 使用 validation 指標選擇 multiclass 與 prototype 的最佳 checkpoint；
      完成 checkpoint 選擇後，test set 只評估一次。
- [ ] 決定三條訓練管線是否應共用 early stopping 規則。
- [x] 讓 split 產生結果可重現，並在 split manifest 記錄 seed 與選中的影片。

## Benchmark 公平性與可重現性

> 目前延後處理，不阻擋近期的影片取樣與 XCLIP 開發。需要正式對外比較或
> 發表結果時，再依優先順序逐項完成。

- [ ] 產生一份 canonical split manifest，要求 multiclass、prototype、
      binary 與未來的 XCLIP 管線都讀取它，而不是各自重新產生 split。
      各任務的 label 應從相同的影片記錄衍生。
      - 已完成基礎支援：四條現有管線都可透過 `--split-manifest` 重用既有 manifest。
- [ ] 在每份實驗報告中明確記錄 split 身分，例如 manifest hash 或穩定的
      split ID；比較不同 manifest 的執行結果時應立即報錯。
- [ ] 正式 benchmark 採用 NTU RGB+D 120 官方 cross-subject protocol；
      從官方 training subjects 內建立 validation，官方 test subjects
      不參與開發。若只使用資料子集，報告中應標示為 official-partition
      subset，而不是完整官方 benchmark。
- [ ] 加入 participant-level stratification 或 deterministic allocation
      search，讓 train／validation／test 保有合理的 action 與 safe／danger
      分布。除了 participant 數，也要報告實際影片數，因為 participant
      比例不等於影片比例。
- [ ] 使用資料量上限時，統一不同任務的語意。目前 multiclass 是限制每個
      action class，binary 則限制聚合後的 safe／danger 類別，可能改變原始
      action 分布。
- [ ] 每次執行都報告原始與實際訓練資料量，包括 augmented samples、
      每個 epoch 的 batch 數、optimizer step 總數與約略運算時間。只有
      epoch 數相同不代表 compute budget 相同。
- [ ] 加入 controlled benchmark mode，讓所有管線使用相同的 augmentation
      與 class-balancing policy。任務專屬政策應是另外命名的實驗，而不是
      隱含預設值。
- [ ] 為 multiclass 與 prototype 加入 macro F1、各類別
      precision／recall／F1、balanced accuracy 與 confusion matrix，不再
      只用 accuracy 選模與報告。
- [ ] 保留各任務自己的主要指標，但定義共同的報告 schema。Binary accuracy
      與 multiclass accuracy 代表不同預測問題，不應直接當成同一指標比較。
- [ ] 報告模型參數量、可訓練參數量、inference latency 與記憶體用量。
      區分「共用 hyperparameters 的 controlled comparison」與「各架構經
      validation 調整後的 best-achievable comparison」。
- [ ] 使用固定 seed 集合重跑 benchmark，報告 mean 與 standard deviation；
      在支援的平台啟用 deterministic PyTorch algorithms，並記錄仍不具
      deterministic 行為的 CUDA／MPS operations。
- [x] 為 multiclass、prototype、binary 與 XCLIP 設定不同的預設 artifact 目錄，避免連續執行時互相覆蓋 checkpoint、plot、manifest 或 evaluation JSON。
- [ ] 為每次執行加入 run ID，並記錄套件版本、device、命令列參數、Git
      revision、split ID、cache 設定與模型設定。
- [x] 在 embedding cache filename 加入來源路徑 hash，避免不同資料夾中
      的同名影片使用同一個 tensor；測試已涵蓋相同 filename、不同路徑。
- [ ] 每次執行保留自己的 cache manifest，或讓共用 manifest 採累積寫入；
      不可在另一條管線執行時覆蓋唯一的 cache audit 記錄。
- [ ] 加入端到端 benchmark validation 命令，在接受比較結果前檢查
      participant disjointness、manifest equality、label coverage、class
      distribution、cache provenance、augmentation policy、metric schema
      與 artifact isolation。

## 影片與影格取樣

- [x] 三條 CLIP＋BiGRU 訓練管線改用 time-based 6 FPS，最多保留 32 個真實 frames；超過上限時從完整影片均勻縮減。
- [x] Embedding cache 只儲存真實長度，DataLoader 組 batch 時才動態補到該 batch 的最長序列。
- [x] XCLIP 與 XCLIPVision experiments 從完整影片 deterministic uniform 取 pretrained checkpoint 原生的 8 幀。
- [ ] 決定 standalone per-frame CLIP experiment 應保留隨機連續片段，還是改為涵蓋整部影片。
- [ ] 評估是否要為長影片取多個 temporal windows，並聚合它們的預測。
- [ ] 將 experiment sampling strategy 與 frame sample rate 暴露為 CLI 選項。
- [x] 在文件中說明 training 與 XCLIP experiments 各自的取幀策略。

## XCLIP 影片分類器

- [x] 新增 visual-only XCLIP video-classification baseline，和目前 frozen CLIP frame encoder＋BiGRU 比較。
- [x] 使用 `XCLIPModel.get_video_features()`，保留 visual projection 與 Multiframe Integration Transformer（MIT）的完整 video path，不執行 text model。
- [x] 使用 pretrained checkpoint 原生的八幀輸入，將 512 維 video feature 傳入 LayerNorm、dropout 與 linear classification head。
- [x] 第一階段凍結全部 XCLIP 參數，只訓練 classification head。
- [ ] 評估解凍 MIT 與 visual projection；只有 validation 結果支持時，才考慮部分或完整 fine-tune vision encoder。
- [x] 將 SkeletonCLIP teacher distillation 作為優先的 skeleton-informed temporal／decision adaptation 路線：Phase 1 在 paired train 上選出 BlockGCN linear classifier teacher（XCLIP text-prototype head 未通過預先固定的 NLL／ECE quality gate），Phase 2 以 ground-truth CE 加 A001–A059 masked teacher-logit KL 訓練 RGB XCLIP student；Gate 4 九組 validation-only search 結論為 `no-go`，最佳 `T=1, λ=0.1` accuracy 比 CE control 低 `0.1157` points，未讀 test、不做 seed-43 confirmation。詳見 `docs/skeletonclip-teacher-phase1.md` 與 `docs/skeletonclip-distillation-phase2.md`。
  - Projected feature cosine loss 未啟動；Phase 2 明確排除此項，若日後研究需另立 phase 與自己的 CE control。
- [x] 在 distillation 前完成 XCLIP MIT `q_proj`／`v_proj` LoRA + ground-truth CE-only RGB student control，並在固定 split 上完成 Stage 0–5 validation-only tuning、multi-seed confirmation 與 locked-winner test evaluation；winner recipe 與結果記錄於 `docs/xclip-lora-tuning-plan.md`。
- [x] Phase 3 將 LoRA 擴展到 XCLIP vision encoder 最後兩個 blocks，定位為 vision adaptation，而非直接證明 joint focus。120-class vision-only CE run 最佳 epoch 13 validation accuracy為`58.9198%`；完整validation logits已用於action taxonomy與subgroup分析，但仍缺同設定完整frozen control，因此不可宣稱Gate 1 paired comparison通過。詳見`docs/skeletonclip-phase3-4-roadmap.md`與`docs/xclip-vision-lora-validation-action-accuracy.md`。
- [x] 完成`reduced-v1` action taxonomy與兩種驗證路徑：既有120-class upper-vision probabilities經aggregation得到`65.8565%` exact／`67.8395%` public validation accuracy；direct 106-output upper-vision retrain最佳為`61.9830%` exact，故短期部署候選維持120-class checkpoint加inference-time probability aggregation，direct checkpoint只保留為control。詳見`docs/action-scope-review.md`。
- [ ] 只有研究目標需要 joint-focused RGB learning 時，才另行設計 patch-token 與 skeleton-to-RGB joint spatial supervision，並用 localization、joint-region masking／occlusion 驗證；不可由 class-level KD accuracy 推論模型更注意 joints。
  - 兩次motion-pooling pilot已完成：joint adaptation為`58.11%`，frozen-baseline attention-only為`58.94%`，baseline為`58.92%`。第二輪僅多答對3／12,960個validation samples，未達confirmation門檻；不選winner。先前依使用者指示暫停並整理成果，之後使用者要求嘗試SKI方法，改以`docs/ski-semantic-v1-plan.md`的semantic distillation繼續。只看120-class；所有logs／checkpoints保留、未讀test。Motion-pooling不是SKI原版semantic distillation的重現。詳見`docs/spatial-attention-v2-results.md`與`docs/skeleton-guided-spatial-supervision-plan.md`。
  - SKI-style semantic smoke通過，單一bounded pilot已在tmux完成：`58.98%`，比baseline多答對8／12,960，gain僅`+0.0617` percentage points；paired淨增19、unpaired淨減11，判為inconclusive。這是supervised120／partial59改版，不是官方zero-shot原版；不選winner、不自動放大full training。詳見`docs/ski-semantic-v1-plan.md`。
  - Direct-decision KD＋unpaired RGB retention的smoke通過、單一tmux pilot完成：`58.96%`，overall net +5、paired -15、unpaired +20；MacroF1下降，未達overall門檻，inconclusive、不選winner、不自動放大budget。Unpaired validation T2 KL由前輪`0.06703`降至`0.03393`，不能據多項改动宣稱anchor的因果貢獻。Logs／checkpoints完整保留、未讀test；設計與分析見`docs/decision-distillation-v1-plan.md`，不是官方SKI重現。
  - RGB-input／alert診斷已完成，無訓練：固定480樣本的whole為`60.83%`、central zoom70為`68.75%`、whole＋zoom mean為`72.29%`，不能宣稱全量突破70%。使用者核准後，48筆GPU smoke與baseline parity通過，12,960筆full validation已完成：whole58.92%、zoom63.91%、whole＋zoom66.94%；log與結果位於`artifacts/whole_zoom_validation_v1/`。全部120類含24個產品排除類別仍保留作negative examples，A042–A048 thresholds只作exploratory internal evaluation，不啟用live alarms；詳見`docs/rgb-input-and-alert-audit.md`。
- [x] Distillation 前先單獨評估 SkeletonCLIP teacher，並以 NTU sample identity 精確對齊 RGB 與 skeleton。Teacher 不得使用 validation／test labels 訓練，最終 validation／test 必須以 RGB student 單獨推理。Phase 1 Gate 3 完成 teacher 比較，Phase 2 Gate 0–1 完成 identity lock 與 45,236 筆 paired train teacher-logit cache。
- [x] LoRA training 從獨立的 pre-MIT `[8, 512]` cache 執行 MIT forward／backward，不沿用會繞過 MIT 的 frozen post-MIT cache；完成 adapter／merged checkpoints、trainable-parameter audit 與 merge parity validation。
- [ ] 以 merged checkpoint 重建可直接從 RGB frames 執行的完整 inference path 與 evaluation cache，並驗證其結果和目前 pre-MIT cached evaluation 一致。
- [ ] 部署合約維持 RGB-only inference：不載入 SkeletonCLIP、skeleton data、pose estimator、distillation loss、optimizer 或 backward graph。Merged LoRA 不增加額外 adapter branch；推理成本主要仍是八幀 decoding 與一次 XCLIP forward。
- [ ] 對長影片評估多個八幀 temporal windows 並聚合 logits，不依賴單一 random window。
- [x] XCLIP 支援相同的 participant-disjoint split manifest、validation checkpoint selection 與 random seed，並以獨立 artifact 目錄執行。
- [ ] 完整 benchmark 再統一共同 metrics、augmentation／balancing policy 與多 seed 報告後，比較 CLIP＋BiGRU 和 XCLIP。

## 訓練管線架構重整

> 目前 classifier 管線與 XCLIP LoRA tuning orchestrator 皆可獨立執行；以下屬於降低重複與方便後續擴充的架構債，不阻擋 frozen XCLIP baseline 或 LoRA tuning 在伺服器執行。

- [ ] 抽出共用 multiclass split preparation，統一 class discovery、`--split-manifest` reuse、video-cap validation、class reindexing、split summary 與 manifest 寫入。
- [ ] 讓 training engine 透過清楚的 batch／forward adapter 同時支援 variable-length sequence 與 fixed video feature，合併目前重複的 `train_epoch()` 與 `train_feature_epoch()`，但不要把 binary 專屬 loss 與選模規則藏進難以理解的萬用介面。
- [ ] 合併 sequence classifier 與 fixed-feature classifier 的 multiclass evaluation，共用 prediction-row schema 與 accuracy 計算。
- [ ] 抽出 validation-selected training loop 與共用 artifact orchestration，統一 history、best-checkpoint restore、test-once evaluation、checkpoint、results JSON 與 accuracy plot 寫入。
- [ ] 重整後將各 pipeline module 保持為薄的 task configuration／backend wiring，並用現有 contract tests 與端到端 smoke test確認輸出、split membership、cache identity 與 validation/test 行為不變。
- [ ] 暫不建立過度泛化的 universal pipeline；binary augmentation、class balancing、macro-F1／balanced-accuracy checkpoint selection 等任務差異必須維持明確且可審查。

## 分層式 Danger 與 Action 預測

- [ ] 訓練前定義並記錄 action-to-danger label ontology。不可在未確認產品
      定義的情況下，假設 IDs `42`、`43`、`48` 以外的 action 全部安全。
- [ ] 優先使用一個共用 XCLIP video encoder 搭配兩個 prediction heads，
      不要執行兩個彼此獨立的模型：
      - binary head：預測 danger probability；
      - multiclass head：預測具體 action。
- [ ] 使用加權 binary danger loss 與 multiclass action loss 聯合訓練兩個
      heads，再評估 shared multi-task features 是否改善任一任務。
- [ ] 避免 hard routing，也就是最初預測為 safe 後就不執行 action
      classification。Action output 應永遠計算，避免 binary false negative
      默默丟失有用證據。
- [ ] 根據 validation set 與安全需求選擇 danger alert threshold，不固定為
      `0.5`；可以先評估 danger recall 至少 95% 的目標。
- [ ] 除了 overall accuracy，還要報告 danger recall、false-negative rate、
      precision、PR-AUC 與 false alarms；action head 則報告 macro F1 與
      confusion matrix。
- [ ] 加入同時要求 danger decision 與具體 action 都正確的 end-to-end
      metric，並在 participant-disjoint test set 評估全部指標。

## Augmentation 稽核與重整

- [ ] 將 class balancing 與 augmentation 分離。Imbalance 使用 class
      weights 或 weighted sampler；safe 與 danger samples 使用相同的
      augmentation 類型與機率，避免 transform 痕跡成為 danger label 的
      shortcut。
- [ ] 在可行時，將目前一次性產生的 offline augmented embedding copies
      改為 training-time clip augmentation。每個 epoch 重新取樣參數，但同一
      clip 內所有 frames 必須使用相同 spatial transform。
- [ ] 新增 video-specific temporal augmentation experiments，包括 random
      temporal windows、temporal jitter、frame dropout 與 playback-speed
      variation。
- [ ] 選擇並記錄適當的 fill strategy，避免 PIL 預設黑色填補造成 rotation
      border。
- [ ] 執行四組受控 ablation：
      - 無 augmentation、無 class weights；
      - 只有 class weights；
      - 只有 augmentation；
      - augmentation 與 class weights 都使用。
- [ ] 在 cache compatibility metadata 加入來源影片 fingerprint，例如
      file size＋modification time 或 content hash，讓相同路徑的影片被替換時
      能使舊 embedding 失效。
- [ ] 新增明確、可審查的 cache-pruning 命令，移除 active manifest 已不再
      引用的 augmented embeddings；training 過程不可自動刪除 stale cache。
- [ ] 在 cache manifest 與 experiment report 記錄每個 augmented sample 的
      split、class、seed 與完整 transform parameters。
- [ ] 新增測試，涵蓋 train-only augmentation、clip 內所有 frames 使用相同
      transform、deterministic seeds、不同 cache keys、cache invalidation，
      以及 validation／test 不含 augmented samples。

## 其他可能的訓練改善

- [ ] 增加 regularization 與 sequence augmentation，例如 random frame
      dropout、temporal jitter 或在 embeddings 加入小幅 Gaussian noise。
      - 已在 XCLIP LoRA tuning mode 加入 cache-safe、training-only 的 frame dropout、temporal jitter 與 Gaussian feature noise；其他訓練管線仍未統一。
- [ ] 新增 pooled-feature branch，使用 mean、max、standard deviation 等
      temporal statistics，再與 GRU output 融合。
- [x] Multiclass 與 prototype 預設使用全部找到的影片，避免先前固定的每類
      sample budget 成為主要資料瓶頸；快速實驗仍可選用
      `--max-videos-per-class`。

- [x] 產品範圍新增排除 A030／A089／A090／A067／A010／A032／A079／A023／A107／A106，現保留 86 類、排除 34 類；既有 validation 結果已重算，後續不優先處理這十類。見 `docs/product-evaluation-scope.md`。
