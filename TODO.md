# TODO

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
- [ ] 將 SkeletonCLIP teacher distillation 作為優先的 representation adaptation 路線：teacher 使用 NTU skeleton 並在 training split 訓練，RGB XCLIP student 先以 ground-truth action cross-entropy 加 teacher／student logit KL loss 訓練，再用受控 ablation 評估 projected feature cosine loss。
- [ ] 第一版 distillation student 在 XCLIP MIT attention 加入 LoRA；只有 validation 支持時才將 LoRA 擴展到 vision encoder 上層。Ground-truth loss 必須保留，避免 student 完全複製 teacher 錯誤。
- [ ] Distillation 前先單獨評估 SkeletonCLIP teacher，並以 NTU sample identity 精確對齊 RGB 與 skeleton。Teacher 不得使用 validation／test labels 訓練，最終 validation／test 必須以 RGB student 單獨推理。
- [ ] LoRA training 必須執行線上 XCLIP forward／backward，不能沿用 frozen baseline 的 post-XCLIP 512 維 feature cache；訓練完成並固定 student 後，merge LoRA 到 base weights，再以 merged checkpoint 重建新的 evaluation feature cache。
- [ ] 部署合約維持 RGB-only inference：不載入 SkeletonCLIP、skeleton data、pose estimator、distillation loss、optimizer 或 backward graph。Merged LoRA 不增加額外 adapter branch；推理成本主要仍是八幀 decoding 與一次 XCLIP forward。
- [ ] 對長影片評估多個八幀 temporal windows 並聚合 logits，不依賴單一 random window。
- [x] XCLIP 支援相同的 participant-disjoint split manifest、validation checkpoint selection 與 random seed，並以獨立 artifact 目錄執行。
- [ ] 完整 benchmark 再統一共同 metrics、augmentation／balancing policy 與多 seed 報告後，比較 CLIP＋BiGRU 和 XCLIP。

## 訓練管線架構重整

> 目前四條管線皆可獨立執行；以下屬於降低重複與方便後續擴充的架構債，不阻擋 frozen XCLIP baseline 在伺服器執行。新增第五條訓練管線前應優先完成。

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
- [ ] 新增 pooled-feature branch，使用 mean、max、standard deviation 等
      temporal statistics，再與 GRU output 融合。
- [x] Multiclass 與 prototype 預設使用全部找到的影片，避免先前固定的每類
      sample budget 成為主要資料瓶頸；快速實驗仍可選用
      `--max-videos-per-class`。
