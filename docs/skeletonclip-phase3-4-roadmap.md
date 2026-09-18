# SkeletonCLIP Phase 3–4 roadmap

日期：2026-09-18。狀態：Phase 2 已完成 Gate 0–4，class-level masked-KL 結果為 no-go；Phase 3 Gate 0 已通過，vision-only CE run 在 epoch 37 後因 validation 退步而停止，最佳 epoch 13 validation accuracy 為 58.92%；同設定的完整 frozen control 尚未完成，因此原 Gate 1 的 paired comparison 尚未通過；Phase 4 Gate 0 與 Gate 1 功能 smoke 已通過，Gate 2 三組各 20 epochs 已完成，定位有小幅 conditional signal，分類 accuracy 無明顯改善，Gate 3 暫不啟動。Gate 編號在各 Phase 內獨立計算，本文以「Phase 3 Gate 0」或「Phase 4 Gate 0」避免混淆。

## 1. 目的與階段邊界

Phase 2 只測試 frozen XCLIP vision encoder 下的 class-logit KD。結果不能回答 vision representation 是否需要更新，也不能回答模型是否真的使用人體 joints。因此後續拆成兩個獨立 phase，避免把不同機制的效果混在一起。

| Phase | 研究問題 | 主要改動 | 部署輸入 | 是否依賴前一階段成功 |
| --- | --- | --- | --- | --- |
| Phase 3：vision adaptation | 更新 vision features 是否比 frozen vision 更有效？ | 對 vision encoder upper spatial Transformer blocks 加 LoRA；先測 CE-only，再視 gate 結果測 vision LoRA + KD | RGB-only | 不要求 Phase 2 KD 成功，但需先通過自己的 CE control |
| Phase 4：spatial joint supervision | 直接監督模型關注 skeleton joints 對應的 RGB regions 是否可行？ | 保留 patch tokens，加入 joint-to-frame 對齊及 spatial loss／評估 | 預設仍 RGB-only；training-only skeleton supervision | 不要求 Phase 3 accuracy 成功，但需通過資料對齊與可觀測性 gate |

`xclip-mit-lora-ce-plan.md` 中的「Phase 3」是早期 XCLIP baseline 的 pre-MIT cache/training path，已屬歷史實作階段，不等同於本 roadmap 的新 Phase 3。

## 2. Phase 3：vision adaptation

### 假設

目前 student 只訓練 MIT LoRA 與 classifier，使用 pooled `[8, 512]` pre-MIT cache；因此 teacher signal 無法改變 spatial vision features。Phase 3 測試解凍 vision upper blocks 是否能帶來額外分類效益。

### 實驗順序

1. 在同一個 RGB forward／dataloader／evaluation path 上，以相同 split、seed、training budget 與 MIT/head recipe 分別訓練 frozen-vision CE 與 vision-LoRA CE；兩組都是 Phase 3 Gate 1 的正式 control pair。
2. 以這兩組的 validation 差值判斷 vision adaptation；Phase 2 的 `λ=0` CE 結果只作歷史參考，Phase 2 KD no-go 不作 vision-LoRA baseline。
3. Class-level KD 不列為預設實驗；只有在 Gate 2 提出新的明確假設與固定 search／control contract 時才另行測試。Test 只在 locked 決策後執行一次。
4. 分開報告完整 120 類、A001–A059、A060–A120，以及 paired／unpaired subgroup；不得只報 aggregate accuracy。

### 必要架構與資料變更

- 不再完全依賴 pooled pre-MIT cache，必須從 RGB frames 執行 vision forward，並保存 patch-token／feature shape 與 model revision。
- 明確列出可訓練的 vision blocks、LoRA target modules、rank、alpha、dropout、learning rate 與 trainable-parameter audit。
- Gate 1 的 frozen-vision CE 與 vision-LoRA CE 必須共享同一 RGB dataloader、processor、初始化政策、MIT/head、feature noise、optimizer budget、evaluation 與 checkpoint selection path；唯一架構差異是 vision LoRA 是否啟用。
- KD 仍是 class-level supervision；即使 accuracy 改善，也不能宣稱模型已學會 joint focus。

### Gate 0：vision path compatibility smoke

#### 工作

- 鎖定 pretrained XCLIP revision、RGB 八幀 sampling 與 processor identity，使用與 Phase 2 相同的 MIT/head recipe；第一輪 vision LoRA 明確指定最後兩個 spatial Transformer blocks 的 q/v projections、rank 16、alpha 32、dropout 0.05。新增 vision LoRA learning rate 時先記入 experiment lock，不把它隱藏在既有 MIT learning rate 內。
- 從 RGB frames 執行 vision encoder、visual projection、MIT 與 120-class head；保留可供 Phase 4 使用的 patch-token 接點，但 Phase 3 不加入 spatial loss。
- 驗證 trainable whitelist、vision LoRA gradient、adapter checkpoint round trip、LoRA merge parity、RGB-only reload，以及真實 pretrained model 在預定 batch size 下的 GPU memory／throughput。

#### Pass gate

- 固定輸入的 forward 輸出為 `[batch, 120]` 且 finite；只有選定 vision／MIT LoRA 與 classifier 可訓練，vision LoRA 收到非零梯度，base weights 沒有 gradient。
- Adapter strict reload 後 logits 一致；FP32 merge 前後 `max_abs_diff ≤ 1e-5`；RGB-only reload 不依賴 skeleton 或 teacher cache。
- 真實 pretrained model、實際 RGB processor 與預定 batch size 通過顯存／速度檢查。縮小版模型的 shape／gradient smoke 不能單獨視為本 Gate pass。

#### 預期產物

- 固定的 Phase 3 config／identity lock、trainable-parameter audit、smoke 結果與 adapter／merged checkpoint schema。

#### 目前執行結果

Phase 3 Gate 0 已以 `uv run vlm train skeleton-phase3 --batch-size 16 --artifacts-dir artifacts/skeletonclip_phase3/gate0_locked` 在 RTX 2080 Ti 完成。從 canonical train split 讀取 16 支不同 RGB 影片，whole-video 八幀經鎖定 revision 的 `XCLIPProcessor` 得到 `[16, 8, 3, 224, 224]`，模型輸出有限值的 `[16, 120]` logits。最後兩個 vision blocks 與 MIT 的 q/v 共六個 LoRA targets，加上 head 共 193,656 個可訓練參數；vision LoRA 有非零梯度，frozen base 無梯度。一次 FP32 forward／backward 約 0.413 秒（僅模型步驟，不含 decode／processor），peak allocated VRAM 約 1.47 GB；merge `max_abs_diff=1.13e-6`、adapter reload `0.0`、merged RGB-only reload `1.13e-6`，均通過 Gate 0。Lock、報告與 checkpoints 位於 `artifacts/skeletonclip_phase3/gate0_locked/`；`test_evaluated=false`。這是單 batch 的相容性與資源 smoke，不代表完整 epoch 的 dataloader throughput 或分類品質。

### Gate 1：同路徑 frozen-vision／vision-LoRA CE control

#### 工作

- 先鎖定 vision LoRA learning rate、batch size／gradient accumulation、AMP、RGB decode／processor、projected feature noise、checkpoint selection 與兩組 artifact identities；從相同 pretrained XCLIP 與 seed 42 初始化，在同一訓練程式中以 vision upper blocks `0` 與 `2` 跑兩組 CE-only full training。兩組均使用 canonical train 80,640／validation 12,960、60 epochs、相同 MIT/head recipe，維持 validation-only selection 與 test embargo。
- 每 epoch 保存兩組的 sample counts、finite loss／gradient、validation accuracy 與 macro-F1；完整記錄 RGB 前處理、各 parameter group learning rate、optimizer steps、best epoch、validation predictions、耗時與 peak VRAM。Phase 2 seed-42 frozen-vision CE 的 52.1065% 僅用來檢查新 RGB path 是否大幅漂移，不取代同路徑 control。

#### Pass gate

- 兩組每 epoch 各恰好消費 80,640 個 CE samples；trainable audit 證明 frozen control 只有 MIT LoRA/head 可更新，vision arm 額外只有預定 upper-block LoRA 可更新。Checkpoint 均可 strict reload，merged RGB-only logits 符合 Gate 0 parity；完整 validation artifacts 可重建各自的 best epoch 與 metric。
- 主比較只使用兩組同路徑 seed-42 validation 差值，並同步報告相對 Phase 2 歷史 CE 的差距與可能原因；不把舊路徑的結果當作唯一 control。

#### 預期產物

- `artifacts/skeletonclip_phase3/gate1/frozen_vision_ce_seed42/` 與 `artifacts/skeletonclip_phase3/gate1/vision_lora_ce_seed42/` 下各自的 config、trainable audit、progress、best adapter／merged checkpoint、validation predictions，以及一份 paired CE comparison summary。Gate 1 預定兩組完整 60-epoch training。

#### 目前執行狀態

兩組共用 RGB dataloader、CE 訓練、validation 與 checkpoint path；各自的一批 train／validation smoke 與 merged RGB-only strict reload 已通過，smoke accuracy 不作模型品質解讀。原定兩組各 60 epochs 的首次正式 run 曾啟動，先以 4 個再以 16 個資料載入 workers 測速；16 workers 在 frozen-vision 第一個 epoch 約 20.8 samples/s，推算兩組訓練需約一週或更久。使用者認為此耗時不可接受，故在第一個 epoch 尚無 checkpoint 前停止；該次沒有正式 validation 結果。後續 frozen cached run 僅完成 epoch 1，vision-only run 則有完整 validation；由於兩組未在同一訓練設定下完成，Gate 1 的 paired comparison 仍未通過。

使用者選擇先做數小時 pilot。Pilot 從 Phase 2 的 seed-42 CE-only best adapter／head 共同 warm-start，對 canonical train 各 class 依 sample path SHA256 排序取 10%（合計 8,040 筆、120 類），兩組共用完整 12,960 筆 validation、RGB path、seed、MIT/head 與八個 training epochs；只有最後一個 epoch 評估完整 validation，避免以小樣本 smoke accuracy 選模型。兩組固定 MIT LoRA LR `2e-5`、head LR `8e-5`，vision arm 額外 upper-two-block LoRA LR `1e-4`；test embargo 持續有效。Pilot 是既有 student 的短期增量適應性診斷，不等同於從 pretrained XCLIP 跑滿 canonical 60 epochs 的 Gate 1 結論；pilot 的 full-validation accuracy 差值與 subgroup 結果只決定是否值得設計下一步，不直接宣稱正式 go。共同初始化的 RGB logits hash 必須相同；兩組一批 train／validation、checkpoint merge／strict reload 的 warm-start smoke 已通過。Pilot 於 2026-09-17 14:05（台北時間）以 `tmux` session `phase3_pilot` 啟動，輸出位於 `artifacts/skeletonclip_phase3/pilot_seed42/`，頂層及每組均使用 `train.log`；完整 validation 結果見下文。

#### 速度瓶頸與後續加速檢查

Phase 2 的快速訓練重用已算好的 pooled `[8, 512]` pre-MIT features；Phase 3 的 RGB path 每個 epoch 都要解碼影片、取八幀、跑 processor 和完整 vision forward。Frozen vision 只表示 vision 權重不更新，仍須執行 forward；vision-LoRA arm 額外對上層兩個 blocks 做 backward，因此每批可能更慢，幅度以兩組實測 throughput 為準。Pilot frozen arm 第 1 個 epoch 於 350/503 batches 的累計速率約 20.0 samples/s；早期速率受載入暖機影響，不能直接當作正式訓練的固定 ETA。

加速候選先以相同資料與模型做短程吞吐測試：在顯存允許下比較 batch 16、32、64，以及資料載入 workers 的實際效益，再依 GPU 利用率與 CPU 解碼時間判斷瓶頸；改 batch 必須同時套用兩組並重鎖訓練設定，不能與既有結果直接當作完全同預算比較。若重複影片解碼仍主導耗時，可建立帶有影片／抽幀／processor revision provenance 的 processed RGB frames cache，先驗證 cached 與線上 pixels 一致，再計算建置時間和磁碟成本是否值得。Phase 2 的 pre-MIT feature cache 已繞過 vision encoder，不能直接拿來訓練 vision LoRA。Pilot 與後續 benchmark、run 的實測結果如下。

Gate 1 已加入可選的 `--rgb-cache-dir cache/skeletonclip_phase3_rgb`：逐影片保存 lossless FP32 processor pixels `[8, 3, 224, 224]`，以來源絕對路徑、大小、mtime、whole-video 抽幀規則、processor 設定及 model revision 驗證 cache。首次讀取仍需解碼及 processor；後續 epoch 和另一組 arm 可重用，兩組 RGB forward 仍相同。每筆約 4.8 MB，完整 train+validation 約需 450 GB，pilot train+validation 約需 101 GB（不含檔案系統額外負擔）；啟用前應先做 cached／online pixels parity 和短程吞吐量測。已在執行的 pilot 不會因程式碼更新自動改用 cache。

Pilot 已完成：frozen validation accuracy `0.5169753`，vision LoRA `0.5323303`，差值 `+1.5355` 個百分點；兩組初始 logits hash 相同，未評估 test。這是 10% train／8 epochs 的 warm-start 訊號，不能取代正式 Gate 1。正式雙組 60-epoch run 於 2026-09-17 16:23（台北時間）在 `tmux` session `phase3_gate1_full` 啟動，從 pretrained XCLIP、seed 42 開始，使用完整 canonical train／validation、batch 16、16 workers 和 `cache/skeletonclip_phase3_rgb`；完整指令及輸出在 `artifacts/skeletonclip_phase3/gate1_full_cached_seed42/launch.sh` 與同目錄 `console.log`。真實影片的 cached／online pixels 已確認 `max_abs_diff=0`；cache 對長期訓練的實際加速幅度尚待量測。

此正式 run 已依使用者要求停止：frozen epoch 1 完成並保存 `last_checkpoint.pt`，train 吞吐約 `16.9 samples/s`，首次完整 validation accuracy `0.3663`；cache 全部建完後 frozen epoch 2 前 8,000 筆只有約 `14.5 samples/s`，比首次解碼更慢，主機 I/O pressure 長時間偏高。中斷發生於 epoch 2，該 epoch 尚未 checkpoint，vision arm 尚未開始。相同程式碼與 cache 設定可由 epoch 2 重啟，但更換 cache 策略需新 run identity；在短程吞吐對比前不重啟正式 run。

無 cache、vision LoRA、16 workers 的 batch benchmark 使用同一批 1,792 支 train 影片，各組前 256 筆暖機、後 1,536 筆計時。順序 16→32→64 的吞吐分別為 `20.77`、`22.32`、`35.96 samples/s`；反向順序 64→32→16 為 `34.43`、`20.73`、`16.97 samples/s`。Batch 64 的 peak allocated VRAM 約 `2.78 GB`，兩次均最快；兩種順序的差異提醒短測仍受 I/O 狀態影響。Benchmark 腳本與原始結果在 `artifacts/skeletonclip_phase3/batch_benchmark/`。若改用 batch 64，optimizer steps／epoch 會變成 batch 16 的四分之一，應作為新訓練設定，不與既有正式 run 視為相同 budget。

使用者決定使用 batch 64、關閉 processed RGB cache，先跳過 frozen arm。Vision-only 60-epoch run 於 2026-09-17 19:30（台北時間）在 host `tmux` session `phase3_vision_b64` 啟動；指令與輸出在 `artifacts/skeletonclip_phase3/vision_batch64_nocache_seed42/launch.sh` 和同目錄 `console.log`。新 CLI 旗標 `--allow-batch-size-change` 明確記錄偏離 Phase 2 winning batch 16 的決定，其餘 MIT/head recipe 維持既定設定。此 run 可評估 vision LoRA 單組，但未完成同 batch／同預算的 frozen control，不能宣稱通過原 Gate 1 的 paired comparison。

Batch 64 vision-only run 完成兩個 epochs：train 各約 `69.9`、`68.1` 分鐘，validation accuracy `0.4282`、`0.4584`，但完整資料吞吐約 `19–20 samples/s`，未重現短測的 `34–36 samples/s`。使用者決定停止並改回 batch 16；batch 64 的兩個 checkpoints 與結果保留，epoch 3 部分進度不保留。新的無 cache、batch 16、vision-only 60-epoch run 於 2026-09-17 22:24（台北時間）在 host `tmux` session `phase3_vision_b16` 啟動，從 pretrained XCLIP 與 seed 42 重新開始；指令與輸出在 `artifacts/skeletonclip_phase3/vision_batch16_nocache_seed42/launch.sh` 和同目錄 `console.log`。此 run 仍缺同設定 frozen control，不能視為原 Gate 1 的 paired comparison。

對當前瓶頸的 24 支影片 CPU profile 顯示每支平均 decode `0.52 s`、processor `0.19 s`。使用者要求改善資料管線後，batch 16／無 cache／相同 RGB 路徑的 DataLoader-only benchmark 比較 16、24、32 workers：正向依序為 `22.45`、`34.50`、`34.98 samples/s`，反向依序為 `22.21`、`34.98`、`39.20 samples/s`；24 workers 明顯優於 16 並留 CPU 餘裕。原 batch 16／16 workers run 於 epoch 1 尚未完成時停止，沒有 checkpoint。新的 batch 16／24 workers vision-only run 於 2026-09-17 22:40（台北時間）在 host `tmux` session `phase3_vision_b16_w24` 啟動，完整指令與 log 在 `artifacts/skeletonclip_phase3/vision_batch16_workers24_nocache_seed42/`。只增加 DataLoader workers，抽幀、processor pixels、模型與 optimizer 設定不變；短測加速需以完整 epoch 實測確認。

新 run 的初期累計吞吐在 4,800 筆時約 `23.05 samples/s`；前一個 16 workers run 在 7,600 筆時約 `19.97 samples/s`。這支持 workers 增加有幫助，但樣本位置與啟動暖機不同，不能當作完整 epoch 的最終加速率。

在不中斷 24-workers 訓練下，以低 CPU／I/O 優先權對 24 支真實影片測試替代資料路徑，原始數據在 `artifacts/skeletonclip_phase3/rgb_io_probe/results.json`。將相同 processor 的 resize／crop 後、normalize 前影像存為 uint8，每支約 `1.204 MB`，約為 FP32 pixels cache 的四分之一；讀回並 normalize 後對 24 支影片的原始 processor pixels 皆為逐值相同，短程熱 cache 命中約 `46.9 samples/s`，但樣本量小且剛寫入，不能代表完整約 `113 GB` 的冷 cache／旋轉式磁碟吞吐。FFmpeg 的 `mpeg4_cuvid` 可在主機上解碼本資料集影片，但此每影片啟動 FFmpeg 的 GPU 抽八幀路徑對 24 支影片約 `0.64 samples/s`，慢於同測試 CPU 線上 decode＋processor 約 `0.79 samples/s`；normalize 後約 `89%` 的 tensor 元素不同、最大差約 `0.315`。因此目前不將此 GPU 路徑替換成正式訓練輸入，也不在尚未通過完整資料吞吐驗證前切換 uint8 cache。

另對同 24 個 uint8 cache 檔先使用 `posix_fadvise(DONTNEED)` 提示系統移出 page cache，再以低 CPU／I/O 優先權逐筆讀取並 normalize，約 `1.59 samples/s`，結果在 `artifacts/skeletonclip_phase3/rgb_io_probe/cold_read_probe.json`。這比同一低優先權初測的單程序線上 CPU 路徑約 `0.79 samples/s` 快，但遠低於熱 cache 的 `46.9 samples/s`；測試期間正式訓練仍持續，絕對速率受並行負載影響。這支持繼續做多 workers 的中型冷讀吞吐測試，但尚不足以預測完整訓練會加速。

依使用者要求，在不停止當時的正式訓練下完成 512 支影片、batch 16、24 workers 的中型 uint8 cache 對照：建置 512 個 cache 檔共 `616.6 MB`，以低 CPU／I/O 優先權耗時 `146.4 s`，每 16 筆抽 1 筆共 32 筆驗證與線上 processor pixels 逐值相同。建置後第一輪線上／cache 吞吐分別為 `16.26`／`24.07 samples/s`，第二輪因來源與系統快取狀態變化，線上／cache 為 `24.49`／`24.64 samples/s`，故第一輪無法單獨證明加速。待建置負載結束後，對兩種來源都使用 `posix_fadvise(DONTNEED)` 提示移出 page cache，交錯兩輪的 cache 為 `37.80`、`30.79 samples/s`，線上為 `16.01`、`13.88 samples/s`；結果在 `artifacts/skeletonclip_phase3/uint8_cache_benchmark/`。這表示小於 1 GB 的受測子集有明確加速潛力，但 fadvise 只是提示、測試和正式訓練共用磁碟且非完整約 113 GB cache，不能保證全量端到端同倍率；在決定切換前，訓練維持原路徑。

其後無 cache、batch 16、24 workers 的 vision run 完成 epoch 1，validation accuracy `0.4344907`，`last_checkpoint.pt` 保存在 `artifacts/skeletonclip_phase3/vision_batch16_workers24_nocache_seed42/vision_lora_ce_seed42/`。使用者決定改用 compact uint8 cache，故在 epoch 2 的部分進度中斷後，從 epoch 1 checkpoint 在新 output `artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/` 接續 epoch 2；模型、optimizer、scheduler、scaler 與 RNG state 均從 checkpoint 恢復，未重訓 epoch 1。新 cache `cache/skeletonclip_phase3_rgb_uint8` 保存每影片 resize／crop 後、rescale／normalize 前的 uint8 RGB，檔案約 `1,204,352 bytes`；四支真實影片的首次建立與命中結果均與原 processor float32 pixels 逐值相同，續跑 identity 僅容許 cache transport 與程式來源版本變更。93,600 支 train／validation 影片的 cache 已建立，暖 cache 的 epoch 4 訓練約 `539 s`、`149 samples/s`，相較舊 run epoch 1 約 `3335 s`；跨 epoch 吞吐仍受主機負載影響。舊 FP32 cache `cache/skeletonclip_phase3_rgb` 已依使用者要求刪除，uint8 cache 保留。

Vision-only run 的完整 validation 最佳成績在 epoch 13：accuracy `0.5891975`、macro-F1 `0.6045542`、NLL `1.3922710`；歷史 XCLIP baseline validation `0.3927469`，調參後的 XCLIP + MIT LoRA validation `0.5212191`，此 vision run 分別高約 `19.65`、`6.80` 個百分點，但模型輸入／訓練路徑不同，舊結果僅作歷史參考。epoch 37 的 train loss 降至 `0.9444`，validation accuracy 回落至 `0.5727624`、NLL 升至 `1.5811`；epoch 13 後未再超越最佳 accuracy，顯示過擬合跡象。使用者於 2026-09-18（台北時間）停止 run；最後完整 checkpoint 是 epoch 37，epoch 38 的部分訓練未保存。最佳 adapter、validation logits、最後 checkpoint、run lock 與進度紀錄保存在上述 uint8 run output，console 記錄位於其上層 `train.log`。此 run 沒有同設定且完成的 frozen control，也沒有評估 test，因此不能宣稱 Phase 3 Gate 1 paired comparison 通過或通用化效果已由 test 證實。

### Gate 2：validation 決策

#### 工作與決策

- 分別報告完整 120 類、A001–A059、A060–A120 及 paired／unpaired subgroups，依 validation 判斷 CE-only vision adaptation 是否帶來可解釋的分類效益；不可只看 aggregate accuracy。
- 若沒有改善，只允許一次在執行前寫明 hypothesis／設定的受控診斷，不自動擴大搜尋或讀 test。若目標只有 accuracy，可在此結束；若目標包含 Phase 4 spatial grounding，通過 Phase 3 Gate 0 的 vision path 仍可用於獨立 Phase 4 control。

#### 預期產物

- Validation-only go／conditional／no-go 決策與 subgroup、成本、限制報告；任何額外診斷的 lock 須在執行前保存。

### Gate 3：有限 class-level KD

#### 工作與 Pass gate

- Phase 2 class-level KD 已 no-go，因此本 Gate 預設不啟動。只有 Phase 3 Gate 2 提出不同於已失敗 Phase 2 設定的明確 representation hypothesis，才在讀取任何 Phase 3 test 前鎖定少量 KD candidates、selection rule 與同架構 CE control。KD 與 CE 共用 RGB dataloader、初始化、training budget、evaluation 與 checkpoint selection；不混入 Phase 4 spatial loss。
- 以 validation locked winner 決定是否 test-once，保留完整 CE／KD artifacts；即使 KD 改善，也只主張 class-level adaptation，不主張 joint focus。

## 3. Phase 4：spatial joint supervision

### 假設

Class logits 只約束動作類別分布，沒有指定模型應該看哪個 region。若研究目標是 joint-focused visual learning，必須將 skeleton joints 與 RGB frame 的空間位置建立可驗證的 supervision contract。

目前有效 skeleton supervision 僅涵蓋 A001–A059 的部分 RGB 樣本，且主要來自 S001–S017；完整 120 類仍以 RGB CE 訓練。Joint supervision 可能學到可遷移的姿態或互動特徵，但不能預設會改善 A060–A120，也可能因過度偏向人體 region 而傷害依賴物件或場景的類別。完整 120 類改善是待驗證的結果，不是此資料 coverage 可保證的效果。若主要目標是可靠改善完整 120 類，應另行評估補齊 S018–S032 skeleton 與有效 coverage 的成本和收益。

### 預定工作

- 先從原始 `.skeleton` 逐幀讀取 joint 的 color-image 2D 座標、tracking state 與 body ID，保留有效性 mask；現有 `load_skeleton_tensor()` 只保留 root-centered XYZ 並重採樣至 64 frames，不能直接作為 RGB heatmap 來源。
- 記錄八張 RGB 實際解碼的 frame indices／時間資訊，核對 skeleton 與 RGB 的 frame count、時間偏差及 body tracking；將原始 color-image 座標依相同的 XCLIPProcessor resize／crop 幾何轉到模型輸入，再映射至 patch grid。不可直接假設兩邊的同序號 frame 或直接使用 3D XYZ 作像素座標。
- 先抽取分層樣本，將有效 joints 畫回實際模型輸入影像，人工核查單人／多人、遮擋、邊界與不同 setup／camera；保存逐 sample audit、排除原因、各 class／setup／split 的可用 coverage。對齊不可靠時先修資料 contract，不進入 spatial training。
- 保留 vision patch tokens，明確指定 spatial loss 的梯度須能更新哪些 vision upper blocks／LoRA modules；若只更新輔助 head，不能據此主張分類器的 vision features 已學到 joint focus。Phase 4 可重用 Phase 3 的 vision-forward／LoRA 技術檢查，但不以 Phase 3 accuracy 改善為前提。
- 第一輪只鎖定一種 patch-level joint heatmap／region-mask supervision，依 patch32 的粗空間解析度、joint visibility 與有效面積定義 target 和 loss；joint-to-patch feature alignment 與 attention alignment 延後為獨立 ablation，需各自定義 feature-space／attention 解讀 contract。
- 對照組使用相同 RGB forward、vision LoRA、MIT／head、初始化、split、seed、training budget、checkpoint selection 與 evaluation，僅將 spatial loss weight 設為 0；另加 shuffled-joint control 與簡單的中心位置／人體區域 baseline，避免把 dataset center bias 誤判為 joint grounding。
- 分開評估完整 120 類、A001–A059、A060–A120，以及 paired／unpaired 的 classification metrics；spatial evidence 使用 held-out paired validation 的 joint-region localization，並以 masking／occlusion 或 attribution consistency 作輔助診斷。若只改善 A001–A059，結論限定為受監督類別受益，不稱為完整 120-class 任務改善。
- 明確區分 training-only skeleton supervision 與 deployment input；若 inference 仍 RGB-only，部署 artifact 不得依賴 skeleton runtime。

### Phase 4 kickoff gate

- Gate 0：先完成小規模、跨 class／setup／camera 的原始 joint-to-RGB 可視化 audit，再完成逐 sample 的時間、座標轉換、tracking／visibility 與可用 coverage 統計；在看結果前鎖定可接受的對齊誤差及有效 coverage 門檻。未通過時不啟動訓練。
- Gate 1：patch-token forward、spatial target／loss、vision LoRA gradient path、gradient isolation、checkpoint round trip 及 RGB-only reload 通過；確認 spatial loss 確實更新預定的 vision representation。
- Gate 2：在固定架構與 CE-only control 上完成一種最小 heatmap／region-mask ablation；localization 必須優於 CE control 與中心 baseline，shuffled-joint control 不應有相同提升。若缺乏可重現的 spatial signal，停止擴大實驗。
- Gate 3：只有在 spatial evidence 可重現時，才進行有限的 loss／weight search，並依預先鎖定的 validation 決策規則選 winner、執行一次 test；完整 120 類 accuracy 與各 coverage subgroup 分開判讀。

#### Gate 2 啟動鎖（2026-09-18）

`vlm train skeleton-phase4-gate2` 已完成資料與初始化 preflight，產生 `artifacts/skeletonclip_phase4/gate2/experiment_lock.json`。先前 sandbox 看不到 GPU；解除 sandbox 後確認本機 RTX 2080 Ti 可用。真實模型 GPU Gate 1 resource smoke 通過，峰值 allocated VRAM 約 0.56 GiB；Gate 2 三組各 2 個 training batches 的 GPU smoke 也完成，batch 16 峰值約 1.1 GiB，結果位於 `artifacts/skeletonclip_phase4/gate2_smoke/`。正式三組訓練於 2026-09-18 10:52 台北時間在 `tmux` session `phase4_gate2` 啟動並完成，輸出目錄為 `artifacts/skeletonclip_phase4/gate2/`。選用 Phase 3 vision-only 最佳 adapter 作三組共同起點，固定 seed 42、batch 16、20 epochs、spatial weight 0.1；CE、joint、shuffled-joint 共享 RGB split、模型與訓練預算。20 epochs 是首輪受控訊號檢查的固定預算，不代表已證實足夠收斂；0.1 是首輪單一權重假設，Gate 2 不搜尋權重。CE arm 的 spatial probe 從 detached patch tokens 學習，以相同 target 和 loss 建立 localization 參考，但 probe 梯度不進入 vision；joint arm 讓 spatial 梯度進入 vision，shuffled-joint arm 只置換 train paired target。Validation localization 使用 held-out paired validation 有效 frame 的 patch AP 與依 target occupied count 的 top-k recall，並與 CE probe、中心 prior 比較；test embargo 維持。完整結果與決策見 `docs/skeletonclip-phase4-gate2-results.md`。

#### Gate 0 首輪執行（2026-09-18）

已新增 `vlm data skeleton-phase4-gate0`，從原始 `.skeleton` 讀取 color-image `colorX/colorY`、joint tracking state、body ID，使用與 Phase 3 相同的 PyAV whole-video 八幀 indices，記錄實際 decoded PTS／時間，依鎖定的 XCLIP shortest-edge resize 與 center crop 幾何轉成 224×224 輸入座標並輸出 overlay、逐 sample JSONL 和分組摘要。執行前門檻與抽樣規則保存在 `docs/skeletonclip-phase4-gate0-lock.json`；執行產物在 `artifacts/skeletonclip_phase4/gate0/`，沒有讀取 test。

首輪從 paired manifest 抽取 train 24、validation 12 筆，涵蓋 30 類、S001–S017 與 C001–C003。36 筆的 RGB／skeleton frame count 均相同；288 張 frame 的 decoded timestamp 與以 FPS 推得的時間一致；每張至少 11 個 tracked joints 落在 processor crop 內，36／36 筆達到預先鎖定的自動 coverage 門檻。人工抽看單人與雙人 overlay，點位在人體附近；這項目視結果尚未量出 16 px 中位誤差，也無法獨立證明沒有固定 frame offset。原始 NTU skeleton 檔缺逐幀 timestamp，目前只能依 frame index 和影像內容核對時間對齊。

全量 train／validation paired manifest 的 49,416 筆已完成不解碼影像的 metadata／原始 joint coverage 掃描；逐筆結果在 `full_coverage.jsonl`，分組摘要在 `full_coverage_report.json`。全部影片與 skeleton frame count 相同，沒有解析錯誤。依預先鎖定的「八幀中至少 75% 的 frame 各有至少 10 個 crop 內 tracked joints」條件，49,171／49,416 筆（99.50%）可用，245 筆因 joint coverage 不足排除；train 為 45,007／45,236（99.49%），validation 為 4,164／4,180（99.62%）。A059 明顯較低，只有 597／734（81.34%）；A055 為 760／795（95.60%），其他類別最低為 A043 的 823／835（98.56%）。全部 36 筆可視化樣本的 tracked-joint 計數與全量掃描逐筆一致。這個全量統計只檢查 frame index、座標與 tracking 可用性，沒有量到影像上的 joint 誤差。

Gate 0 的目視與量化審核現已完成。從 36 筆分層樣本中選取 12 筆中段 frame，另加入 4 筆 A059（兩筆自動可用、兩筆排除），在未畫點的 224×224 processor 影像上目視標註 62 個可見 head／ankle／hand landmarks；依鎖定座標轉換，誤差中位數 `2.45 px`、最大 `7.23 px`，低於事前固定的 `16 px` 中位數門檻。這些標註是同一 agent 的近似、非盲測目視估計，故不視為獨立標註資料。16 組 RGB `t−3/t/t+3` 面板均已目視檢查；對其中有至少 8 px 動作變化且七個相鄰 skeleton frames 皆 tracked 的 8 個 landmarks，將 skeleton `t−3…t+3` 與 RGB `t` 的標註點比較，`t=0` 的中位誤差 `2.93 px` 最低，`t−1`／`t+1` 分別為 `5.79`／`5.58 px`，沒有觀察到一致的 signed frame offset。4 筆 A059 的八幀 overlay 也已目視檢查；兩筆排除案例的部分 frame 是人物尚未進入或已離開有效 crop，而有人物的 frame 仍可對齊。標註、目視紀錄及機器產生的詳細結果分別在 `docs/skeletonclip-phase4-gate0-landmarks.json`、`docs/skeletonclip-phase4-gate0-visual-review.json`、`artifacts/skeletonclip_phase4/gate0/manual_alignment_report.json`。

**Phase 4 Gate 0：pass。** `artifacts/skeletonclip_phase4/gate0/gate0_decision.json` 對鎖定門檻、來源 hashes、sample／full coverage、人工誤差、時間偏移與 test embargo 逐項核對並記錄 pass。這項結論僅適用 NTU60 A001–A059 的 paired train／validation，以及本次小規模影像審核；原始 skeleton 沒有獨立逐幀 timestamp，不能據此聲稱所有樣本的每個 joint 都有同等精度。A059 可用率 `81.34%`，後續 spatial supervision 須保留逐 frame／joint mask，不得把排除 frame 當作零座標目標。Gate 1 可開始 patch-token／spatial-loss 相容性 smoke，但尚未啟動分類或 spatial training。

#### Gate 1 真實模型功能 smoke（2026-09-18）

新增 `vlm train skeleton-phase4-gate1`、`XCLIPSpatialLoRAClassifier` 與原始 color-joint 到 patch32 occupancy target。分類路徑沿用 Phase 3 的 RGB vision／MIT／120-class head；training-only `spatial_head` 讀取每張 vision `last_hidden_state` 的 49 個 patch tokens，輸出 `[batch, 8, 49]` logits。target 只使用兩個主要 body IDs 中 tracking state 2 且落在 224×224 crop 內的 joints；每 frame 至少 10 個有效 joints 才施加 balanced patch BCE，無效 frame／unpaired sample 的 spatial loss 為零。Gate 2 若將 spatial weight 設為 0，CE-only 路徑不對 spatial head 反向傳播。

在鎖定 XCLIP revision、真實 train A059 雙人影片與原始 `.skeleton` 上，processor pixels 為 `[1, 8, 3, 224, 224]`、patch tokens 為 `[1, 8, 49, 768]`、分類 logits 為 `[1, 120]`；八幀均有 spatial target，共 335 個有效 joints。spatial-only loss 的非零梯度進入 vision blocks 10／11 的 q/v LoRA 與 spatial head，不進入 MIT／classifier／frozen base；一步 optimizer 更新後，patch tokens 最大變化 `0.02984`、RGB 分類 logits 最大變化 `0.00325`。CE-only backward 更新 vision／MIT／classifier，但 spatial head 無梯度。trainable parameters 為 194,425；adapter strict reload 的分類與 spatial logits 差異皆為 `0`，FP32 merge 與 merged RGB-only strict reload 的最大差異分別不超過 `6.26e-7`。`artifacts/skeletonclip_phase4/gate1/` 保存 experiment lock、checkpoint 與完整報告；`test_evaluated=false`。

**Phase 4 Gate 1 功能契約：pass。** 最初 sandbox 看不到 NVIDIA GPU，因此先用真實 pretrained XCLIP 在 CPU 完成單支影片 smoke；其後在主機 GPU 上重新跑 resource smoke 並通過，peak allocated VRAM 約 0.56 GiB。Gate 1 本身不包含分類品質或 localization 改善結論；正式 Gate 2 結果另見 `docs/skeletonclip-phase4-gate2-results.md`。

時程校正：Gate 1 沿用 Phase 3 vision LoRA、RGB forward 與 checkpoint 路徑，加上 Gate 0 已完成的 joint 幾何，實際在本次工作中不到一小時完成；先前以數天估計 Gate 1，是把可能的整合問題當作預期工期。Phase 3 uint8 cache run 的完整訓練加 validation 在連續 epoch 5–37 的中位數約 `13.13 分鐘／epoch`，範圍約 `10.85–15.25 分鐘`；早期未加速 RGB path 約一小時／epoch，不能直接推估 Gate 2。Gate 2 啟動前另行固定三組各 20 epochs，完整 run 的穩態 epoch 約 9–10 分鐘；第一組前兩個 epoch 含 cache／系統暖機，較慢。

## 4. Phase 3 對 Phase 4 的決策規則

Phase 3 是 Phase 4 的風險指標，不是硬性前置條件。Phase 3 accuracy no-go 不會自動取消 Phase 4，因為 spatial supervision 驗證的是不同假設。

- 若專案目標只有分類 accuracy，且 Phase 3 vision adaptation 也 no-go，Phase 4 預設不啟動。
- 若專案目標包含 joint grounding、可解釋性或 spatial evidence，即使 Phase 3 accuracy 沒改善，Phase 4 仍可在通過對齊與可觀測性 gates 後啟動。
- 若 Phase 4 Gate 0 發現 RGB/skeleton 時間或空間對齊不可靠，先修 provenance 與 alignment；在修正前不得啟動 Phase 4 training。
- Phase 4 的任何改善都不能回填成 Phase 2 KD 或 Phase 3 LoRA 的效果；每個 phase 必須保留自己的 CE control、lock、artifacts 與 test embargo。

## 5. 目前決策

Phase 3 Gate 0 已通過；短期 warm-start pilot 的 vision LoRA validation 比 frozen 高 `1.5355` 個百分點。正式 vision-only CE run 在 epoch 37 後停止，最佳 epoch 13 validation accuracy `58.92%`，高於歷史調參後 XCLIP + MIT LoRA 的 `52.12%`；後期 validation 退步且仍缺同設定完整 frozen control，Gate 1 paired comparison 尚未通過。Phase 3 Gate 2 的 subgroup 分析與 validation 決策尚未完成，Gate 3 class-level KD 預設不啟動，test embargo 維持。Phase 4 Gate 0／1 已通過；Gate 2 三組各 20 epochs 已完成。Joint 最終 patch AP `0.956869`，CE `0.954959`，shuffled-joint `0.759423`，方向性比較成立但 joint 對 CE 僅高 `0.001911`；120-class accuracy 各組約 `58.5–58.6%`。單一 seed、未估計樣本層級不確定性，且缺分類 subgroup logits，尚不足以支持可重現改善或完整分類解讀，故 Phase 4 Gate 3 暫不啟動；不讀 test。

## 6. 跨 session 續接

工作目錄為 `/home/fallingapart/VLM`。Pilot 已完成；歷史指令為 `uv run vlm train skeleton-phase3-gate1 --arm both --pilot-fraction 0.1 --epochs 8 --validate-every 8 --init-adapter artifacts/skeletonclip_phase2/control_seed42/xclip_lora_classifier.pt --mit-lr 2e-5 --head-lr 8e-5 --vision-lr 1e-4 --workers 16 --batch-size 16 --artifacts-dir artifacts/skeletonclip_phase3/pilot_seed42`，產物在 `artifacts/skeletonclip_phase3/pilot_seed42/`。

原雙組 FP32 cached run 的 host `tmux` session `phase3_gate1_full` 已停止；紀錄在 `artifacts/skeletonclip_phase3/gate1_full_cached_seed42/console.log`、frozen arm `progress.json` 和 `last_checkpoint.pt`。`launch.sh` 保存原啟動參數；其超過 400 GB 的 `cache/skeletonclip_phase3_rgb` 已刪除，重啟該設定將重新建 cache。舊 `artifacts/skeletonclip_phase3/gate1/` 是更早的中斷紀錄。

目前沒有 Phase 3 訓練程序；最後的 host `tmux` session `phase3_vision_b16_w24_uint8` 已於 epoch 38 中途停止。最佳 adapter 在 `artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt`，最後完整 checkpoint 為同目錄 `last_checkpoint.pt`，epoch 指標可由 checkpoint history 或上層 `train.log` 重建。新 run 的 `run_lock.json` 記錄舊 epoch 1 checkpoint SHA256、cache 路徑與設定；`cache/skeletonclip_phase3_rgb_uint8` 仍保留。不要把此 vision-only 結果誤認為已具備 frozen control 的完整 Gate 1。

最新 vision-only run 是使用者於 2026-09-18 明確要求停止的，不應在新 session 自動重啟。`last_checkpoint.pt` 可從 epoch 38 重啟，但因 epoch 13 後持續沒有 validation 新高且 NLL 上升，目前沒有續跑到 epoch 60 的決策。中途停止不會產生最終 `run_report.json`；以 `best_adapter.pt`、`best_validation_logits.pt`、`last_checkpoint.pt`、`run_lock.json` 和上層 `train.log` 為現有依據，`progress.json` 停在未完成的 epoch 38，不能當作完成記錄。需要重建歷史時，checkpoint history 包含 epoch 1–37，其中 epoch 1 來自舊無 cache run；cache handoff 的原始 checkpoint 位於 `artifacts/skeletonclip_phase3/vision_batch16_workers24_nocache_seed42/vision_lora_ce_seed42/last_checkpoint.pt`。

下一個獨立工作是 Phase 3 Gate 2：從最佳 epoch 的完整 validation logits 計算 120 類、A001–A059、A060–A120、paired／unpaired subgroup metrics，並與歷史 XCLIP／LoRA 結果及 pilot 對照，明確標註訓練與輸入路徑差異。若要正式回答 vision adaptation 相對 frozen 的效果，還需另行決定是否以相同 RGB／uint8 cache、初始化、seed、batch 16、24 workers、訓練預算及 selection rule 執行完整 frozen control；目前沒有該對照，也沒有 test-once 決策，勿讀 Phase 3 test。Phase 3 Gate 3 是可選的 class-level KD，並非 Phase 4 的前置步驟。

Phase 4 Gate 0／1 與 Gate 2 三組訓練均已完成。完整 Gate 2 比較在 `docs/skeletonclip-phase4-gate2-results.md`、`artifacts/skeletonclip_phase4/gate2/gate2_decision.json` 與各 arm `run_report.json`；實驗鎖、啟動指令、checkpoint 和追加式 `train.log` 仍在同一 artifact 目錄。Gate 2 沒有產生 Gate 3 的啟動決策，後續若要研究 spatial signal，需先另行鎖定可重現性與最小有效差距；目前不自動啟動其他訓練或讀 test。Phase 4 推論仍只需 RGB，skeleton 僅為 training supervision。

Phase 3 入口為 `vlm train skeleton-phase3-gate1`，Phase 4 Gate 0／1 入口分別為 `vlm data skeleton-phase4-gate0` 與 `vlm train skeleton-phase4-gate1`。主要 Phase 4 實作在 `src/vlm/phase4_alignment.py`、`src/vlm/spatial_joint_targets.py`、`src/vlm/models/xclip_vision_lora.py`、`src/vlm/training/skeletonclip_phase4_gate1.py`；Gate 0／1 目前仍在未提交工作樹，接手時先執行 `git status --short`，不要 reset／clean 這些檔案。Gate 1 使用 CPU 真實模型 smoke，沒有可見 CUDA driver；完整 Gate 2 training 與新 spatial target pipeline 的長程吞吐仍未驗證。
