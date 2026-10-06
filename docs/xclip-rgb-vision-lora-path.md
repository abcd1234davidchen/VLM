# XCLIP RGB 路徑與下一輪 vision LoRA 計畫

日期：2026-10-02。本文整理目前 84.2245% 候選的權重來源、調參歷史與評估範圍，並記錄下一輪擴大 vision LoRA 的設計。最後四層 LoRA 的嚴格初始化、GPU smoke 與評估 runner 已實作；正式單 pass pilot 與完整 retained86 validation 已完成：四層為 84.5486%，相較兩層 84.2245% 淨增28筆，屬有限收益。

## 1. 從原始 XCLIP 到目前候選

原始模型為 `microsoft/xclip-base-patch32`，revision `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`。每支影片從完整時間範圍均勻抽八個真實 RGB frames，模型輸入為 224×224，patch size 32。這是已預訓練的 XCLIP，不是從隨機權重開始訓練。

| 階段 | 權重或配置來源 | 實際做法 | 已確認結果 |
|---|---|---|---|
| 歷史 MIT LoRA 調參 | 預訓練 XCLIP | Vision 凍結，訓練 MIT q/v LoRA 與分類頭；35 個 completed trials | 最佳配置的正式 run：120 類 validation 52.1219%，best epoch49 |
| Phase 3 vision adaptation | 重新從預訓練 XCLIP 開始；沿用調參配置，`init_adapter=null` | Vision 最後兩層 q/v LoRA、MIT q/v LoRA、120 類分類頭；80,640 train、batch16 | 選中 epoch13，120 類 validation 58.9198% |
| Whole＋zoom | Phase 3 epoch13 checkpoint | 原畫面與 center zoom70 各 forward，softmax 機率等權平均；沒有更新權重 | 完整 120 類 validation 66.94%；保留 86 類範圍 79.9769% |
| RGB interaction local expert | 複製 Phase 3 epoch13 checkpoint | RGB 定位手與頭／肩互動區域，局部影像另訓練 2,280 筆、單 pass、batch4、570 steps | 同批診斷中，相對未訓練局部融合，保留類別額外淨增 21 筆 |
| 固定三視角融合 | 原 Phase 3 whole／zoom，加獨立 local expert | Whole、zoom、local 的 120 類 softmax 機率各占三分之一 | 完整保留 86 類 validation：7,277／8,640，84.2245% |

MIT LoRA 52.12% 的 checkpoint 沒有直接接續成為本候選的 Phase 3 權重。`artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/run_lock.json` 記錄 `init_adapter_sha256=null`；cache handoff 只是同一訓練從 epoch1 checkpoint 改換讀取方式後續跑，不是另一輪模型適應。

## 2. 目前實際權重路徑的更新量

- Whole／zoom：Phase 3 的 epoch13 checkpoint；每 epoch 為 80,640／16＝5,040 optimizer steps，選中權重累積 65,520 steps。
- Local expert：從同一 epoch13 checkpoint 另開分支，增加 2,280／4＝570 steps，權重路徑合計 66,090 steps。
- Phase 3 實驗跑完 epoch37，在 epoch38 中途停止；epoch13 之後的更新沒有納入選中的 baseline。完整實驗消耗與選中 checkpoint 的累積更新量不同。
- Local pilot 前有兩次功能 smoke，各四個 steps；兩次更新皆未接續到正式 pilot，正式 pilot 重新從 Phase 3 baseline 開始。
- 全量 8,640 支評估沒有訓練、沒有搜尋 crop 大小或融合權重，也沒有再挑另一個 local epoch。

35 個歷史 tuning trials 不等於目前模型順序接續了 35 次訓練。它們提供配置選擇，並非全部權重更新累積到目前 checkpoint；這也不代表整個 repository 歷來只做過 35 個實驗。

## 3. 推理與訓練需要的資料

推理只需 RGB 影片。流程為抽八幀，再建立 whole、中央 zoom70、interaction crop 三種輸入。MediaPipe PoseLandmarker lite 從 RGB 偵測人體 landmarks；手部 landmarks 加頭／肩 context 形成 clip-static 正方形 crop，再縮放至 224×224。少於四幀有效手部定位或缺 torso scale 時，local 輸入回退 whole frames；所有影片照常計分。

目前採用的 baseline 與 local expert 訓練使用 RGB 與動作類別標籤。RGB local expert 不讀 `.skeleton`，也不使用額外 attention loss。早期 privileged skeleton crop 僅是可行性診斷，後續 RGB 定位取代其位置來源。

先前 skeleton KD、spatial supervision、native attention supervision 與 crop-aware pilot 都是獨立研究分支；其更新沒有接入目前 84.2245% 候選。

## 4. 評估範圍必須保持一致

120 類完整 validation 有 12,960 支；使用者指定排除 34 類後，保留 86 類的 ground-truth validation 有 8,640 支。分類器仍輸出全部 120 logits，保留樣本被預測成排除類別仍算錯，沒有依 ground truth 選擇視角或遮蔽輸出。

58.9198%／66.94% 是 120 類範圍；79.9769%／84.2245% 是保留 86 類範圍。改變範圍本身造成的分數差異不能當成模型提升。可直接比較的最新收益為同樣 8,640 支上的 79.9769%→84.2245%，即 +4.2477 個百分點。

| 比較 | 原 whole＋zoom | 加入 trained local | 淨變動 |
|---|---:|---:|---:|
| 完整 retained86，n=8,640 | 6,910，79.9769% | 7,277，84.2245% | +367 |
| 不在先前診斷中的 retained86，n=7,548 | 6,167，81.7038% | 6,450，85.4531% | +283 |
| A042–A048，n=504 | 349，69.2460% | 371，73.6111% | +22 |
| A047 摸頸，n=72 | 24 | 20 | −4 |

全量 retained86 修正 505 筆、退步 138 筆；69 類改善、11 類持平、6 類退步。資料仍為開發 validation；先前診斷以外的 7,548 支不是獨立 test。完整資料尚未評估未訓練 local ensemble，因此不能把全量 +367 全部歸因於局部訓練本身。

## 5. 下一輪假設：擴大 local expert 的 vision LoRA

目前 vision 有 12 個 Transformer blocks，只有最後兩個 blocks（zero-based 10、11）的 `q_proj`／`v_proj` 使用 rank16、alpha32、dropout0.05 LoRA。下一個優先候選為最後四個 blocks（8–11），新增 blocks8、9 的 q/v LoRA，rank 與其他設定先保持不變。

假設是：局部裁切已提供更清楚的手與接觸區域，讓更多 vision blocks 適應這些 RGB inputs，可能增加分類收益。這仍需實驗；它不提高 patch 解析度、不增加八幀的時間資訊，也不保證解決摸頸。原 Phase 3 在 epoch13 之後出現 validation 退步與 NLL 上升，不能據更多容量推定更好的泛化。

CLIP 模組選擇研究支持檢查哪些部分適合調整，但沒有證明本專案最後四層一定優於兩層：[Adaptive Parameter Selection for Tuning Vision-Language Models，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Adaptive_Parameter_Selection_for_Tuning_Vision-Language_Models_CVPR_2025_paper.pdf)。先試 block 深度；rank16→32、擴展其他 projections 或 MLP 留待這輪結果後再決定。

## 6. 初始化與相同比較預算

下一輪從 Phase 3 epoch13 baseline 初始化 local expert，保持與既有兩層 local pilot 相同的起點與 570-step 訓練預算；不從已訓練 570 steps 的 local expert 再接續一輪，避免把額外訓練量混入 block 深度比較。

- 保留 baseline blocks10、11 的 LoRA、MIT LoRA 與 classifier 權重；新增 blocks8、9 的 LoRA B 初始化為零，確保 eval 時初始函數與 baseline 一致。
- 既有 loader 要求 checkpoint `target_modules` 完全相等，因此不能直接把兩層 checkpoint 交給四層 loader；需明確映射既有 LoRA keys，只允許新增 blocks8、9 的 adapter keys 沒有來源，其他權重缺失或不匹配一律報錯。
- 不直接重新呼叫現有 injector 包覆已存在的 LoRA 層；其接口只接受 `nn.Linear`。實作應在四層模型中載入已存在的兩層權重，或只對原本沒有 adapter 的新增 blocks 做注入。
- 在 eval mode 核對 baseline whole logits 的 parity、新增 adapters 的零增量與 target-module 清單；維持只有 LoRA 與 classifier 可訓練。依現有慣例先做短 smoke，確認新增 blocks 的 gradients、checkpoint reload 與 merged RGB-only parity。

## 7. 訓練與評估規格

- Train：沿用 `artifacts/hand_roi_followup_v1/train_selection.json` 的 2,280 支與 `train_localization/crop_manifest.json`，保持 train／validation IDs 不交集；11 個 focus classes 各128支，其他109類各8支，仍包含排除類別。
- Recipe：seed42、batch4、單 pass／570 steps、AdamW lr1e-5、weight_decay5e-4、cosine、CE label_smoothing0.1、gradient_clip1；沿用 dropout／feature noise。使用相同 crop geometry 與 RGB detector，不同時增加 rank、修改輸入或融合權重。
- Whole／zoom：繼續使用 Phase 3 epoch13 checkpoint 與既有 canonical validation predictions，權重不更新。
- Local：只新增四層版本的獨立 expert；保存新的 source snapshots、init adapter hash、target-module 清單、selection／crop hashes、實際 step 數與進度，使用 fresh artifacts directory。
- Evaluation：先沿用固定 1,228 支診斷確認執行與方向，然後對完整 retained86 的 8,640 支評估；固定三視角等權機率平均，所有120 outputs與fallback均保留，不依 validation 調 crop 或權重，不讀 test。
- 比較：以現有兩層 local ensemble 的 7,277／8,640＝84.2245% 為主要基準，逐筆計算修正／退步；同時列出對原 whole＋zoom 的差異、各86類、A042–A048、三個目標類、摸頸，以及先前診斷以外的7,548支。
- 執行安排：訓練前核對 GPU 使用狀態，保持既有工作運作；batch／workers 上限沿用四。重用已驗證的 RGB localization metadata，避免重跑全量定位。

這輪優先重用已有兩層 pilot 作比較，不要求另跑控制組。如果後續改了資料次序、recipe、初始化或其他實質條件，應明確記錄比較差異，不能將全部變動解釋成增加 vision blocks 的效果。

## 8. 結果判斷

完整 retained86 正確數高於7,277才記為此配置的整體提升，同時列出退步類別與A042–A048結果，不以單一小樣本新高決定採用。若整體下降或沒有額外收益，保留目前兩層候選並停止延長這個設定；若收益很小，記為有限收益，視下一輪用途再決定是否重複seed確認。

即使四層候選改善，也不自動替換部署模型。需另行量測 RGB 定位與三次 forward 的實際成本，並保持目前 threshold／警報流程的決策獨立於這份分類實驗。

## 9. 來源與產物

- [Phase 3 訓練歷史](skeletonclip-phase3-4-roadmap.md)：epoch13 baseline、訓練初始化與停止紀錄。
- [產品評估範圍](product-evaluation-scope.md)：保留86類、排除34類與計分規則。
- [RGB 定位可行性](rgb-hand-roi-v1-results.md)：176支固定診斷與RGB-only定位。
- [局部 expert 短訓練](hand-roi-followup-v1.md)：1,228支診斷、2,280筆train、570steps與smoke。
- [完整 86 類結果](hand-roi-full86-v1-results.md)：84.2245%、逐類分數與新增樣本比較。
- Baseline：`artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_adapter.pt`。
- 現有 local expert：`artifacts/hand_roi_followup_v1/pilot/local_adapter.pt`；merged state：同目錄 `merged_local_rgb_state.pt`。
- 全量 predictions／verification：`artifacts/hand_roi_full86_v1/evaluation/`。`artifacts/`與RGB datasets不納入Git，文件與runner是repository中的重現入口。

## 10. 四層實作與重現（2026-10-02）

`load_expanded_lora_state_dict` 明確核對 source／新增 targets、完整 keys、tensor shapes 與有限值，只允許新增 blocks8／9 的零 B adapters；其餘 baseline adapter 與 classifier 權重完整保留。原 strict loader 仍要求完整 target keys。CPU 臨時檢查通過 exact warm-start parity、缺 key／多 key／shape 錯誤拒絕、新增 q/v gradients、trainable whitelist 與 merged strict reload；既有 14 項 LoRA tests 通過，未新增 persistent tests。

GPU smoke 使用 16 train／16 validation、四 steps，`status=smoke_pass`；whole warm-start max error 0，merged RGB-only strict reload max error `7.5101852e-06`，新增 blocks8／9 的四個 q/v gradients 全非零。Smoke 更新不接續到 pilot；pilot 重新從 Phase 3 baseline 開始。Artifacts：`artifacts/hand_roi_upper4_v1/smoke/`。

已核對本輪與既有 upper2 pilot 的 baseline adapter、ordered train IDs、train／validation selection、crop、reference 與 split hashes 相同。保持 seed42 與相同 DataLoader shuffle；新增 adapters 的初始化及 dropout 會消耗額外 RNG，因此兩輪的隨機 dropout／feature-noise realization 不會完全相同，不能當成逐步完全配對的隨機控制。這項差異記於 experiment lock。

一鍵重跑需使用 fresh root，依序執行 smoke、570-step pilot 與完整 retained86 evaluation；任何階段失敗會停止並留下 stage log，不會跳過 smoke 或接續其權重。全量定位直接重用既有 RGB crop manifest，完成後寫出獨立 `docs/hand-roi-upper4-v1-results.md`，保留 upper2 歷史報告。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_upper4 --artifacts-dir artifacts/hand_roi_upper4_repeat
```

本次分階段執行使用以下命令；每個目錄必須 fresh，pilot 要求同版 runner／LoRA／model 與相同初始化來源的成功 smoke。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_followup --mode smoke --upper-blocks 4 --artifacts-dir artifacts/hand_roi_upper4_v1/smoke
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_followup --mode pilot --upper-blocks 4 --smoke-report artifacts/hand_roi_upper4_v1/smoke/report.json --artifacts-dir artifacts/hand_roi_upper4_v1/pilot
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m vlm.training.hand_roi_full_validation --upper-blocks 4 --local-adapter artifacts/hand_roi_upper4_v1/pilot/local_adapter.pt --artifacts-dir artifacts/hand_roi_upper4_v1/evaluation
```

### Pilot 完成與完整評估

正式 pilot 已完成 2,280 train／570 steps／單 pass，whole warm-start max error 0，merged strict reload max error `4.0531158e-06`。固定 1,228 支診斷中的 retained86（n=1,092）對既有兩層 ensemble 為 827→840，修正15／退步2，net +13；三個目標類（n=216）128→131，net +3；A042–A048（n=164）86→92，修正7／退步1；摸頸（n=72）20→22，仍低於原 whole＋zoom 的24。這是加密診斷選樣，不當成 full retained86 accuracy。

完整 8,640 支 evaluation 已在 tmux session `roi-upper4` 完成，退出碼0，重用既有 full86 RGB crop manifest。產物為 `artifacts/hand_roi_upper4_v1/evaluation/`，log 為同 root 的 `evaluation.log`；`evaluation_exit_code.txt` 記錄程序結束碼。完整逐類結果見 [upper4 評估報告](hand-roi-upper4-v1-results.md)。

## 11. 完整評估與判讀

| 範圍 | 兩層 local ensemble | 四層 local ensemble | 修正／退步 | 淨增 |
|---|---:|---:|---:|---:|
| retained86，n=8,640 | 7,277，84.2245% | 7,305，84.5486% | 48／20 | +28 |
| 先前診斷之外，n=7,548 | 6,450，85.4531% | 6,465，85.6518% | 33／18 | +15 |
| 三個目標類，n=216 | 128 | 131 | 3／0 | +3 |
| A042–A048，n=504 | 371 | 372 | 7／6 | +1 |
| A002 吃東西，n=72 | 45 | 48 | 3／0 | +3 |
| A041 咳嗽，n=72 | 49 | 49 | 0／0 | 0 |
| A049 搧風，n=72 | 34 | 34 | 0／0 | 0 |
| A047 摸頸，n=72 | 20 | 22 | 2／0 | +2 |

四層配置通過原先「完整正確數高於7,277」的整體提升判準，但增幅僅 +0.3241 percentage points，記為單 seed 的有限收益；不自動延長訓練、追加 seed 或替換部署模型。相對原 whole＋zoom，6,910→7,305／8,640，修正532／退步137，net +395（+4.5718 points）；其中大部分收益已由兩層 local 候選提供，這輪四層額外收益為28筆。

摸頸雖20→22，仍低於原 whole＋zoom 的24；A042–A048 整體只多一筆，A042、A043各−1、A048−2，不宣稱警報相關動作普遍改善。其他相對兩層的退步類別為A006、A025、A026、A063、A068、A102、A112、A120（各−1）。RGB定位與三次forward的部署成本仍屬採用前另外評估的工作。

所有197支 RGB localization fallback 均計分，checkpoint／cached logits parity max error 0；來源 hashes 在結束時再次核對。獨立 requirement audit 從保存 predictions 重算總分、修正／退步、全部86類與7,548筆診斷之外的範圍，並核對固定三視角平均、120 outputs、資料分布與初始化來源；結果在 `artifacts/hand_roi_upper4_v1/requirement_audit.json`，可用同 root 的 `requirement_audit_snapshot.py` 重跑。沒有讀test、調crop／融合權重或更新whole／zoom baseline。
