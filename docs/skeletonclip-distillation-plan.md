# SkeletonCLIP teacher distillation 計畫

日期：2026-09-11。狀態：資料盤點與設計；teacher architecture、checkpoint 尚未定案，本文件不代表 training pipeline 已實作。

## 1. 目標與建議

在現有 tuned XCLIP MIT LoRA + CE baseline 加入 skeleton teacher supervision，推理維持八幀 RGB-only。第一版建議採 frozen SkeletonCLIP teacher、離線 teacher logits cache，以及只對有效 RGB–skeleton 配對樣本計算的 KL loss。先保留 120-class RGB 任務，不因 skeleton 缺漏刪除 RGB training samples。

第一版的研究問題是「部分 skeleton teacher supervision 是否改善 120-class RGB classification」，定位為 **skeleton-informed temporal／decision adaptation**，不是 joint-focused visual learning。現有 student 使用 frozen XCLIP vision encoder 產生已 pooled 的 `[8, 512]` frame vectors，KD 只能更新 MIT LoRA 與 classifier；student 沒有 joint coordinates 或 spatial patch supervision，因此即使 accuracy 改善，也不能據此宣稱 vision encoder 的注意力轉向人體關節。

Teacher 優先評估 Hyperformer backbone + frozen CLIP text encoder 的 SkeletonCLIP 路線，但應先確認 checkpoint 的訓練資料與 preprocessing，再決定使用外部初始化或自行訓練。現有資料足以開始 60-class teacher pilot；無法提供完整 NTU120 配對監督。

## 2. 本地資料盤點

來源為 `RGB_vids/專題啊/nturgbd_skeletons_s001_to_s017.zip`。已讀取 ZIP central directory：共 56,880 個 `.skeleton` entries，S001–S017，A001–A060，未壓縮總大小 14,484,726,761 bytes（約 13.49 GiB）。ZIP 內建目錄是 `nturgb+d_skeletons/`，解壓目的地為 `RGB_vids/nturgb+d_skeletons/`，保留原 ZIP。

`RGB_vids` 是指向 `/export/fallingapart` 的 symlink，實際寫入位於 repository 外；解壓需環境寫入核准。解壓策略為檢查 member paths、拒絕 symlink／越界路徑、不覆寫既有檔案，讀完每個 member 驗證 CRC。盤點時可用磁碟空間約 2.97 TB。

以完整 `SsssCcccPpppRrrrAaaa` identity 將 ZIP entries 與 `artifacts/xclip/selected_split.json` 比對，僅去除影片 stem 的 `_rgb` suffix，得到以下結果；tuning winner 的 train／val／test rows 與 canonical manifest 相同。

| Split | RGB samples | Skeleton 檔名配對 | 配對比例 | 配對類別數 |
| --- | ---: | ---: | ---: | ---: |
| Train | 80,640 | 46,440 | 57.59% | 60 |
| Validation | 12,960 | 4,320 | 33.33% | 60 |
| Test | 20,880 | 6,120 | 29.31% | 60 |

以上是檔名覆蓋上限，尚未扣除官方 missing-skeleton list、損壞內容或無有效人體的樣本；不是可直接訓練的有效 skeleton 數量。下一步必須產生逐 sample 的有效性報告，不能把 ZIP entry 存在直接當作 `has_valid_skeleton=true`。

官方資料說明確認 S001–S017 屬於 NTU60，NTU120 延伸至 S032；A061–A120 不在本 ZIP。官方也列出 NTU60 的 302 個、NTU120 的 535 個 missing／incomplete skeleton samples，應使用對應清單及實際 parser 檢查。[NTU 官方資料說明](https://github.com/shahroudy/NTURGB-D)

### S001–S017 的影響

- 可以開始：teacher 僅以 canonical train 中有效的 A001–A060 skeleton 訓練，student 在全部 RGB train 上做 120-class CE，KD 只作用於有效配對樣本。
- 監督有偏：paired training 主要涵蓋較早 setups／participants／60 classes；validation 與 test 的配對率更低，不能把 paired subset 的改善推論成全部 120 classes 都改善。
- 不可補零假裝完整 teacher：60-class teacher 的 logits 不能直接填零擴成 120-class probability distribution。
- 要完整 NTU120 配對，需補 `nturgbd_skeletons_s018_to_s032.zip`，再重新審核有效樣本、teacher class coverage 與 cache identity；這是後續資料補齊工作。
- NTU60-only 可以另做 pilot，但必須建立明確衍生 manifest，結果不能直接與目前 120-class accuracy 並列比較。

## 3. 論文如何幫助本專案

指定論文是 **SKI Models: SKeleton Induced Vision-Language Embeddings for Understanding Activities of Daily Living**。其 SkeletonCLIP 使用 Hyperformer 與 frozen CLIP text encoder；SKI-VLM 的 SCD 在 skeleton–text／video–text 類別相似度空間做 MSE，並同時訓練兩個模型、保留各自 CE；推理移除 skeleton 分支。論文採 zero-shot class splits，與本專案 participant-disjoint 120-class supervised classification 不同。[論文 v1，§3.1–3.2、§4](https://arxiv.org/html/2502.03459v1)

因此第一版 frozen teacher + classifier-logit KL 是本專案的工程改編，不能宣稱重現原論文 SCD。現有 student 的 `LayerNorm → Dropout → Linear` head 也不是 video–text similarity head。若日後重現 SCD，需另建 text-aligned student、配對 CE-only control、online teacher 與 MSE 實驗，不應在這次加入 KD 時同步替換 head。

官方 SKI repository 提供 skeleton／video 配對 CSV 格式、兩階段訓練 scripts 與 configs；README 未提供足以直接確認可用 SkeletonCLIP checkpoint 及其 split provenance 的資訊，本次未下載或驗證權重。舊環境還包含 Apex，建議抽取所需元件並隔離 teacher 環境，不直接搬入本專案依賴。[SKI Models 官方程式碼](https://github.com/thearkaprava/SKI-Models)

## 4. Teacher 選項與決策條件

| 選項 | 適用性 | 必須先解決 |
| --- | --- | --- |
| 官方 SkeletonCLIP checkpoint | 若可取得且資料來源符合 split，最接近參考方法 | 權重可用性、text model、prompt、class order、訓練 IDs、license、前處理 |
| Hyperformer + CLIP text，自行訓練／對齊 | 建議優先路線；可控制 canonical train 邊界 | backbone 初始化來源、projection、訓練成本與 teacher validation 表現 |
| 純 Hyperformer 60-class classifier | 可作 skeleton KD sanity control | 不具 skeleton–language alignment，必須稱為 skeleton classifier teacher，不能標成 SkeletonCLIP |

Hyperformer 官方 repository 提供 NTU60／NTU120 pretrained weights 連結與 preprocessing scripts，但 action classifier checkpoint 不等於已對齊 CLIP 的 SkeletonCLIP。其資料準備也明列兩份 S001–S017／S018–S032 ZIP。[Hyperformer 官方程式碼](https://github.com/ZhouYuxuanYX/Hyperformer)

Teacher checkpoint 的完整 training provenance 是正式實驗的必要條件。官方 cross-subject／cross-view 或 zero-shot class split 不一定與本專案 participant split 相同；即使只載入 backbone，若曾用本專案 validation／test 樣本及標籤訓練，也會污染比較。無法證明無重疊的權重只能標成 exploratory external-pretraining 實驗，不能用來支持乾淨的 KD 增益結論。

乾淨路線：以 canonical train 的有效 skeleton 從頭訓練 Hyperformer action backbone，再以相同 train 做 skeleton–text CE alignment；CLIP text encoder frozen，只更新 skeleton encoder 及必要 projection。Teacher checkpoint selection 僅看 paired validation，不用 test 選 architecture／epoch／prompt。若使用額外 skeleton training samples，也必须先確保 participant 不跨 canonical 邊界，並另列為額外訓練資料實驗；第一版不擴大 teacher training set。

Text encoder 具體 model ID、feature dimension、prompt templates、tokenizer revision、normalization 與 logit scale 皆需鎖定；不要因兩邊都有 512 維 feature 就假設 embedding space 相同。Teacher 先使用 ordered A001–A060 text vocabulary；選 teacher 主要依 paired validation accuracy、macro-F1、per-class error、校準及 logits 品質，不要求 teacher aggregate accuracy 必須高於 student 才能測試互補效果。

## 5. Skeleton preprocessing 與配對 contract

- 新增逐 sample manifest：`sample_id`、`video_path`、`skeleton_path`、`class_id`、`class_index`、`split`、setup／camera／participant／replication、`has_valid_skeleton`、排除原因、source fingerprint；拒絕重複 identity、label 不一致與跨 split identity。
- 依官方格式解析 frame、body 與 25 joints 的 3D XYZ；檢查 frame/body/joint counts、截斷、非有限數值、空序列及有效 tracking 資訊，保留 excluded reason。Missing skeleton 只關閉該 RGB sample 的 KD，不取消 CE。
- Backbone input 暫定 contract 為 `[N, 3, T_s, 25, M]`，實際 `T_s`、`M`、joint／bone／motion stream 依所選 checkpoint 明確鎖定。多人互動不可無條件只留第一個 body；body tracking、排序、denoising 與 padding 必須與 teacher 訓練一致。
- Root centering、座標尺度／旋轉、temporal resampling、valid-frame masks 與 padding 版本化；不可拿通用 normalization 取代 checkpoint 原本 preprocessing。Padding 不得被當作有效 motion。
- RGB 仍採 whole-video 八幀；skeleton 可採較密時間取樣，但必須涵蓋同一影片時間範圍。不得假設八個 RGB indices 可以直接索引 skeleton frames；先檢查 frame counts 與時間對應。
- 第一版 teacher cache 使用 deterministic whole-clip skeleton input。Student winner 的 feature noise 可保留；若後續新增 temporal crop，teacher logits 也需對應相同時間窗或明確另做 full-clip supervision ablation。

## 6. Student 與 masked KL 設計

沿用 `src/vlm/models/xclip_lora.py` 的 MIT LoRA classifier 與 `src/vlm/xclip_pre_mit_features.py` 的 `[8, 512]` cache。KD 不解凍 vision encoder、不改 RGB inference head，也不需要每個 epoch 重算 vision features。

這個設計傳遞的是 teacher 對動作類別相對關係的判斷，而不是關節位置。共享的 MIT／head 參數可能從 A001–A060 學到可遷移的 temporal cues，因而間接幫助 A061–A120；但這不是 KD 的保證，也可能因 class、setup 或 participant coverage 偏差而傷害後 60 類。必須分開報告 A001–A060 與 A061–A120，只有後者相對 CE-only control 也穩定改善時，才能主張存在跨類別 transfer。

目前 `artifacts/xclip_lora_tuning/winning_config.json` 的設定為 rank 16、alpha 32、LoRA dropout 0.05、head dropout 0.25、LoRA LR `2e-4`、head LR `8e-4`、weight decay `5e-4`、label smoothing 0.05、feature noise std 0.01、60 epochs、batch size 16、seed 42。實作時讀取並保存完整設定及 hash，不只複製此摘要。Winner artifact 記錄 validation accuracy 0.521219、final test accuracy 0.545163；這是既有結果，不能拿來調 KD。

公平比較採相同 pretrained XCLIP 初始化及 seed 重跑 CE-only 與 CE+KD，固定原 winning recipe。不要只讓 KD 從已訓練 60 epochs 的 winner 接續訓練；若研究續訓，須給 CE-only 相同初始化與額外步數。

定義 `C_T` 為 teacher 真正受監督的 ordered class IDs，目前為 A001–A060。對 batch 中有效配對集合 `P`，先用 class ID 對應取出 student 的 60 個 logits，再計算兩邊各自在這 60 類上的 softmax：

```text
p_i = softmax(teacher_logits_i / T)
q_i = softmax(student_logits_i[C_T] / T)
L_KD = T² / |P| × Σ(i∈P) KL(p_i || q_i)
L = CE(student_logits_120, labels_120) + λ × L_KD
```

`P` 為空時 `L_KD=0`，保持 CE 正常更新。KL 以 FP32 的 `log_softmax`／`softmax` 計算，先對 classes 求和再對 paired samples 平均；teacher 為 `eval()` 且 no-grad，logits detach。這只傳遞前 60 類的相對分布，不直接約束 student 在 A061–A120 的總機率質量；120-class CE 持續負責全類別判別。

記錄每個 epoch 的 CE、unweighted KD、weighted KD、paired sample count／fraction，避免把無配對資料誤認為低 KD loss。Gradient accumulation 時按有效樣本數處理 normalization，避免每個 micro-batch 等權平均不同大小的 paired 集合。第一版維持既有 shuffle／sampling，不以 oversampling 配對樣本改變資料分布。

Distillation-only 初始 validation search 可用 `T ∈ {1, 2, 4}`、`λ ∈ {0.1, 0.5, 1.0}`，先檢查 CE／KD 實際尺度再縮小搜尋；`λ=0` 是必備 control。這些值是工程起點，並非引用論文最佳值。Feature cosine loss 留待 logit KD 有結果後再做獨立 ablation。

## 7. Cache、模組與 artifacts

以下為提案介面，目前尚未實作；優先小幅擴充既有 pipeline，不複製整套 LoRA training／evaluation。

| 路徑 | 責任 |
| --- | --- |
| `src/vlm/data/skeleton.py`（新增） | NTU parser、有效性檢查、identity pairing 與 manifest |
| `src/vlm/models/skeletonclip.py`（新增） | Teacher backbone adapter、text projection、ordered class logits |
| `src/vlm/training/skeletonclip.py`（新增） | Teacher 訓練、paired validation selection、provenance |
| `src/vlm/skeleton_teacher_cache.py`（新增） | Frozen teacher deterministic logits extraction／validation |
| `src/vlm/training/distillation.py`（新增） | Explicit class mapping、masked KL 與 loss accounting |
| `src/vlm/training/xclip_lora.py` | 重用既有 training path，接入 optional teacher batch／loss，保持 CE-only 預設 |
| `src/vlm/cli.py` | 提案新增 teacher train／cache 與 `xclip-lora-distill` mode |

Teacher logits cache 放在 `cache/skeleton_teacher_logits/<identity>/`，與 RGB cache 分離。Identity 至少含 teacher checkpoint SHA256、architecture／revision、training split hash、ordered class IDs、text model／prompts、logit scale、preprocessing version、sampling policy、來源 skeleton fingerprint 與 dtype。保存 temperature scaling 前的 raw logits，T／λ 改變不需重新 extraction；teacher 權重或前處理改變則必須重建。不可將 online teacher 與靜態 logits cache 混用。

資料 audit 建議輸出 `artifacts/skeleton_data/coverage.json`、`paired_manifest.json`、`excluded_samples.json`；teacher 放 `artifacts/skeletonclip_teacher/<run>/`；student 放 `artifacts/xclip_lora_distill/<run>/`。Run artifacts 保存 source manifests／hash、teacher provenance、config、class mapping、validation selection、loss history、coverage、compute cost 與 prediction rows。RGB merged checkpoint 沿用既有 inference contract，只在 metadata 記錄 KD 來源，不載入 teacher runtime state。

## 8. 實作順序與驗收

1. 完成解壓與資料 audit：核對 56,880 檔案，套用官方排除清單並完整 parse；輸出各 split／class／setup 的有效配對率、missing 原因與 duplicate 檢查，保留 canonical RGB split。
2. Teacher feasibility：確認 architecture、checkpoint provenance、text vocabulary 與 preprocessing；跑小型 forward／overfit smoke、量測 GPU memory／throughput，再決定完整 teacher 訓練預算。外部權重無法證明 split 安全時使用乾淨自訓路線。
3. Teacher train／validation：只用 train 更新參數，依 paired validation 選 checkpoint；確認 60-class text alignment 與 error profile，再 freeze 並建立 train logits cache。
4. Student integration smoke：混合 paired／unpaired batch 完成 CE+KL backward、空配對 batch、checkpoint round trip；確認 teacher 無 gradient，student 僅 LoRA／head 可訓練。
5. Controlled experiments：相同 split／seed／recipe／training budget 的 CE-only 與 CE+KD；只以完整 RGB validation 選 T／λ 及 checkpoint，paired validation 僅作診斷。選定方案後才對該方案做一次 RGB-only test。
6. 報告：完整 120-class accuracy，另列 A001–A060／A061–A120、paired／unpaired subgroup accuracy、macro-F1、樣本數及 training cost；任何新增 metrics 對 control 同步計算。依預先固定 seeds 比較穩定性，不以 test 選 seed。

只新增保護重要契約的測試：identity／split 防漏、parser 失效樣本、class reorder 對應、teacher-cache invalidation、partial-class KL／無配對 batch、teacher gradient isolation，以及 student checkpoint 的 RGB-only reload／merge parity。沿用既有 LoRA checks；一般測試不下載權重或掃完整資料集。

成功標準首先是資料與實驗可信：KD 關閉時重現 CE-only contract，有效配對才提供 teacher supervision，test 不參與選模，推理不依賴 skeleton。是否改善全體及動作子群結果，由 validation 與最終受控評估回答，不預設缺一半類別的 teacher 必然有效。第一版若成功，可主張 skeleton-informed RGB classification 或 temporal representation adaptation；沒有 spatial evidence 時不得描述為 joint-focused model。

## 9. 後續決策

第一個要定案的是 Hyperformer 初始化與 SkeletonCLIP checkpoint 來源，不是先擴大 student。接著依資料 audit 和 teacher validation 決定是否值得補 S018–S032。完整資料到位後才考慮 120-class teacher、online MSE SCD 或 feature alignment，且每項都需要自己的 control。

Student adaptation 分成以下兩個後續層次，不與第一版 class-level KD 同時引入：

1. **Vision upper-layer LoRA + class-level KD**：若第一版 validation 顯示 KD 有效，再對 vision encoder 最後數個 spatial Transformer blocks 加 LoRA，讓 teacher loss 可以影響 RGB spatial features。這會取消完全依賴離線 `[8, 512]` pre-MIT cache 的訓練方式，需從 RGB frames 執行 vision forward。它只能證明 vision adaptation 是否帶來額外效益；class-level loss 仍未指定模型必須看 joints，因此不能單憑分類結果宣稱 joint focus。
2. **Spatial joint supervision**：只有研究目標明確要求 joint-focused RGB learning 時才投入。保留 patch tokens，將 skeleton joints 可靠地投影並同步到 RGB frames，再以 joint heatmap、joint-to-patch feature 或 attention alignment 直接監督 spatial regions。除分類 metrics 外，還需 joint-region localization、masking／occlusion 等證據，才能支持模型更專注 joints 的主張。

兩個層次都維持 training-only teacher 與 RGB-only deployment。訓練完成後，推理不載入 SkeletonCLIP、skeleton、pose estimator 或 KD loss；只有當未來架構明確把 pose／joint heatmap 設為 inference input 時，才會產生額外 teacher／pose runtime dependency。
