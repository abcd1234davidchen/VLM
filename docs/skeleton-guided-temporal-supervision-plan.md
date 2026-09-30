# Skeleton-guided temporal importance supervision plan

日期：2026-09-29。狀態：proposal；這是與 spatial supervision分開的獨立實驗。目標是利用 skeleton motion產生 training-only frame／interval importance targets，直接指導 XCLIP MIT辨識哪些時間點包含判別性動作，而不是模仿 BlockGCN的最終 class logits。

## 1. 研究問題

目前 XCLIP從完整影片 uniform取 8 frames，但不同 action的關鍵訊號可能只出現在很短的區段，例如物件被拿起或放下、手勢形成的瞬間、兩人接觸或動作方向反轉。Skeleton velocity與joint configuration change能否提供較好的 temporal target，使 MIT把較多 decision weight放在真正發生動作的 frames，並改善 RGB-only classification？

這條路線不假設「motion最大的一幀一定最重要」。靜態手勢、持續姿勢與細小物件互動可能具有低 skeleton velocity但高度判別性，因此 target必須同時保留 uniform floor與configuration-change signal，並使用 shuffled／energy-only controls驗證。

## 2. Temporal target construction

以完整 skeleton sequence計算每個 skeleton frame的下列 signals，再映射到 RGB影片時間軸與實際抽取的 8 個 XCLIP frames：

- `joint velocity`：root-normalized 3D joint displacement，依有效 joints與people聚合，使用 robust normalization避免 tracking spike主導。
- `joint acceleration/change point`：捕捉起身、放下、接觸、方向反轉等短暫轉換。
- `configuration change`：相鄰 frame的 normalized bone-direction／pairwise-joint geometry差異，降低純 global translation造成的假高 motion。
- `validity`：無可靠 skeleton或時間對齊的 frame不提供 supervision。

將 signals平滑後整合到每個 sampled RGB frame的 temporal cell，形成 8 維 nonnegative distribution。Target加入固定 uniform floor `ε`再 normalize，避免所有 attention壓到單一 frame；`ε`、平滑窗口與 signal weights必須在 full run前由資料統計與少量可視化鎖定，不能依 validation accuracy搜尋。

## 3. Model objective

XCLIP projected frame features送入 MIT前，新增輕量 temporal scoring head，為 8 frames產生 importance distribution。第一版不直接監督 transformer內部各 layer／head attention，因 raw attention不等同 causal importance且難以跨 heads定義唯一 target。

```text
8 RGB frame features → temporal scoring head → learned importance distribution
                    └→ gated/reweighted features → XCLIP MIT → 120-class logits
Skeleton motion ── training only ───────────────→ temporal target distribution
```

Feature gate使用 residual形式，例如 `x_t × (1 + α(w_t - 1/F))`，並限制初始狀態等同原模型，避免 auxiliary branch在 initialization就破壞 pretrained MIT。總 loss為：

```text
L = L_CE_120 + λ_temporal × KL(target_time || predicted_time)
```

只有 MIT q/v LoRA、classification head與 temporal scoring／gate parameters可訓練；vision encoder沿用固定 CE student contract，避免同時測 temporal與vision adaptation。部署時保留由 RGB features計算的 temporal scoring head，但不需要 skeleton；因此必須另外報告新增 parameters與latency。若最終希望零額外 branch，可在機制成立後另做 merge／distill ablation，不能在第一輪混入。

## 4. Required controls

- `CE identity-gate`：模型含相同 temporal branch與parameter count，但 gate固定 identity且沒有 temporal loss，用來隔離新增 architecture本身的效果。
- `learned RGB-only gate`：temporal branch只由 CE學習，沒有 skeleton target；回答提升是否其實不需要 skeleton。
- `correct temporal`：加入 identity正確的 skeleton-derived target。
- `shuffled temporal`：在 action class與valid-length條件下置換 target，保留target sharpness與coverage但破壞 sample timing。
- `motion-energy fixed gate`：不學 temporal predictor，直接在 paired validation diagnostic中使用 skeleton energy；只用來估計 target oracle方向，不作可部署模型。
- `uniform target`：檢查任何收益是否只是正則化 attention接近平均，而非 skeleton timing。

正式第一輪至少需要 identity-gate、RGB-only gate、correct temporal與shuffled temporal。若算力有限，uniform與oracle只做 offline diagnostic，不刪除主要 controls。

## 5. Pairing and missing skeletons

所有 120-class samples仍計算 CE。只有具可靠 skeleton、frame-time mapping與足夠 valid frames的 samples計算 temporal KL；all-unpaired optimizer step的 temporal loss為 differentiable zero。Sample identity、video frame count、skeleton frame count、mapping policy與 source fingerprint必須進 cache identity並 fail closed。

目前 skeleton supervision主要涵蓋 A001–A059與部分 participants，因此不可把 paired improvement當成全 120 類 improvement。Batch sampling不因是否 paired而改變，避免 supervision coverage同時改變 data distribution。

## 6. Evaluation

Model selection只使用 canonical validation。Classification報告 overall accuracy／macro-F1、A001–A059與A060–A120、paired與unpaired、static／dynamic action buckets，以及完整 per-action accuracy。Temporal mechanism報告 predicted-target KL、top-k temporal recall、target entropy、gate entropy與相對 uniform／motion-energy baselines的結果。

必須增加 causal temporal diagnostic：依 predicted importance遮掉或替換最高權重 frame，並和相同數量的 random／lowest-weight frames比較 correct-class logit下降；另做 time-order perturbation，檢查依賴順序的 action是否比近似靜態手勢更敏感。只有 correlation改善不足以聲稱 model真正使用了關鍵 frames。

`docs/xclip-validation-action-accuracy.md` 提供 action difficulty baseline。預先人工將 classes標成短暫事件、持續動作、靜態手勢、小物件依賴與interaction categories後才做 subgroup analysis；不能看完新模型結果再移動分類。

## 7. Gates

### Gate 0：target validity

隨機與stratified抽樣可視化完整影片 motion curve、8-frame aggregation與對應 RGB thumbnails。量化 alignment error、valid coverage、entropy、tracking spikes、single／two-person差異，以及每個 action的 target sharpness。若 target主要追蹤camera/global body translation或大量樣本近似uniform，先修 target，不啟動 training。

### Gate 1：mechanism smoke

驗證 correct identity、cache invalidation、all-unpaired loss、gradient whitelist、identity-gate initialization parity、gradient accumulation normalization、strict reload與 RGB-only inference。小型 overfit中 correct temporal target必須顯著優於 shuffled target的 temporal loss。

### Gate 2：single-seed full comparison

所有 arms使用相同 initialization、training budget與checkpoint selection。Correct temporal相對 RGB-only gate必須：(a) temporal mechanism primary metric優於 shuffled與uniform；(b) causal masking方向一致；(c) overall accuracy或macro-F1達至少 `+0.20` points，且 A060–A120與unpaired各不退步超過 `0.50` points，才進 confirmation。

### Gate 3：second-seed confirmation

補第二 seed並以 paired difference與bootstrap confidence interval報告。兩個 seeds方向一致後才鎖定 winner；未達門檻則保留為mechanism-only或no-go，不讀 test。Test-once只能在所有 target、weight、epoch與class scope不可變後執行。

## 8. Interaction with action filtering

Action filtering是第三條 task-definition路線，不可和 temporal supervision在同一首次 comparison一起做。先由人依產品需求在 validation ranking上決定 keep／remove／merge；若採用新 class set，必須先重跑該 task的 CE／RGB-only temporal baseline，再比較 skeleton temporal target。否則無法知道 gain來自較容易的 label space還是 temporal supervision。

## 9. Stop conditions

若 correct與shuffled targets近似、RGB-only gate已得到全部收益、causal masking不支持 learned importance、或改善只出現在paired A001–A059而未跨到目標 deployment classes，停止或縮小 claim。不得以 test結果挑 signal weights、uniform floor、loss weight或 action subset。

## 10. Deliverables

- Temporal target specification、cache schema與 immutable source fingerprints。
- Motion curve／RGB overlay audit與 coverage statistics。
- Identity-gate、RGB-only、correct、shuffled的完整 configs、predictions與 compute report。
- Per-action及預先定義 subgroup results、causal temporal diagnostics與 uncertainty。
- RGB-only checkpoint、inference latency／parameter delta與 go／mechanism-only／no-go decision。
