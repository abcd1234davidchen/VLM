# XCLIP MIT LoRA 超參數調整計畫

## 目的

在加入 SkeletonCLIP distillation 前，先建立一套只依 validation 選出的 `XCLIP MIT LoRA + CE` 訓練設定。目前 rank 8、MIT `q_proj`／`v_proj` 的實驗保留為未調整的 control；超參數調整的目標是在不使用 test 結果選設定的前提下，改善並穩定這條 baseline。

## 建議命令

新增一個獨立的 training mode：

```bash
uv run vlm train xclip-lora-tune \
  --split-manifest artifacts/xclip/selected_split.json
```

專案的 executable 是 `vlm`，因此命令沿用既有的 `uv run vlm train <mode>` 慣例。`xclip-lora-tune` 負責協調多個 trial，但應重用 `xclip-lora` 所擁有的 model、cache、training loop、checkpoint 與 evaluation contracts，不應複製出另一套 training implementation。

## Cache 共用

Tuning mode 應與一般 `xclip-lora` run 共用既有的 `cache/xclip_pre_mit_features/` cache。這些 tensor 在 MIT module 之前產生，因此 LoRA rank、alpha、target modules、LoRA dropout、head dropout、learning rates、weight decay、label smoothing、scheduler、warmup、epochs 與 random seed 都不會改變 cache identity，也不應觸發重新 extraction。

只有在來源影片路徑、pretrained XCLIP model ID、Transformers version、whole-video frame-sampling strategy、frame count、pre-MIT feature stage、tensor shape 與 storage dtype 仍相容時，cache 才能共用。若更換 base model、frame policy、frame count、feature extraction stage 或 dependency version，應選用不同的 cache identity，或重建不相容的 entries。

Split manifest 決定每個 trial 讀取哪些 cached videos，但同一部影片不需要因不同 split 或 trial 而複製 tensor。Tuning orchestrator 應在啟動 trials 前統一準備或驗證所需 cache，之後將它視為 shared read-only input。Concurrent trials 不應各自強制 extraction，也不應同時重寫 shared manifest。

Feature-level training augmentation 必須在載入 cached tensor 後才抽樣，且不得覆寫 canonical cache。Pixel-level spatial 或 color augmentation 發生在 vision encoder 之前，因此不能把 canonical tensor 當作已 augmentation 的樣本；這類實驗需要 online extraction 或具有獨立 identity 的 cache variants。

目前 cache 以影片路徑識別來源，尚未對檔案內容建立 fingerprint。在既有 cache-provenance TODO 完成前，即使路徑沒有改變，只要影片在原位置被替換，就可能需要明確執行 forced refresh。

## 實驗邊界

- 所有 trials 使用同一份 canonical participant-disjoint split manifest。
- Extraction identity 沒有改變時，所有 trials 共用同一份已驗證的 pre-MIT cache。
- 只使用 validation metrics 選擇 configuration；個別 tuning trial 不得評估 test split。
- Winning configuration 固定後才評估一次 test split，最好再搭配預先定義的 seed set。
- 除非某項 planned ablation 明確改變其中一項，否則 task、class mapping、pretrained model、八幀輸入政策、pre-MIT cache contract、classification head shape 與 checkpoint-selection metric 都應保持不變。
- 成功、失敗與中斷的 trials 都要記錄，讓 search 可以稽核與 resume。

## 分階段搜尋

Search 應採分階段方式，不直接計算所有選項的 Cartesian product。每個階段只讓 validation accuracy 有足夠且穩定改善的 configurations 進入下一階段。

### Stage 0：重現目前的 control

- 重現目前的 `rank=8`、`alpha=16`、MIT `q_proj`／`v_proj`、LoRA dropout `0.05`、head dropout `0.4`、LoRA learning rate `1e-4`、head learning rate `8e-4`、weight decay `5e-4` 與 cosine schedule 設定。
- 確認新 mode 在相同 seed 與設定下，能重現一般 `xclip-lora` 的結果。

### Stage 1：Optimization

- 將 LoRA 與 classification head learning rates 視為不同 parameter groups 分別探索。
- 比較目前的 cosine schedule 與加入短暫 warmup 的 cosine schedule。
- 加入 early stopping 或明確的 plateau rule，避免 validation 停止改善後仍完整跑完所有 epochs。
- 此階段固定 LoRA 結構與 augmentation，讓 optimization 造成的影響可以獨立辨認。

### Stage 2：LoRA capacity 與 targets

- 比較 `4`、`8`、`16` 等實用 LoRA ranks，並有意識地選擇 alpha，讓各組 scaling 保持可比較。
- 以 MIT `q_proj`／`v_proj` 作為主要 target set。
- 只有主要 target set 得到穩定的 tuned result 後，才考慮更廣的 MIT targets，例如 `q_proj`／`k_proj`／`v_proj` 或特定 MIT MLP projections。
- Main sweep 不擴展到 vision encoder；這應視為獨立的 representation-adaptation experiment，另行評估 compute 與 overfitting。

### Stage 3：Regularization

- 以目前 defaults 為中心，調整 LoRA dropout、classification-head dropout、weight decay 與 label smoothing。
- 先分別改變各個 regularization family，再組合表現最好的 candidates。
- 除了 validation accuracy，也要追蹤 train／validation gap，因為目前 run 在 validation 進入 plateau 後，training accuracy 仍持續上升。

### Stage 4：Augmentation

- 優先測試與 pre-MIT cache 相容的 feature-level temporal augmentation，例如 random frame dropout、cached frame positions 上的 temporal jitter，或小幅 Gaussian feature noise。
- Pixel-level spatial 或 color augmentation 需要 online vision-encoder work 或多份 cache variants，成本較高，應視為獨立實驗。
- Augmentation 只套用於 training samples，並在每個 trial artifact 記錄 policy、probability、parameters 與 seed。

### Stage 5：確認結果

- 使用預先定義的 seed set，重新執行 validation 表現最好的少數 configurations。
- 依 aggregate validation performance 與穩定性選出最終 recipe，不以單次最高 trial 決定。
- 固定 winning recipe 後才評估一次 test，並將 mean、standard deviation、training cost 與 trainable parameter count，和未調整的 LoRA control、frozen XCLIP baseline 並列報告。

## Trial 輸出

每次 tuning run 應使用獨立的 parent artifact directory，並為每個 trial 分配 stable ID。Parent summary 應包含完整 search space、完成與失敗的 trial states、精確 arguments、seed、Git revision、split identity、best epoch、validation history、runtime 與 selected winner。各 trial 的 checkpoints 與 plots 必須隔離，避免 concurrent 或 resumed searches 互相覆寫。

Tuning 結果應是一份可交回一般 `xclip-lora` training mode 使用的 reusable configuration。Tuning mode 是 experiment orchestrator，不是新的 inference model 或 checkpoint format。

## 完成結果（2026-08-26）

本計畫已在固定的 `artifacts/xclip/selected_split.json` 上完成 Stage 0–5 與最後一次 locked-winner test evaluation。Split 包含 80,640 個 training samples、12,960 個 validation samples 與 20,880 個 test samples；所有 tuning trials 只使用 validation 選擇 configuration，test 僅在 winner 固定後評估一次。

Winning recipe 為 MIT `q_proj`／`v_proj` LoRA rank 16、alpha 32、LoRA learning rate `2e-4`、head learning rate `8e-4`、LoRA dropout `0.05`、head dropout `0.25`、weight decay `5e-4`、label smoothing `0.05` 與 Gaussian feature noise standard deviation `0.01`。它不使用 warmup、early stopping、frame dropout 或 temporal jitter，最終 run 使用 seed 42、60 epochs、effective batch size 16 與 BF16。

| Model | Best validation accuracy | Final test accuracy |
|---|---:|---:|
| Frozen XCLIP baseline | 0.3927 | 0.4254 |
| Tuned MIT LoRA + CE | 0.5212 | 0.5452 |
| Absolute improvement | +0.1285 | +0.1198 |

Final run 的 best validation checkpoint 位於 epoch 49。Epoch 60 的 training accuracy 為 0.6973、validation accuracy 為 0.5211，顯示約 0.176 的 train／validation gap，但 validation 在最後數個 epochs 維持約 0.520–0.521，沒有明顯崩落。最終 test accuracy 高於 validation 約 0.024，且 20,880 筆 predictions 與 test sample count 完全一致。

模型只有 95,352 個 trainable parameters，佔完整 196,585,729-parameter XCLIP 約 0.0485%。Merged checkpoint 的 maximum absolute parity error 為 `1.91e-6`，符合 FP32 `1e-5` tolerance。正式 artifacts 由 Git revision `9f1d5618f226fbef31fadb7055ad479231c3f316`、PyTorch `2.10.0+cu128`、Transformers `5.3.0` 與 Python `3.13.13` 產生，位於 `artifacts/xclip_lora_tuning/winner/`。

這份結果支持將 rank-16 tuned LoRA + CE recipe 作為後續 SkeletonCLIP distillation 的 RGB student control。後續 controlled comparison 應固定本 recipe、split、八幀 whole-video sampling、seed policy 與 training budget，再加入 teacher-logit KL supervision；不可重新使用 test 結果調整 distillation hyperparameters。

## 與 SkeletonCLIP 的關係

只有本計畫產生穩定的 `LoRA + CE` recipe 後，才開始 SkeletonCLIP distillation。第一個 controlled distillation comparison 必須固定選出的 LoRA rank、targets、augmentation、optimizer settings、split、seed policy 與 training budget，然後只把 teacher-logit KL supervision 作為主要新增變因。Temperature 與 KL weight 等 distillation-specific values 可以另外使用 validation 調整。

預定的實驗順序如下：

```text
Frozen XCLIP baseline
→ untuned MIT LoRA + CE control
→ tuned MIT LoRA + CE baseline
→ 相同 tuned LoRA + CE + SkeletonCLIP logit distillation
→ optional feature-level distillation ablation
```

## 初始驗收條件

- `xclip-lora-tune` 以獨立 training mode 顯示，並將 single-trial behavior 委派給既有 LoRA pipeline。
- 小型 search 可以中斷與 resume，不會重跑已完成 trials。
- Tuning trials 不讀取也不報告 test metrics。
- Search summary 能重現 winning configuration 及其以 validation 為依據的 selection。
- Winning configuration 可以透過一般 `xclip-lora` path 執行，並產生既有 adapter 與 merged checkpoint formats。
