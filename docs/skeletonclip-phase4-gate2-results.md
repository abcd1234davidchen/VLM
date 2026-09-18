# Phase 4 Gate 2：spatial joint supervision 對照結果

日期：2026-09-18。三組正式訓練均已完成；決策為 **conditional spatial signal，暫不進 Gate 3**。Machine-readable 摘要在 `artifacts/skeletonclip_phase4/gate2/gate2_decision.json`，三組完整 epoch history 與 immutable experiment lock 在同目錄。未讀取 test split。

## 固定實驗契約

三組共用 Phase 3 vision-only 最佳 adapter、XCLIP revision `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`、seed 42、120-class RGB split、batch 16、20 epochs、相同 optimizer recipe 與 spatial weight 0.1。每個 epoch 使用 80,640 筆 train 與 12,960 筆 validation；paired train／validation 分別為 45,236／4,180 筆，invalid joint frames 在 spatial loss 與定位評估中遮罩。RGB 沿用 Phase 3 uint8 cache，joint patch targets 使用 `cache/skeletonclip_phase4_targets_v1`。CE arm 的 spatial probe 讀取 detached patch tokens，probe loss 不回傳到 vision；joint arm 讓正確 joint targets 的 spatial gradient 更新 vision；shuffled-joint arm 以固定 derangement 置換 paired train targets，validation targets 維持正確配對。模型推論仍只需 RGB。

事前鎖定以第 20 epoch 作同預算比較，另記錄各組最佳 validation accuracy epoch；定位使用 held-out paired validation 的 33,339 個有效 frame，計算 patch AP 與依 occupied-patch 數量的 top-k recall，並比較固定中心 prior。Gate 2 的數值條件是 joint patch AP 高於 CE 與中心 prior，且 shuffled-joint 不得到同樣提升；Gate 3 仍要求可信且可重現的 spatial evidence。

## Validation 結果

| Arm，第 20 epoch | 120-class accuracy | Patch AP | Top-k recall | 最佳 accuracy（epoch） |
| --- | ---: | ---: | ---: | ---: |
| CE | 58.5417% | 0.954959 | 0.892403 | 59.3133%（6） |
| Joint | 58.6188% | 0.956869 | 0.895270 | 59.3519%（6） |
| Shuffled-joint | 58.5880% | 0.759423 | 0.656653 | 59.2438%（6） |

固定中心 prior 在相同有效 frame 上的 patch AP 為 `0.547461`。Joint 相對 CE 的最終 patch AP 高 `0.001911`，top-k recall 高 `0.002867`，分類 accuracy 高 `0.0772` 個百分點；joint 在 20 個對應 epoch 的 patch AP 都高於 CE，但這些 epoch 共用訓練軌跡，不能當成 20 次獨立重複。Joint 相對 shuffled-joint 的最終 patch AP 高 `0.197446`；shuffled-joint 的分類 accuracy 仍與其他兩組近似。三組最佳 validation accuracy 均出現在 epoch 6，此數字只作趨勢描述，不取代事前鎖定的第 20 epoch 比較。各組峰值 allocated VRAM 約 `1.07–1.09 GiB`；第 1 epoch 包含 cache 建立成本，不宜代表穩態吞吐。

## 判讀與決策

三組報告與 12,960 筆完整 validation coverage 均齊全，test embargo intact。依事前鎖定的方向性比較，joint 的定位數值確實高於 CE 與中心 prior，shuffled-joint 也明顯較低，故記錄為 **conditional spatial signal**。但 CE probe 本身用正確 joint targets 訓練且已達 `0.954959` patch AP；joint 對它的額外增益只有 `0.001911`，目前只有單一 seed、沒有逐 sample 成對不確定性估計或獨立重複，無法將小幅差異稱為可重現、具實際意義的 vision grounding 改善。分類 accuracy 幾乎相同，Gate 2 不支持「joint supervision 改善 120-class accuracy」的主張。這次 runner 只保存 aggregate accuracy，沒有保存可計算 A001–A059／A060–A120、paired／unpaired subgroup 的 validation logits；roadmap 要求的分類 subgroup 分析尚未完成，這也是完整 Gate 2 解讀的限制，不得由 aggregate 結果推論各 subgroup。

**Gate 3 暫不啟動**：目前不擴大 spatial weight 搜尋，不讀 test。未來若要重新研究這條路線，需先獨立鎖定重複 seed、樣本層級 uncertainty、預期最小有效定位差距與分類 subgroup 分析；這是後續決策，不是本 Gate 2 結果的一部分。

## 續接資訊

執行指令保存在 `artifacts/skeletonclip_phase4/gate2/launch.sh`，正式 session 原名為 `phase4_gate2`；三組 `run_report.json`、`last_checkpoint.pt`、`progress.json` 和追加式 `train.log` 位於各 arm 子目錄。`train.log` 的 live watcher 在 CE 第 1 epoch 約第 800 batch 才加入，早期 batch 快照未保存，但三組完整 epoch history 均在 checkpoint 與 report 中。`experiment_lock.json` 的 `status=locked_pending_gpu_preflight` 是啟動時的 immutable 狀態；完成狀態以三組 report 與 `gate2_decision.json` 為準。原始 source hash、split／adapter hashes 與 test embargo 皆保留在 lock／decision 中。
