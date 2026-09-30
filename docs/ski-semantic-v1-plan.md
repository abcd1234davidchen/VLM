# SKI-style semantic pilot

## 方向與前提

2026-09-30，使用者在暫停motion-pooling後要求嘗試SKI方法。兩次motion-pooling結果保留，不延伸該變體，也不把失敗外推成SKI不可行。SKI-VLM的核心是online collaborative distillation：對齊video-text與skeleton-text的score matrices，再加兩個action CE losses；不是把joint座標投影成RGB patch targets。[論文 §3.2](https://arxiv.org/html/2502.03459v1#S3.SS2)，[官方訓練入口](https://github.com/thearkaprava/SKI-Models/blob/main/P2_SKIVLM_main.py)。Figure 1右側是qualitative activation maps與LVLM描述，不是joint-to-patch alignment ground truth。

沿用`artifacts/skeletonclip_gate3/xclip_seed42/best_checkpoint.pt`：BlockGCN加XCLIP text prototypes，A001–A059，64-frame skeleton，既有validation accuracy約86.22%。不重訓teacher prealignment。RGB沿用Phase 3 best vision/MIT LoRA與120-class classifier，baseline raw accuracy約58.92%。資料只有45,236 paired train／4,180 paired val；其他RGB樣本仍參與120-class CE，沒有skeleton時不使用語意KD。

## 明確的改版範圍

- 這是supervised120／partial-pair59的SKI-style改版，不是論文NTU120 zero-shot split的完整重現；不拿兩者accuracy直接比較。
- Shared frozen59-class XCLIP text prototypes；RGB從MIT features經identity-initialized linear semantic projection取得text-conditioned scores。120-class主head不換成59-class head，語意branch是訓練輔助。
- 先抽取凍結RGB features，再固定12 epochs prealign semantic projection，AdamW LR`1e-3`；不改baseline RGB logits。Smoke用256 paired train／118 paired val，pilot用pilot train中paired樣本／472 paired val。需held-out semantic CE下降且accuracy高於chance，否則不進入collaborative training。
- Online階段RGB vision/MIT只更新既有LoRA；120-class head與semantic projection可更新；skeleton teacher參數也更新。Small batch下teacher維持eval mode，固定BatchNorm running statistics，並非凍結teacher參數。
- Loss為`CE120(all RGB) + CE59(video-text, paired) + CE59(skeleton-text, paired) + 0.01*MSE(video_scores, skeleton_scores)`。Scores使用teacher初始learned scale約34.47；這與官方raw-cosine logits有temperature差異，權重不宣稱等價。Student scale固定，teacher learned scale保留online更新。
- Online固定1 epoch，AdamW LR`2.25e-5`、cosine到0、gradient clipping1、fp32、batch4、workers4且hard cap24。不搜尋epoch或loss weights。
- 不新增空間pooling；模型部署匯出原120-class RGB classifier，discard semantic branch與skeleton teacher。Strict merged RGB-only reload必須通過。

## Smoke、pilot與判讀

Smoke用128個120-class balanced train樣本與120個validation樣本做短程mechanism檢查。檢查warm-start RGB logit parity、語意prealignment、pure MSE backward確實進入vision/MIT與online teacher、finite gradients，以及RGB-only merged reload。Smoke的分類accuracy不當成勝出證據，不要求完整control group。

Smoke通過且source hashes相符後，只跑一個seed42 pilot：8,192 train，seeded sample-ID hash加equal-class round robin選取；全部12,960 canonical validation。比對既有baseline logits，報raw120 accuracy、macroF1、per-action confusion、paired/unpaired與A001–A059/A060–A120 groups。增益至少0.20 percentage points才標記promising並考慮confirmation；負增益no-go，小正增益inconclusive。單一pilot不足以選winner；不評估test、不訓練106-class模型。

Runner為`src/vlm/training/ski_semantic.py`，獨立artifacts namespace為`artifacts/ski_semantic_v1/`。每個run不可覆寫lock，保存source hashes、train/val IDs、runner snapshot、progress、validation predictions、checkpoint與merged RGB state。Dispatcher沿用`spatial_dispatch`的通用command入口，backend為tmux；`console.log`保存stdout/stderr，`train.log`保存persistent進度，`dispatch.json`記錄PID與session，`dispatch_status.json`保存terminal狀態。

```bash
.venv/bin/python -m vlm.training.spatial_dispatch --artifacts-dir artifacts/ski_semantic_v1/smoke --tmux-session ski-semantic-smoke -- .venv/bin/python -m vlm.training.ski_semantic --mode smoke --artifacts-dir artifacts/ski_semantic_v1/smoke
.venv/bin/python -m vlm.training.spatial_dispatch --artifacts-dir artifacts/ski_semantic_v1/pilot_seed42 --tmux-session ski-semantic-pilot -- .venv/bin/python -m vlm.training.ski_semantic --mode pilot --artifacts-dir artifacts/ski_semantic_v1/pilot_seed42
```

命令只供查核；已存在lock的目錄不可重跑。是否能讓RGB自然關注動作部位仍待驗證，本輪不承諾joint-level localization；primary outcome是120-class分類是否改善。

## 執行紀錄

Semantic smoke已通過：118個held-out paired samples的59-class semantic CE由`3.6461`降至`1.8367`，accuracy由`12.71%`升至`50.85%`。這是輔助語意head，不是120-class主分類收益。Pure KD backward的vision／MIT／teacher梯度皆非零且finite；initial RGB logit parity error為`1.86e-5`，merged RGB-only reload error為`3.04e-6`。Smoke主分類gain為0，不能據此宣稱方法有效。

單一pilot已dispatch至tmux session `ski-semantic-pilot`，PID與命令在`artifacts/ski_semantic_v1/pilot_seed42/dispatch.json`；8,192 train中有4,013 paired樣本用於semantic prealignment與online KD，validation仍為全部12,960。此處不預告勝出，terminal結果以`run_report.json`為準。

Pilot現已完成，dispatcher terminal state為`complete`，run decision為`inconclusive`。Raw120 accuracy由`58.9198%`升至`58.9815%`，gain為`+0.0617` percentage points；baseline答對7,636，新模型答對7,644，修正287個錯誤並新增279個錯誤。MacroF1由`60.4554%`升至`60.6459%`。未達事前`+0.20` percentage points門檻，不選winner、不自動延長epoch或放大full training。

Paired validation答對數由2,774升至2,793（95 fixes／76 regressions），accuracy由`66.3636%`到`66.8182%`；unpaired答對數由4,862降至4,851（192 fixes／203 regressions），accuracy由`55.3759%`到`55.2506%`。這個模式與partial59 guidance改善paired但傷害部分unpaired相容，卻不是因果歸因：沒有matched CE-only新control，不能區分額外CE、semantic prealignment或MSE各自的貢獻。

Pilot held-out semantic CE由`3.6007`降至`1.3434`，59-class輔助accuracy由`11.44%`到`60.59%`，不等同主分類收益。Initial baseline logit parity error為`1.36e-5`，merged RGB-only reload error為`4.05e-6`。Validation IDs與labels已獨立核對既有baseline，12,960 IDs唯一且logits finite；runner snapshot hash符合immutable lock。未讀test。新程式的既有vision與skeleton model checks共7個通過，另有tiny-model semantic parity／vision-gradient檢查；未新增persistent tests。

目前結論是機制可運作，但沒有足夠的分類收益證據支持full training。若繼續，優先分析paired fixes／unpaired regressions與loss gradients，確認如何保護原RGB能力，再决定是否值得下一個改版；不能從activation maps或語意head accuracy決定勝出。

使用者之後同意嘗試direct-decision KD＋unpaired RGB retention，設計、唯讀checkpoint拆分與新pilot結果見[decision distillation紀錄](decision-distillation-v1-plan.md)。新pilot是不同loss與frozen-teacher policy的supervised改版，不改寫本輪結果，也不宣稱官方SKI重現。
