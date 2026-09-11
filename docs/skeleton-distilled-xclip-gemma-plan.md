# Skeleton-distilled XCLIP 接入 Gemma 計畫

日期：2026-09-11。狀態：架構提案；尚未實作 Gemma connector 或 VLM training pipeline。本文件延伸 `docs/skeletonclip-distillation-plan.md`，但不改變其中第一版分類 distillation 的實驗範圍。

## 1. 結論

目前的 XCLIP student 可以在未來接入 Gemma 類 decoder-only LLM，但不能把兩者直接串接後就期待形成有效 VLM。XCLIP 特徵必須先經過 temporal aggregation／resampling 與 projector，轉成 Gemma hidden size 的 visual tokens，之後再用 video–text 或 instruction data 完成跨模態 alignment。

第一版 SkeletonCLIP distillation 並不直接調教 vision encoder。現有 RGB frame 先經 frozen XCLIP vision encoder 與 frozen visual projection，形成 `[8, 512]` pre-MIT features；teacher-logit KL 只更新 MIT LoRA 與 classification head。因此可重用的是受到 skeleton teacher 影響的 temporal module，不應稱為「teacher-distilled vision encoder」。

預期效益主要落在人體動作、時間動態與 NTU 類型的 action understanding。它不保證改善物件辨識、OCR、場景細節、通用 visual question answering 或自由描述，也不保證優於未蒸餾的 XCLIP。是否改善必須在相同 Gemma、connector、訓練資料與計算預算下，以受控實驗回答。

## 2. 現有 student 能提供什麼

現有分類資料流如下：

```text
RGB video
→ deterministic 8-frame sampling
→ frozen XCLIP vision encoder
→ frozen visual projection
→ [8, 512] frame features
→ MIT + LoRA
→ pooled video representation
→ 120-class head
```

SkeletonCLIP teacher 以 class-logit KL 監督 MIT LoRA 與 head。這種 supervision 傳遞的是 teacher 對動作類別相對關係的判斷，而不是 joints、patch localization 或一般語言語意。分類 head 的 120-class logits 不適合作為 Gemma visual input；Gemma path 應從 classifier 前的 temporal features 分支。

第一版 teacher 目前只規劃涵蓋 A001–A060，且 RGB–skeleton 配對不完整。即使 distillation 改善 120-class classification，也不能假設這個效果會自動轉移到 A061–A120、開放詞彙動作描述或 VLM 問答。

## 3. 建議的 Gemma 接法

建議保留分類與 VLM 兩條獨立輸出路徑：

```text
Classification path:
[8, 512] frame features → distilled MIT → pooled feature → classifier

Gemma path:
[8, 512] frame features → distilled MIT → temporal tokens
→ temporal resampler → N visual tokens
→ projector → Gemma hidden-size tokens
→ Gemma
```

Gemma path 至少需要以下元件：

- **Temporal token interface：** MIT 應能回傳 classifier pooling 前的 token sequence，而不只回傳單一 pooled vector。
- **Temporal resampler：** 將八個或更多 frame／temporal tokens 壓縮或轉換為固定數量 `N` 的 visual tokens。第一版可比較 identity／mean pooling 與小型 attention resampler，但不要一開始引入過大的 connector。
- **Projector：** 使用 Linear、兩層 MLP 或 gated MLP，將 XCLIP 的 512 維特徵投影至所選 Gemma checkpoint 的 hidden size。相同維度不代表相同 embedding space，仍必須訓練 alignment。
- **Multimodal input contract：** 明確定義 visual token 插入位置、special tokens、attention mask、position handling、最大 token 數與 batch padding。
- **VLM training objective：** 使用 video caption、action description、QA 或 instruction-response data 做 autoregressive language loss。Classification KD checkpoint 只能作初始化，不能取代這一步。

若只把整段影片 pool 成一個 visual token，實作最簡單，但容易丟失動作階段、多人互動及細節。第一版建議保留 `4–16` 個 temporal visual tokens，實際數量只用 validation 選擇。

## 4. 訓練階段

### Stage A：建立可重現的 connector baseline

固定原始 XCLIP vision encoder與MIT，凍結 Gemma，只訓練 resampler／projector。先確認 shape、mask、checkpoint reload 與 generation 能正常工作，再評估是否需要調整更多參數。

### Stage B：比較 distilled temporal module

在完全相同的 Gemma、connector architecture、VLM data、optimizer、steps 與 seed 下，比較原始 XCLIP、CE-only MIT LoRA 與 Skeleton teacher KD MIT LoRA。三組 connector 均應獨立從相同初始化訓練，避免某組享有額外 alignment steps。

### Stage C：Gemma LoRA

若 frozen Gemma 的 alignment 能力不足，再對 Gemma attention／MLP 的明確 target modules 加入 LoRA。這是 VLM adaptation，不應與 Skeleton KD 的效果混為同一個變因；先完成 Stage B，再以獨立 ablation 判斷 Gemma LoRA 的增益。

### Stage D：vision upper-layer adaptation

只有 distilled MIT 在 action-oriented VLM validation 上呈現穩定增益後，才考慮對 XCLIP vision encoder 最後數個 blocks 加 LoRA，讓 teacher loss實際影響 spatial visual features。此階段不能繼續完全依賴既有離線 `[8, 512]` cache，必須重新執行 RGB vision forward。

即使 vision upper-layer LoRA 有效，class-level KD 仍未直接監督 joint regions。若目標是 joint-focused VLM，需要另外保留 patch tokens，建立 skeleton-to-RGB joint alignment，並以 localization、joint-region masking 或 occlusion 評估提供空間證據。

### Stage E：language-aligned distillation

若最終目標是開放式描述與問答，應獨立研究 feature／relation distillation：讓 RGB temporal tokens 對齊 SkeletonCLIP text-aligned features，或在共享 text similarity space 中蒸餾，而不只模仿 60-class logits。這條路比分類 KD 更直接服務 Gemma，但需明確鎖定 text encoder、prompt、normalization、projection 與 class vocabulary，並保留 class-logit KD 作為 control。

## 5. 受控實驗矩陣

核心比較如下：

| Variant | XCLIP／MIT 初始化 | Skeleton supervision | Gemma alignment |
| --- | --- | --- | --- |
| A | 原始 pretrained XCLIP | 無 | 相同 connector／VLM recipe |
| B | tuned CE-only MIT LoRA | 無 | 相同 connector／VLM recipe |
| C | teacher-logit KD MIT LoRA | A001–A060 class KD | 相同 connector／VLM recipe |
| D | vision upper-layer LoRA + KD | class KD 可進入 vision layers | 只在 C 有效後測試 |
| E | language-aligned student | feature／similarity alignment | 獨立 ablation |

不得只比較 C 與原始 XCLIP；B 是必要 control，用來區分一般 supervised action tuning 與 skeleton teacher knowledge。不得用 test split 選 connector、visual-token 數、checkpoint、KD temperature 或 LoRA targets。

## 6. 評估方式

至少將評估拆成兩組，避免 action accuracy 掩蓋通用能力退化：

- **Action-oriented：** 動作分類、動作描述、temporal-order QA、多人互動 QA，並分列 A001–A060／A061–A120、paired／unpaired subsets。
- **General visual-language：** 物件、場景、屬性、計數、OCR 或其他與 skeleton 無直接關係的題型，用來檢查 catastrophic specialization。

除 task accuracy／generation metrics 外，記錄 trainable parameters、visual token 數、影片 latency、GPU memory、training steps 與資料量。自由生成結果應搭配可重現的 automatic metrics 與固定樣本 qualitative review，不以少數成功案例宣稱整體改善。

只有 C 相對 A 與 B 在預先定義的 action-oriented metrics 上跨 seeds 穩定改善，且 general visual-language 指標沒有不可接受的退化時，才能主張 skeleton teacher 對 Gemma VLM 有實際價值。

## 7. Checkpoint 與介面要求

Distilled classifier checkpoint 應保存 base XCLIP identity、MIT LoRA state、feature stage、token shape、frame-sampling policy 與 teacher provenance。Gemma connector checkpoint 另存 Gemma model ID／revision、tokenizer revision、projector／resampler state、visual token contract、VLM training data identity 與 trainable-module whitelist。

不要將 120-class classification head 綁入 Gemma runtime。部署時可只載入 XCLIP vision components、所選 MIT／LoRA、resampler、projector 與 Gemma；SkeletonCLIP teacher、skeleton input、teacher logits cache及KD loss維持 training-only。

介面驗收至少包含輸入／輸出 shape、attention mask、不同 batch padding、mixed precision、checkpoint round trip、deterministic eval、分類 path 不受 Gemma path 影響，以及不載入 skeleton teacher 的 RGB-only generation smoke test。

## 8. 風險與判讀邊界

- **任務過度特化：** NTU action KD 可能提升封閉集合分類，同時降低一般視覺語意。必須同時報告 action 與 general benchmarks。
- **Teacher coverage 偏差：** A001–A060 與部分 participant／setup 的監督不能代表完整 NTU120 或真實世界影片。
- **單 token bottleneck：** 過早 pooling 可能使 Gemma無法推理動作順序與互動細節。
- **Alignment 混淆：** Connector 或 Gemma LoRA 的訓練差異可能大於 distilled encoder 差異，因此所有 variants 必須使用一致 recipe。
- **語言能力誤歸因：** 生成品質提升可能來自 Gemma 或 instruction data，不一定來自 SkeletonCLIP；CE-only 與原始 XCLIP controls 不可省略。
- **空間主張不足：** Class-logit KD 沒有 joint localization supervision，不能由 QA 或分類提升推論模型更注意人體關節。

## 9. 建議決策順序

1. 完成目前 SkeletonCLIP class-logit distillation，先確認 C 相對 B 是否改善 RGB action classification。
2. 讓 MIT 暴露 pooling 前 temporal tokens，定義穩定的 VLM feature contract。
3. 以小型 resampler／projector 建立原始 XCLIP + frozen Gemma baseline。
4. 在相同 alignment recipe 下完成 A／B／C 比較，分開評估 action 與 general visual-language 能力。
5. 只有 C 穩定有效時，再投入 vision upper-layer LoRA、更多 frames 或 language-aligned distillation。

這個順序能先回答最小且重要的問題：Skeleton teacher 是否真的為 Gemma 可使用的影片表示帶來額外資訊，而不只是讓既有 120-class classifier 更適合 NTU labels。
