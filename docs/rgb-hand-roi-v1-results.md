# RGB 手部與互動區域局部推理：結果

固定 176 筆，沿用第一階段選樣。定位與裁切僅依 RGB，使用原 patch32 baseline，不訓練、不讀 test split；設計見 [hand-roi-v1-audit.md](hand-roi-v1-audit.md)。

## 176 筆診斷結果

| View | 目標三類正確／48 | 混淆類別正確／128 | 全部正確／176 | 目標修正／退步 |
| --- | --- | --- | --- | --- |
| whole_zoom_mean | 19 | 85 | 104 | 0 / 0 |
| hands | 26 | 79 | 105 | 13 / 6 |
| interaction | 26 | 87 | 113 | 14 / 7 |
| whole_zoom_hands_mean | 21 | 95 | 116 | 4 / 2 |
| whole_zoom_interaction_mean | 23 | 98 | 121 | 6 / 2 |

| 類別 | Whole＋zoom | Hands | Interaction | Whole＋zoom＋hands | Whole＋zoom＋interaction |
| --- | --- | --- | --- | --- | --- |
| A002 eat meal/snack | 8 / 16 | 9 / 16 | 11 / 16 | 7 / 16 | 9 / 16 |
| A041 sneeze/cough | 5 / 16 | 11 / 16 | 8 / 16 | 8 / 16 | 8 / 16 |
| A049 use a fan (with hand or paper)/feeling warm | 6 / 16 | 6 / 16 | 7 / 16 | 6 / 16 | 6 / 16 |
| A047 touch neck (neckache) | 4 / 16 | 2 / 16 | 1 / 16 | 4 / 16 | 4 / 16 |

Fallback 共 0 筆；whole logit 最大 parity 誤差為 2.8073788e-05。診斷推理耗時 159.5 秒（不含模型載入）；GPU 與既有 patch16 訓練共用。

### 混淆類別流向目標類的數量

| View | 非目標類被預測為吃東西／咳嗽／搧風 |
| --- | --- |
| whole_zoom_mean | 5 / 128 |
| hands | 5 / 128 |
| interaction | 2 / 128 |
| whole_zoom_hands_mean | 3 / 128 |
| whole_zoom_interaction_mean | 4 / 128 |

這是 top1 的負例混淆數，不是部署警報誤報率。

[實際輸入對照頁](../artifacts/rgb_hand_roi_audit_v1/index.html)。

## 與 privileged 定位比較

| 類別 | 原 whole＋zoom | Skeleton interaction 合併 | RGB interaction 合併 |
| --- | --- | --- | --- |
| A002 | 8 / 16 | 10 / 16 | 9 / 16 |
| A041 | 5 / 16 | 9 / 16 | 8 / 16 |
| A049 | 6 / 16 | 6 / 16 | 6 / 16 |
| A047 | 4 / 16 | 4 / 16 | 4 / 16 |

RGB CPU 定位 207.4 秒，平均 1.18 秒／影片；這是離線八幀診斷的耗時，不是串流端到端 latency。定位 fallback 0／176，全部仍參與分類比較。

逐筆修正／退步保存在 `artifacts/rgb_hand_roi_audit_v1/case_changes.json`。同批選樣、相同 labels 與 IDs、whole／zoom logits 對齊、RGB 推理未記錄 skeleton hash，以及 fallback 計數均已核對。

## 結論與後續方向

這輪完成了「先測局部輸入」的推理實驗與 RGB 自動定位確認，沒有啟動局部分支或手部 attention 的新訓練。固定 whole＋zoom＋RGB interaction 在 176 筆由 104／176（59.09%）提升至 121／176（68.75%），修正 21、退步 4；目標三類由 19／48（39.58%）提升至 23／48（47.92%），混淆類別由 85／128（66.41%）提升至 98／128（76.56%）。這些是固定小樣本上的改善訊號，不是 86 類 full validation 的新準確率，也不是 hand attention supervision 的訓練收益。

收益主要集中在咳嗽（5→8／16），吃東西僅淨增一筆（8→9／16）；搧風（6→6／16）與摸頸（4→4／16）尚未得到合併收益。摸頸的局部單獨推理更差（hands2／16、interaction1／16），因此不支持「只盯手就能救摸頸」。所有類別使用同一套規則，沒有依類別挑不同 view 或以逐筆最佳答案報分。

這輪支持保留 whole 上下文、加入互動區域的方向，優先擴大咳嗽及其混淆類的 RGB 診斷，並觀察吃東西的微小收益能否重現。進一步訓練前仍需較大的固定評估確認收益與退步，不應直接以每類 16 筆的結果替換 baseline。搧風與摸頸應先檢查動作過程、接觸位置及分類器對局部畫面的適應，而非繼續盲目縮小裁切。剪指甲未涵蓋在本輪 paired 選樣中；RGB localizer 已不需其 skeleton 座標，後續可另建固定樣本診斷。
