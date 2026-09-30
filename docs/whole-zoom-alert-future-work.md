# Whole＋zoom、threshold 與 discard labels：後續工作

記錄日期：2026-09-30。這份文件是研究／產品方向筆記，不是runtime configuration，也不代表啟用警報或更換模型。

## Whole＋zoom：目前正在試，是否採用尚未決定

現有RGB120 checkpoint對同一支影片的相同8幀各做一次forward：whole沿用原本preprocessing；zoom先取畫面中央、邊長為原影像短邊70%的正方形，再經同一processor。兩個views的120-class softmax probabilities等權平均；不是平均logits，不是兩個模型，不重新訓練。

480筆subset的whole為60.83%、zoom為68.75%、whole＋zoom為72.29%。使用者核准後，12,960筆完整validation確認已dispatch至tmux `whole-zoom-validation`，產物與log在`artifacts/whole_zoom_validation_v1/`；不讀test、不自動替換部署模型。下一步依完整結果檢查各類別／事件收益、裁切風險及運算成本，再決定是否採用或研究crop-trained模型。

完整validation已完成：whole58.92%、zoom63.91%、whole＋zoom66.94%，淨增加1,039筆正確（+8.02 percentage points）。使用者接著核准短crop-aware training pilot，設定與執行契約見[crop-aware-v1-plan.md](crop-aware-v1-plan.md)；threshold與discard-output仍未接入runtime。

Crop-aware pilot也已完成：whole58.96%、zoom64.31%、ensemble67.14%；ensemble比原始多26筆正確，但attention七類349→343，判為inconclusive、不推薦替換。現階段沒有全量raw120超過70%的结果；72.29%僅是480筆subset，66.94%是多一次forward的ensemble收益，taxonomy合併成績屬不同任務，不能混稱為單次推論模型突破。完整結果與判讀見[crop-aware結果](crop-aware-v1-plan.md#completed-pilot-results-and-decision)。

## Threshold：後續方向，尚未採用警報門檻

候選事件為A042失衡、A043跌倒、A044頭部不適姿態、A045胸部不適姿態、A046背部不適姿態、A047頸部不適姿態與A048噁心／嘔吐相關動作。需逐事件評估，不把所有姿態直接當成已確認醫療緊急事件。

現有runner會產出離線threshold分析：每類按sample-ID hash切成calibration／evaluation halves，探索約95% calibration recall的門檻以及固定門檻比較。這是分析輸出，不是runtime alarm設定；whole-only門檻不能直接套用whole＋zoom。Checkpoint已用全部validation選擇，兩半亦可能含相關subject／camera views，因此不構成獨立部署驗證。

未來工作是依事件分開決定可接受的漏報與誤報，蒐集連續正常活動／事件影片，量測false alarms per hour與偵測延遲，再研究分層通知、人工複核、hysteresis／cooldown等policy。尚未選定deployment thresholds，也沒有啟用live alarms。

## Discard labels：已決定產品scope，尚未接入模型／runtime

使用者新增不要求獨立產品輸出的七類為A086、A034、A011、A078、A012、A110、A004。先前已接受的17類為A065、A066、A069、A070、A071、A072、A073、A074、A076、A077、A081、A082、A083、A084、A091、A093、A094；合計24類。

本次run仍保留全部120類的樣本、logits與probability mass，沒有刪除七類，也沒有移除logits後重新normalize。Report中的`product_excluded24`只作分組統計，不改變prediction或事件分數。未來產品輸出如何隱藏／合併這些類別，以及是否需要新訓練目標，尚未實作；歷史reduced-v1不變。

建議仍保留discarded actions作事件negative examples，以檢查日常動作是否觸發誤報；discard不代表動作安全，尤其A110僅是不在本輪醫療scope。A075未被新增排除，也不自行推定其他未點名動作應刪除。
