# 目前產品評估範圍：保留 86 類

2026-10-01：依使用者指示，累計新增排除 A030 打字、A089 放入袋子、A090 拿出袋中物品、A067 噓手勢、A010 拍手、A032 自拍、A079 聞東西、A023 揮手、A107 持刀朝向他人、A106 用物品打人。目前共排除 34 類；後續錯誤診斷不再優先處理這十類。A049 搧風仍保留。

僅排除 ground-truth 屬於這些類別的評估樣本；模型仍輸出 120 類，保留樣本被預測成排除類別仍算錯。沿用既有 validation predictions 重算，沒有新訓練或推理。

| 方法 | 正確／樣本 | 準確率 |
| --- | --- | --- |
| whole | 6171 / 8640 | 71.4236% |
| center_zoom70 | 6587 / 8640 | 76.2384% |
| whole_zoom_mean | 6910 / 8640 | 79.9769% |
| native_attention_epoch2 | 6153 / 8640 | 71.2153% |

排除 source action IDs：A004, A010, A011, A012, A023, A030, A032, A034, A065, A066, A067, A069, A070, A071, A072, A073, A074, A076, A077, A078, A079, A081, A082, A083, A084, A086, A089, A090, A091, A093, A094, A106, A107, A110。

結果保存在 `artifacts/retained86_scope_v1/report.json`。此前 retained96／retained93／retained90 結果保留作歷史紀錄；範圍變更造成的分數差異不代表模型改善。剩餘錯誤排序見 [retained86-remaining-errors.md](retained86-remaining-errors.md)。
