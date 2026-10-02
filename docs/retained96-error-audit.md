# 保留96類：zoom後殘餘錯誤分析

日期：2026-10-01。使用者詢問能否提供部位運動、物件與時間資訊，先以已保存predictions分析下一步的證據；本輪不訓練、不新增inference、不讀test。移除ground-truth屬24個排除類別的評估樣本，但全部120個輸出保留，預測成排除類別仍算錯。

## Full validation

| View | Correct / 9,792 | Accuracy | Fixes / regressions vs whole |
| --- | ---: | ---: | ---: |
| whole | 6671 / 9792 | 68.1270% | 0 / 0 |
| center_zoom70 | 7137 / 9792 | 72.8860% | 1408 / 942 |
| whole_zoom_mean | 7491 / 9792 | 76.5012% | 1093 / 273 |

## 保留類別的最高殘餘錯判率

依whole＋zoom的per-class錯判率由高到低，非依錯誤筆數排序。不同類別有72或144樣本；top confusions包含產品排除類別，避免隱藏這些錯誤。

| Action | Name | Whole correct | Zoom correct | Ensemble correct / n | 最常錯認為 |
| --- | --- | ---: | ---: | ---: | --- |
| A047 | touch neck (neckache) | 24 | 21 | 24 / 72 | A037 wipe face (16)；A045 touch chest (stomachache/heart pain) (5)；A041 sneeze/cough (4) |
| A067 | hush (quiet) | 61 | 51 | 56 / 144 | A105 blow nose (25)；A103 yawn (17)；A079 sniff (smell) (12) |
| A010 | clapping | 27 | 20 | 29 / 72 | A034 rub two hands together (21)；A039 put the palms together (11)；A013 tear up paper (4) |
| A107 | wield knife towards other person | 56 | 58 | 59 / 144 | A109 grab other person's stuff (17)；A120 finger-guessing game (rock-paper-scissors) (13)；A111 step on foot (9) |
| A089 | put something into a bag | 65 | 60 | 65 / 144 | A090 take something out of a bag (73)；A066 juggling table tennis balls (1)；A073 staple book (1) |
| A049 | use a fan (with hand or paper)/feeling warm | 22 | 31 | 33 / 72 | A037 wipe face (10)；A003 brushing teeth (4)；A039 put the palms together (4) |
| A106 | hit other person with something | 54 | 70 | 66 / 144 | A107 wield knife towards other person (25)；A109 grab other person's stuff (12)；A112 high-five (8) |
| A002 | eat meal/snack | 28 | 37 | 35 / 72 | A019 take off glasses (8)；A041 sneeze/cough (6)；A013 tear up paper (5) |
| A041 | sneeze/cough | 25 | 38 | 37 / 72 | A037 wipe face (9)；A033 check time (from watch) (4)；A046 touch back (backache) (4) |
| A030 | typing on a keyboard | 35 | 39 | 38 / 72 | A029 playing with phone/tablet (23)；A012 writing (8)；A011 reading (2) |
| A023 | hand waving | 31 | 38 | 39 / 72 | A049 use a fan (with hand or paper)/feeling warm (7)；A003 brushing teeth (5)；A022 cheer up (4) |
| A032 | taking a selfie | 32 | 38 | 40 / 72 | A031 pointing to something with finger (22)；A001 drink water (3)；A047 touch neck (neckache) (2) |
| A079 | sniff (smell) | 57 | 70 | 80 / 144 | A105 blow nose (16)；A067 hush (quiet) (6)；A075 cutting nails (5) |
| A075 | cutting nails | 69 | 77 | 82 / 144 | A076 cutting paper (using scissors) (14)；A078 open bottle (12)；A073 staple book (9) |
| A108 | knock over other person (hit with body) | 77 | 74 | 82 / 144 | A117 whisper in other person's ear (17)；A111 step on foot (12)；A116 follow other person (8) |

## Temporal診斷的界線

既有480筆balanced subset中，保留96類共384筆。下表是固定views，不為每個sample挑最好答案；不是full validation，也不能據half-window下降否定temporal modeling。

| View | Correct | Accuracy |
| --- | ---: | ---: |
| whole | 267 / 384 | 69.5312% |
| early | 148 / 384 | 38.5417% |
| middle | 224 / 384 | 58.3333% |
| late | 196 / 384 | 51.0417% |
| center_zoom70 | 290 / 384 | 75.5208% |
| three_window_mean | 239 / 384 | 62.2396% |
| whole_zoom_mean | 306 / 384 | 79.6875% |

## 優先診斷候選

A089 put something into a bag在whole＋zoom下仍有73／144筆被判為A090 take something out of a bag，是方向／動作過程的明確混淆，值得先查取幀是否保留放入前後狀態；尚不能據logits證明時間訊號缺失。A047 touch neck仍只有24／72 correct，16筆流向A037 wipe face，值得查看手接觸位置是否在現有解析度清楚可辨。A030 typing on a keyboard有23筆流向A029 playing with phone/tablet，值得查物件與手部細節。這三類分別提供時序、接觸部位與物件診斷候選，不預設採用同一attention target。

## 下一步與推論限制

Saved logits可列出具體錯誤與view收益，不能判定物件不可見、取幀漏掉事件或attention因果。下一步先從高殘餘錯判的保留類別，按sample-ID hash挑固定代表錯誤，查看原始whole／zoom八幀及必要的較密時間取樣，記錄手／物件pixels、context clipping與動作階段是否被保留。只在train設定固定crop／時間policy，再用validation確認；不得用ground-truth action在部署決定ROI，也不得用逐sample最佳view作oracle成績。

若視覺證據支持手／物件細節缺失，優先比較label-independent RGB person／hand ROI的局部view；若關鍵事件跨幀不足，另立temporal sampling或window研究。這些是待驗證方向，既有crop-aware pilot沒有把ensemble收益學進單次推論，因此不能直接承諾新訓練有效。推理是否接受detector與额外forward，需在新實驗前明確決定。

完整per-class metrics、示例錯誤IDs、source hashes與固定temporal subset結果保存在`artifacts/retained96_error_audit_v1/report.json`。此分析不改變原native-attention pilot的inconclusive決策，也不啟動新訓練。
