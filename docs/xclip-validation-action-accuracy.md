# XCLIP validation action accuracy：由低至高

> 此文件是較舊的 52.11% MIT-only CE control。若要依目前最佳已保存的 vision-adaptation model審查 action scope，請使用 `docs/xclip-vision-lora-validation-action-accuracy.md`。

日期：2026-09-29。這份清單用於人工檢視哪些 action 對目前 RGB model 過於含糊、太依賴小物件，或不值得納入目標任務；它不是自動刪除類別的決策。

## Evidence contract

- Model：Phase 2 CE-only control，亦即 tuned XCLIP MIT q/v LoRA student recipe，不含 BlockGCN KD。
- Split：canonical validation，共 12,960 筆；沒有使用 test predictions，因此後續依此清單調整 action scope 不會直接讀取 test labels。
- Overall validation accuracy：52.1065%。
- Prediction source：`artifacts/skeletonclip_phase2/control_seed42/validation_predictions.json`，SHA256 `d22a84b30c69721b1c6ccebda5b0ca66e02733e3b34fc8e22a6218157afc98e6`。
- 排序：per-class top-1 accuracy 由低至高，同 accuracy 再依 action ID 排序。A001–A060 每類 72 筆，A061–A120 每類 144 筆，因此跨類別比較百分比時也應查看分母。
- Action names：[NTU RGB+D authors repository](https://github.com/shahroudy/NTURGB-D#action-classes) 的官方 class list；文字只用於可讀性，計算以 numeric class ID 為準。

## 使用方式與限制

低 accuracy 不必然代表「不需要模型知道」。它可能表示八幀 uniform sampling 看不到短暫關鍵動作、動作依賴小物件、兩個 label 本身近義、participant／camera domain shift，或目前 model capacity 不足。建議先依產品需求標記 keep／remove／merge／needs-review，再用固定的新 class set 重新建立 head、split summary 與 baseline；不要只把舊 120-class prediction 中被刪類別忽略後直接宣稱 accuracy 提升。

若移除或合併 action，必須在讀取 test-based新結果之前鎖定 mapping，並同時報告原 120-class baseline與新任務結果。最常見誤判只列前三名，括號是誤判樣本數；它能協助辨識值得 merge 的 label pair，但不是 merge 的充分依據。

## Ranking

| Rank | Action | Name | Correct / total | Accuracy | Most common wrong predictions |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | A073 | staple book | 11/144 | 7.64% | A076 cutting paper (using scissors) (31), A074 counting money (17), A075 cutting nails (13) |
| 2 | A084 | play magic cube | 19/144 | 13.19% | A074 counting money (23), A075 cutting nails (23), A076 cutting paper (using scissors) (20) |
| 3 | A072 | make victory sign | 20/144 | 13.89% | A071 make ok sign (24), A067 hush (quiet) (16), A069 thumb up (10) |
| 4 | A071 | make ok sign | 22/144 | 15.28% | A067 hush (quiet) (22), A072 make victory sign (17), A069 thumb up (11) |
| 5 | A082 | fold paper | 24/144 | 16.67% | A076 cutting paper (using scissors) (19), A091 open a box (17), A075 cutting nails (15) |
| 6 | A083 | ball up paper | 28/144 | 19.44% | A082 fold paper (15), A074 counting money (12), A084 play magic cube (11) |
| 7 | A078 | open bottle | 31/144 | 21.53% | A091 open a box (15), A086 apply cream on hand back (10), A083 ball up paper (9) |
| 8 | A086 | apply cream on hand back | 31/144 | 21.53% | A075 cutting nails (14), A078 open bottle (12), A091 open a box (10) |
| 9 | A077 | snapping fingers | 32/144 | 22.22% | A093 shake fist (14), A072 make victory sign (14), A069 thumb up (9) |
| 10 | A093 | shake fist | 32/144 | 22.22% | A067 hush (quiet) (11), A070 thumb down (10), A103 yawn (10) |
| 11 | A074 | counting money | 33/144 | 22.92% | A075 cutting nails (21), A076 cutting paper (using scissors) (17), A084 play magic cube (15) |
| 12 | A011 | reading | 17/72 | 23.61% | A029 playing with phone/tablet (24), A012 writing (13), A013 tear up paper (4) |
| 13 | A012 | writing | 17/72 | 23.61% | A029 playing with phone/tablet (34), A011 reading (7), A030 typing on a keyboard (5) |
| 14 | A069 | thumb up | 36/144 | 25.00% | A070 thumb down (17), A071 make ok sign (13), A072 make victory sign (9) |
| 15 | A081 | toss a coin | 38/144 | 26.39% | A077 snapping fingers (21), A096 cross arms (10), A069 thumb up (8) |
| 16 | A004 | brushing hair | 20/72 | 27.78% | A044 touch head (headache) (15), A003 brushing teeth (6), A037 wipe face (4) |
| 17 | A041 | sneeze/cough | 20/72 | 27.78% | A025 reach into pocket (8), A005 drop (7), A018 wear on glasses (5) |
| 18 | A049 | use a fan (with hand or paper)/feeling warm | 20/72 | 27.78% | A044 touch head (headache) (6), A037 wipe face (5), A010 clapping (5) |
| 19 | A079 | sniff (smell) | 40/144 | 27.78% | A105 blow nose (13), A078 open bottle (10), A077 snapping fingers (9) |
| 20 | A107 | wield knife towards other person | 40/144 | 27.78% | A110 shoot at other person with a gun (15), A120 finger-guessing game (rock-paper-scissors) (14), A106 hit other person with something (12) |
| 21 | A047 | touch neck (neckache) | 21/72 | 29.17% | A003 brushing teeth (9), A037 wipe face (7), A044 touch head (headache) (6) |
| 22 | A045 | touch chest (stomachache/heart pain) | 22/72 | 30.56% | A048 nausea or vomiting condition (26), A035 nod head/bow (4), A036 shake head (3) |
| 23 | A070 | thumb down | 44/144 | 30.56% | A069 thumb up (17), A067 hush (quiet) (7), A097 arm circles (6) |
| 24 | A023 | hand waving | 23/72 | 31.94% | A038 salute (8), A049 use a fan (with hand or paper)/feeling warm (5), A003 brushing teeth (5) |
| 25 | A075 | cutting nails | 46/144 | 31.94% | A076 cutting paper (using scissors) (15), A086 apply cream on hand back (11), A084 play magic cube (11) |
| 26 | A003 | brushing teeth | 24/72 | 33.33% | A037 wipe face (8), A036 shake head (5), A044 touch head (headache) (3) |
| 27 | A034 | rub two hands together | 24/72 | 33.33% | A010 clapping (17), A029 playing with phone/tablet (8), A039 put the palms together (7) |
| 28 | A002 | eat meal/snack | 25/72 | 34.72% | A036 shake head (7), A019 take off glasses (5), A041 sneeze/cough (5) |
| 29 | A106 | hit other person with something | 50/144 | 34.72% | A107 wield knife towards other person (25), A110 shoot at other person with a gun (11), A112 high-five (10) |
| 30 | A110 | shoot at other person with a gun | 51/144 | 35.42% | A115 take a photo of other person (27), A107 wield knife towards other person (14), A120 finger-guessing game (rock-paper-scissors) (13) |
| 31 | A010 | clapping | 26/72 | 36.11% | A034 rub two hands together (16), A039 put the palms together (8), A003 brushing teeth (2) |
| 32 | A028 | make a phone call/answer phone | 26/72 | 36.11% | A003 brushing teeth (6), A037 wipe face (4), A005 drop (4) |
| 33 | A066 | juggling table tennis balls | 52/144 | 36.11% | A086 apply cream on hand back (9), A091 open a box (8), A077 snapping fingers (6) |
| 34 | A076 | cutting paper (using scissors) | 52/144 | 36.11% | A073 staple book (14), A074 counting money (13), A075 cutting nails (13) |
| 35 | A065 | tennis bat swing | 53/144 | 36.81% | A081 toss a coin (10), A066 juggling table tennis balls (8), A070 thumb down (7) |
| 36 | A067 | hush (quiet) | 53/144 | 36.81% | A103 yawn (13), A105 blow nose (13), A079 sniff (smell) (9) |
| 37 | A032 | taking a selfie | 27/72 | 37.50% | A031 pointing to something with finger (11), A038 salute (5), A033 check time (from watch) (5) |
| 38 | A090 | take something out of a bag | 54/144 | 37.50% | A089 put something into a bag (37), A091 open a box (8), A088 take off bag (4) |
| 39 | A068 | flick hair | 55/144 | 38.19% | A093 shake fist (13), A061 put on headphone (11), A105 blow nose (8) |
| 40 | A091 | open a box | 55/144 | 38.19% | A082 fold paper (17), A078 open bottle (9), A073 staple book (9) |
| 41 | A001 | drink water | 29/72 | 40.28% | A002 eat meal/snack (9), A028 make a phone call/answer phone (6), A003 brushing teeth (3) |
| 42 | A037 | wipe face | 29/72 | 40.28% | A044 touch head (headache) (8), A019 take off glasses (7), A018 wear on glasses (5) |
| 43 | A038 | salute | 29/72 | 40.28% | A047 touch neck (neckache) (4), A004 brushing hair (4), A036 shake head (4) |
| 44 | A039 | put the palms together | 30/72 | 41.67% | A010 clapping (9), A034 rub two hands together (5), A040 cross hands in front (say stop) (5) |
| 45 | A046 | touch back (backache) | 30/72 | 41.67% | A047 touch neck (neckache) (5), A036 shake head (5), A010 clapping (4) |
| 46 | A105 | blow nose | 60/144 | 41.67% | A067 hush (quiet) (17), A103 yawn (9), A079 sniff (smell) (7) |
| 47 | A063 | shoot at the basket | 61/144 | 42.36% | A104 stretch oneself (13), A094 throw up cap/hat (9), A093 shake fist (7) |
| 48 | A025 | reach into pocket | 32/72 | 44.44% | A036 shake head (10), A005 drop (6), A041 sneeze/cough (3) |
| 49 | A089 | put something into a bag | 65/144 | 45.14% | A090 take something out of a bag (31), A087 put on bag (10), A091 open a box (6) |
| 50 | A085 | apply cream on face | 66/144 | 45.83% | A079 sniff (smell) (9), A105 blow nose (7), A077 snapping fingers (6) |
| 51 | A031 | pointing to something with finger | 34/72 | 47.22% | A023 hand waving (4), A049 use a fan (with hand or paper)/feeling warm (4), A040 cross hands in front (say stop) (4) |
| 52 | A017 | take off a shoe | 35/72 | 48.61% | A016 wear a shoe (24), A006 pickup (6), A048 nausea or vomiting condition (4) |
| 53 | A103 | yawn | 71/144 | 49.31% | A067 hush (quiet) (20), A068 flick hair (7), A105 blow nose (7) |
| 54 | A109 | grab other person's stuff | 72/144 | 50.00% | A118 exchange things with other person (11), A107 wield knife towards other person (10), A111 step on foot (8) |
| 55 | A019 | take off glasses | 37/72 | 51.39% | A018 wear on glasses (20), A013 tear up paper (3), A025 reach into pocket (3) |
| 56 | A030 | typing on a keyboard | 37/72 | 51.39% | A029 playing with phone/tablet (19), A012 writing (9), A011 reading (5) |
| 57 | A050 | punching/slapping other person | 37/72 | 51.39% | A052 pushing other person (13), A057 touch other person's pocket (7), A053 pat on back of other person (5) |
| 58 | A108 | knock over other person (hit with body) | 74/144 | 51.39% | A117 whisper in other person's ear (27), A111 step on foot (10), A119 support somebody with hand (9) |
| 59 | A013 | tear up paper | 38/72 | 52.78% | A011 reading (7), A010 clapping (5), A005 drop (3) |
| 60 | A094 | throw up cap/hat | 78/144 | 54.17% | A065 tennis bat swing (7), A093 shake fist (7), A081 toss a coin (5) |
| 61 | A007 | throw | 40/72 | 55.56% | A042 staggering (10), A031 pointing to something with finger (5), A023 hand waving (5) |
| 62 | A029 | playing with phone/tablet | 40/72 | 55.56% | A012 writing (9), A030 typing on a keyboard (5), A033 check time (from watch) (4) |
| 63 | A056 | giving something to other person | 40/72 | 55.56% | A058 handshaking (9), A057 touch other person's pocket (6), A053 pat on back of other person (5) |
| 64 | A104 | stretch oneself | 80/144 | 55.56% | A097 arm circles (18), A063 shoot at the basket (16), A095 hands up (both hands) (10) |
| 65 | A061 | put on headphone | 83/144 | 57.64% | A062 take off headphone (15), A068 flick hair (13), A087 put on bag (7) |
| 66 | A097 | arm circles | 84/144 | 58.33% | A098 arm swings (21), A104 stretch oneself (8), A096 cross arms (6) |
| 67 | A033 | check time (from watch) | 43/72 | 59.72% | A005 drop (7), A047 touch neck (neckache) (4), A040 cross hands in front (say stop) (3) |
| 68 | A058 | handshaking | 43/72 | 59.72% | A056 giving something to other person (10), A054 point finger at the other person (9), A053 pat on back of other person (5) |
| 69 | A095 | hands up (both hands) | 86/144 | 59.72% | A104 stretch oneself (16), A097 arm circles (7), A094 throw up cap/hat (4) |
| 70 | A096 | cross arms | 87/144 | 60.42% | A086 apply cream on hand back (8), A069 thumb up (7), A098 arm swings (6) |
| 71 | A120 | finger-guessing game (rock-paper-scissors) | 87/144 | 60.42% | A112 high-five (9), A110 shoot at other person with a gun (9), A117 whisper in other person's ear (6) |
| 72 | A042 | staggering | 44/72 | 61.11% | A005 drop (5), A017 take off a shoe (5), A016 wear a shoe (4) |
| 73 | A115 | take a photo of other person | 88/144 | 61.11% | A112 high-five (13), A120 finger-guessing game (rock-paper-scissors) (8), A110 shoot at other person with a gun (7) |
| 74 | A044 | touch head (headache) | 45/72 | 62.50% | A047 touch neck (neckache) (8), A004 brushing hair (3), A003 brushing teeth (3) |
| 75 | A048 | nausea or vomiting condition | 46/72 | 63.89% | A016 wear a shoe (11), A035 nod head/bow (4), A045 touch chest (stomachache/heart pain) (3) |
| 76 | A102 | side kick | 92/144 | 63.89% | A100 butt kicks (kick backward) (25), A065 tennis bat swing (5), A092 move heavy objects (3) |
| 77 | A062 | take off headphone | 94/144 | 65.28% | A061 put on headphone (12), A068 flick hair (8), A097 arm circles (4) |
| 78 | A054 | point finger at the other person | 48/72 | 66.67% | A058 handshaking (9), A053 pat on back of other person (8), A056 giving something to other person (4) |
| 79 | A119 | support somebody with hand | 96/144 | 66.67% | A116 follow other person (24), A108 knock over other person (hit with body) (11), A092 move heavy objects (2) |
| 80 | A099 | running on the spot | 97/144 | 67.36% | A102 side kick (10), A100 butt kicks (kick backward) (6), A088 take off bag (4) |
| 81 | A112 | high-five | 97/144 | 67.36% | A115 take a photo of other person (11), A117 whisper in other person's ear (8), A110 shoot at other person with a gun (8) |
| 82 | A016 | wear a shoe | 49/72 | 68.06% | A006 pickup (10), A017 take off a shoe (7), A002 eat meal/snack (1) |
| 83 | A064 | bounce ball | 99/144 | 68.75% | A105 blow nose (7), A088 take off bag (5), A089 put something into a bag (4) |
| 84 | A018 | wear on glasses | 50/72 | 69.44% | A019 take off glasses (5), A044 touch head (headache) (4), A013 tear up paper (2) |
| 85 | A040 | cross hands in front (say stop) | 50/72 | 69.44% | A047 touch neck (neckache) (4), A049 use a fan (with hand or paper)/feeling warm (4), A037 wipe face (3) |
| 86 | A087 | put on bag | 100/144 | 69.44% | A088 take off bag (10), A090 take something out of a bag (5), A079 sniff (smell) (3) |
| 87 | A117 | whisper in other person's ear | 100/144 | 69.44% | A111 step on foot (7), A108 knock over other person (hit with body) (7), A119 support somebody with hand (6) |
| 88 | A027 | jump up | 51/72 | 70.83% | A035 nod head/bow (4), A006 pickup (4), A042 staggering (3) |
| 89 | A111 | step on foot | 102/144 | 70.83% | A108 knock over other person (hit with body) (8), A116 follow other person (7), A107 wield knife towards other person (6) |
| 90 | A005 | drop | 52/72 | 72.22% | A036 shake head (8), A025 reach into pocket (3), A049 use a fan (with hand or paper)/feeling warm (3) |
| 91 | A057 | touch other person's pocket | 52/72 | 72.22% | A053 pat on back of other person (7), A056 giving something to other person (3), A058 handshaking (3) |
| 92 | A100 | butt kicks (kick backward) | 105/144 | 72.92% | A102 side kick (11), A099 running on the spot (5), A080 squat down (3) |
| 93 | A113 | cheers and drink | 105/144 | 72.92% | A112 high-five (9), A115 take a photo of other person (7), A110 shoot at other person with a gun (4) |
| 94 | A088 | take off bag | 106/144 | 73.61% | A087 put on bag (9), A090 take something out of a bag (6), A098 arm swings (2) |
| 95 | A098 | arm swings | 106/144 | 73.61% | A096 cross arms (7), A070 thumb down (5), A065 tennis bat swing (5) |
| 96 | A015 | take off jacket | 54/72 | 75.00% | A005 drop (2), A024 kicking something (2), A089 put something into a bag (1) |
| 97 | A026 | hopping (one foot jumping) | 55/72 | 76.39% | A024 kicking something (7), A042 staggering (5), A027 jump up (3) |
| 98 | A035 | nod head/bow | 55/72 | 76.39% | A016 wear a shoe (8), A048 nausea or vomiting condition (4), A041 sneeze/cough (2) |
| 99 | A118 | exchange things with other person | 110/144 | 76.39% | A120 finger-guessing game (rock-paper-scissors) (11), A109 grab other person's stuff (8), A106 hit other person with something (2) |
| 100 | A021 | take off a hat/cap | 57/72 | 79.17% | A019 take off glasses (2), A049 use a fan (with hand or paper)/feeling warm (2), A020 put on a hat/cap (2) |
| 101 | A101 | cross toe touch | 114/144 | 79.17% | A092 move heavy objects (23), A064 bounce ball (3), A080 squat down (3) |
| 102 | A024 | kicking something | 58/72 | 80.56% | A026 hopping (one foot jumping) (6), A005 drop (3), A027 jump up (2) |
| 103 | A022 | cheer up | 59/72 | 81.94% | A004 brushing hair (2), A040 cross hands in front (say stop) (2), A023 hand waving (2) |
| 104 | A051 | kicking other person | 59/72 | 81.94% | A057 touch other person's pocket (8), A050 punching/slapping other person (3), A048 nausea or vomiting condition (1) |
| 105 | A053 | pat on back of other person | 59/72 | 81.94% | A056 giving something to other person (4), A057 touch other person's pocket (4), A058 handshaking (2) |
| 106 | A006 | pickup | 60/72 | 83.33% | A016 wear a shoe (4), A017 take off a shoe (3), A035 nod head/bow (2) |
| 107 | A020 | put on a hat/cap | 61/72 | 84.72% | A049 use a fan (with hand or paper)/feeling warm (2), A038 salute (2), A004 brushing hair (1) |
| 108 | A052 | pushing other person | 61/72 | 84.72% | A050 punching/slapping other person (4), A055 hugging other person (4), A056 giving something to other person (2) |
| 109 | A008 | sitting down | 62/72 | 86.11% | A017 take off a shoe (4), A027 jump up (1), A016 wear a shoe (1) |
| 110 | A036 | shake head | 62/72 | 86.11% | A005 drop (5), A038 salute (2), A053 pat on back of other person (1) |
| 111 | A092 | move heavy objects | 124/144 | 86.11% | A101 cross toe touch (11), A080 squat down (3), A064 bounce ball (2) |
| 112 | A116 | follow other person | 126/144 | 87.50% | A119 support somebody with hand (6), A108 knock over other person (hit with body) (6), A059 walking towards each other (1) |
| 113 | A080 | squat down | 128/144 | 88.89% | A064 bounce ball (3), A102 side kick (2), A092 move heavy objects (2) |
| 114 | A009 | standing up (from sitting position) | 66/72 | 91.67% | A006 pickup (2), A027 jump up (1), A016 wear a shoe (1) |
| 115 | A014 | wear jacket | 66/72 | 91.67% | A015 take off jacket (3), A007 throw (1), A035 nod head/bow (1) |
| 116 | A055 | hugging other person | 67/72 | 93.06% | A052 pushing other person (3), A050 punching/slapping other person (1), A045 touch chest (stomachache/heart pain) (1) |
| 117 | A043 | falling | 68/72 | 94.44% | A008 sitting down (3), A016 wear a shoe (1) |
| 118 | A114 | carry something with other person | 136/144 | 94.44% | A092 move heavy objects (3), A108 knock over other person (hit with body) (2), A117 whisper in other person's ear (1) |
| 119 | A060 | walking apart from each other | 70/72 | 97.22% | A059 walking towards each other (1), A116 follow other person (1) |
| 120 | A059 | walking towards each other | 71/72 | 98.61% | A054 point finger at the other person (1) |
