# XCLIP vision LoRA validation action accuracy：由低至高

日期：2026-09-29。這是目前保存的最高 validation accuracy XCLIP vision-adaptation run之 per-action ranking，用於 action scope人工審查；不自動決定 remove、merge、irrelevant或unknown。

## Evidence contract

- Model：Phase 3 upper-two vision-block LoRA + MIT q/v LoRA + 120-class head，seed 42，best epoch 13。
- Split：canonical validation，共 12,960 筆；沒有使用 test predictions。
- Overall validation accuracy：58.9198%（7,636／12,960）。
- Logit source：`artifacts/skeletonclip_phase3/vision_batch16_workers24_uint8_seed42/vision_lora_ce_seed42/best_validation_logits.pt`，SHA256 `f6aa77323bcca6c49468037bb4f47b43cb83bb7b7fe09b7c1a5303d4d8479c4c`；payload包含 `[12960, 120]` FP32 logits、labels與sample IDs。
- 排序：per-class top-1 accuracy由低至高，同 accuracy再依action ID。A001–A060每類72筆，A061–A120每類144筆。
- Action names：[NTU RGB+D authors repository](https://github.com/shahroudy/NTURGB-D#action-classes) 的官方 class list；計算以numeric class ID為準。

## Interpretation

低 accuracy可能來自短暫事件被8-frame uniform sampling漏掉、小物件或細小手勢解析不足、labels近義、domain shift或model capacity，不等同產品不需要該action。Scope決策應另外記錄產品重要性；若改成較小class set，需重新訓練baseline，不能把舊120-class predictions直接過濾後當成新任務成績。

## Ranking

| Rank | Action | Name | Correct / total | Accuracy | Most common wrong predictions |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | A074 | counting money | 4/144 | 2.78% | A075 cutting nails (47), A076 cutting paper (using scissors) (28), A073 staple book (17) |
| 2 | A072 | make victory sign | 5/144 | 3.47% | A071 make ok sign (63), A067 hush (quiet) (15), A069 thumb up (8) |
| 3 | A084 | play magic cube | 10/144 | 6.94% | A075 cutting nails (46), A076 cutting paper (using scissors) (26), A078 open bottle (18) |
| 4 | A073 | staple book | 17/144 | 11.81% | A076 cutting paper (using scissors) (41), A075 cutting nails (32), A083 ball up paper (10) |
| 5 | A082 | fold paper | 20/144 | 13.89% | A076 cutting paper (using scissors) (33), A075 cutting nails (24), A083 ball up paper (18) |
| 6 | A086 | apply cream on hand back | 23/144 | 15.97% | A078 open bottle (22), A075 cutting nails (18), A083 ball up paper (17) |
| 7 | A034 | rub two hands together | 15/72 | 20.83% | A039 put the palms together (19), A010 clapping (16), A040 cross hands in front (say stop) (5) |
| 8 | A011 | reading | 16/72 | 22.22% | A029 playing with phone/tablet (21), A012 writing (17), A033 check time (from watch) (5) |
| 9 | A078 | open bottle | 38/144 | 26.39% | A083 ball up paper (17), A075 cutting nails (14), A096 cross arms (14) |
| 10 | A069 | thumb up | 39/144 | 27.08% | A071 make ok sign (41), A070 thumb down (9), A067 hush (quiet) (9) |
| 11 | A081 | toss a coin | 39/144 | 27.08% | A077 snapping fingers (19), A071 make ok sign (14), A066 juggling table tennis balls (12) |
| 12 | A049 | use a fan (with hand or paper)/feeling warm | 22/72 | 30.56% | A037 wipe face (11), A001 drink water (4), A031 pointing to something with finger (4) |
| 13 | A077 | snapping fingers | 47/144 | 32.64% | A071 make ok sign (16), A093 shake fist (12), A096 cross arms (9) |
| 14 | A047 | touch neck (neckache) | 24/72 | 33.33% | A037 wipe face (10), A045 touch chest (stomachache/heart pain) (6), A004 brushing hair (6) |
| 15 | A012 | writing | 25/72 | 34.72% | A029 playing with phone/tablet (28), A011 reading (6), A030 typing on a keyboard (5) |
| 16 | A041 | sneeze/cough | 25/72 | 34.72% | A025 reach into pocket (6), A046 touch back (backache) (5), A018 wear on glasses (4) |
| 17 | A110 | shoot at other person with a gun | 50/144 | 34.72% | A115 take a photo of other person (33), A107 wield knife towards other person (20), A120 finger-guessing game (rock-paper-scissors) (12) |
| 18 | A093 | shake fist | 51/144 | 35.42% | A071 make ok sign (17), A067 hush (quiet) (13), A077 snapping fingers (11) |
| 19 | A004 | brushing hair | 26/72 | 36.11% | A044 touch head (headache) (13), A003 brushing teeth (7), A047 touch neck (neckache) (7) |
| 20 | A083 | ball up paper | 52/144 | 36.11% | A075 cutting nails (19), A076 cutting paper (using scissors) (17), A078 open bottle (11) |
| 21 | A010 | clapping | 27/72 | 37.50% | A034 rub two hands together (12), A039 put the palms together (12), A029 playing with phone/tablet (5) |
| 22 | A106 | hit other person with something | 54/144 | 37.50% | A107 wield knife towards other person (31), A109 grab other person's stuff (9), A120 finger-guessing game (rock-paper-scissors) (8) |
| 23 | A002 | eat meal/snack | 28/72 | 38.89% | A019 take off glasses (9), A036 shake head (6), A001 drink water (4) |
| 24 | A107 | wield knife towards other person | 56/144 | 38.89% | A109 grab other person's stuff (16), A110 shoot at other person with a gun (11), A120 finger-guessing game (rock-paper-scissors) (10) |
| 25 | A079 | sniff (smell) | 57/144 | 39.58% | A105 blow nose (16), A077 snapping fingers (8), A075 cutting nails (8) |
| 26 | A065 | tennis bat swing | 60/144 | 41.67% | A066 juggling table tennis balls (14), A096 cross arms (9), A070 thumb down (8) |
| 27 | A070 | thumb down | 60/144 | 41.67% | A069 thumb up (18), A071 make ok sign (11), A068 flick hair (6) |
| 28 | A091 | open a box | 60/144 | 41.67% | A076 cutting paper (using scissors) (17), A078 open bottle (16), A082 fold paper (13) |
| 29 | A067 | hush (quiet) | 61/144 | 42.36% | A079 sniff (smell) (14), A071 make ok sign (14), A103 yawn (11) |
| 30 | A068 | flick hair | 61/144 | 42.36% | A061 put on headphone (12), A105 blow nose (11), A093 shake fist (7) |
| 31 | A023 | hand waving | 31/72 | 43.06% | A049 use a fan (with hand or paper)/feeling warm (7), A038 salute (6), A022 cheer up (4) |
| 32 | A028 | make a phone call/answer phone | 32/72 | 44.44% | A003 brushing teeth (9), A013 tear up paper (3), A001 drink water (3) |
| 33 | A032 | taking a selfie | 32/72 | 44.44% | A031 pointing to something with finger (20), A023 hand waving (5), A001 drink water (3) |
| 34 | A076 | cutting paper (using scissors) | 64/144 | 44.44% | A075 cutting nails (28), A073 staple book (21), A078 open bottle (6) |
| 35 | A089 | put something into a bag | 65/144 | 45.14% | A090 take something out of a bag (58), A087 put on bag (4), A064 bounce ball (3) |
| 36 | A075 | cutting nails | 69/144 | 47.92% | A073 staple book (13), A076 cutting paper (using scissors) (13), A078 open bottle (12) |
| 37 | A103 | yawn | 69/144 | 47.92% | A105 blow nose (16), A067 hush (quiet) (13), A068 flick hair (11) |
| 38 | A003 | brushing teeth | 35/72 | 48.61% | A001 drink water (7), A036 shake head (4), A037 wipe face (4) |
| 39 | A030 | typing on a keyboard | 35/72 | 48.61% | A029 playing with phone/tablet (19), A012 writing (12), A033 check time (from watch) (2) |
| 40 | A037 | wipe face | 36/72 | 50.00% | A044 touch head (headache) (5), A004 brushing hair (5), A001 drink water (4) |
| 41 | A104 | stretch oneself | 72/144 | 50.00% | A097 arm circles (26), A095 hands up (both hands) (11), A063 shoot at the basket (11) |
| 42 | A105 | blow nose | 72/144 | 50.00% | A067 hush (quiet) (16), A103 yawn (8), A071 make ok sign (6) |
| 43 | A071 | make ok sign | 73/144 | 50.69% | A067 hush (quiet) (11), A069 thumb up (10), A096 cross arms (8) |
| 44 | A044 | touch head (headache) | 37/72 | 51.39% | A037 wipe face (7), A047 touch neck (neckache) (6), A038 salute (4) |
| 45 | A066 | juggling table tennis balls | 74/144 | 51.39% | A083 ball up paper (12), A082 fold paper (6), A077 snapping fingers (5) |
| 46 | A108 | knock over other person (hit with body) | 77/144 | 53.47% | A117 whisper in other person's ear (15), A111 step on foot (13), A109 grab other person's stuff (10) |
| 47 | A038 | salute | 40/72 | 55.56% | A023 hand waving (6), A003 brushing teeth (5), A004 brushing hair (3) |
| 48 | A045 | touch chest (stomachache/heart pain) | 40/72 | 55.56% | A048 nausea or vomiting condition (14), A025 reach into pocket (3), A026 hopping (one foot jumping) (2) |
| 49 | A063 | shoot at the basket | 82/144 | 56.94% | A093 shake fist (10), A095 hands up (both hands) (9), A104 stretch oneself (8) |
| 50 | A090 | take something out of a bag | 82/144 | 56.94% | A089 put something into a bag (25), A087 put on bag (5), A091 open a box (5) |
| 51 | A025 | reach into pocket | 42/72 | 58.33% | A036 shake head (9), A033 check time (from watch) (5), A028 make a phone call/answer phone (2) |
| 52 | A029 | playing with phone/tablet | 42/72 | 58.33% | A012 writing (9), A033 check time (from watch) (8), A034 rub two hands together (3) |
| 53 | A085 | apply cream on face | 85/144 | 59.03% | A083 ball up paper (10), A079 sniff (smell) (9), A105 blow nose (6) |
| 54 | A007 | throw | 43/72 | 59.72% | A042 staggering (10), A023 hand waving (6), A038 salute (5) |
| 55 | A042 | staggering | 43/72 | 59.72% | A005 drop (9), A016 wear a shoe (4), A024 kicking something (3) |
| 56 | A050 | punching/slapping other person | 43/72 | 59.72% | A052 pushing other person (10), A053 pat on back of other person (8), A055 hugging other person (2) |
| 57 | A019 | take off glasses | 44/72 | 61.11% | A018 wear on glasses (12), A025 reach into pocket (4), A015 take off jacket (2) |
| 58 | A031 | pointing to something with finger | 44/72 | 61.11% | A049 use a fan (with hand or paper)/feeling warm (5), A032 taking a selfie (4), A047 touch neck (neckache) (3) |
| 59 | A046 | touch back (backache) | 44/72 | 61.11% | A025 reach into pocket (5), A047 touch neck (neckache) (5), A023 hand waving (4) |
| 60 | A056 | giving something to other person | 44/72 | 61.11% | A058 handshaking (8), A053 pat on back of other person (6), A057 touch other person's pocket (5) |
| 61 | A109 | grab other person's stuff | 89/144 | 61.81% | A111 step on foot (15), A118 exchange things with other person (9), A120 finger-guessing game (rock-paper-scissors) (6) |
| 62 | A048 | nausea or vomiting condition | 45/72 | 62.50% | A045 touch chest (stomachache/heart pain) (16), A016 wear a shoe (3), A030 typing on a keyboard (2) |
| 63 | A039 | put the palms together | 46/72 | 63.89% | A034 rub two hands together (5), A049 use a fan (with hand or paper)/feeling warm (5), A010 clapping (4) |
| 64 | A102 | side kick | 93/144 | 64.58% | A100 butt kicks (kick backward) (33), A080 squat down (3), A090 take something out of a bag (2) |
| 65 | A117 | whisper in other person's ear | 95/144 | 65.97% | A111 step on foot (8), A119 support somebody with hand (6), A112 high-five (5) |
| 66 | A094 | throw up cap/hat | 97/144 | 67.36% | A095 hands up (both hands) (7), A103 yawn (5), A071 make ok sign (5) |
| 67 | A095 | hands up (both hands) | 97/144 | 67.36% | A104 stretch oneself (14), A097 arm circles (8), A071 make ok sign (6) |
| 68 | A058 | handshaking | 49/72 | 68.06% | A054 point finger at the other person (12), A053 pat on back of other person (5), A056 giving something to other person (4) |
| 69 | A097 | arm circles | 99/144 | 68.75% | A098 arm swings (8), A104 stretch oneself (7), A070 thumb down (7) |
| 70 | A120 | finger-guessing game (rock-paper-scissors) | 100/144 | 69.44% | A115 take a photo of other person (12), A112 high-five (7), A107 wield knife towards other person (5) |
| 71 | A061 | put on headphone | 101/144 | 70.14% | A062 take off headphone (13), A068 flick hair (10), A105 blow nose (4) |
| 72 | A096 | cross arms | 101/144 | 70.14% | A078 open bottle (5), A098 arm swings (4), A077 snapping fingers (3) |
| 73 | A112 | high-five | 101/144 | 70.14% | A113 cheers and drink (13), A115 take a photo of other person (13), A120 finger-guessing game (rock-paper-scissors) (9) |
| 74 | A001 | drink water | 51/72 | 70.83% | A003 brushing teeth (3), A028 make a phone call/answer phone (3), A040 cross hands in front (say stop) (3) |
| 75 | A016 | wear a shoe | 51/72 | 70.83% | A006 pickup (9), A017 take off a shoe (7), A048 nausea or vomiting condition (2) |
| 76 | A017 | take off a shoe | 52/72 | 72.22% | A016 wear a shoe (13), A006 pickup (3), A014 wear jacket (1) |
| 77 | A027 | jump up | 52/72 | 72.22% | A042 staggering (4), A035 nod head/bow (4), A007 throw (3) |
| 78 | A100 | butt kicks (kick backward) | 104/144 | 72.22% | A102 side kick (15), A099 running on the spot (5), A080 squat down (4) |
| 79 | A018 | wear on glasses | 53/72 | 73.61% | A019 take off glasses (5), A025 reach into pocket (3), A037 wipe face (2) |
| 80 | A098 | arm swings | 106/144 | 73.61% | A096 cross arms (10), A070 thumb down (6), A097 arm circles (5) |
| 81 | A051 | kicking other person | 54/72 | 75.00% | A050 punching/slapping other person (7), A058 handshaking (3), A053 pat on back of other person (2) |
| 82 | A005 | drop | 55/72 | 76.39% | A036 shake head (5), A025 reach into pocket (4), A031 pointing to something with finger (4) |
| 83 | A035 | nod head/bow | 55/72 | 76.39% | A025 reach into pocket (4), A041 sneeze/cough (4), A016 wear a shoe (3) |
| 84 | A040 | cross hands in front (say stop) | 55/72 | 76.39% | A039 put the palms together (5), A049 use a fan (with hand or paper)/feeling warm (3), A047 touch neck (neckache) (2) |
| 85 | A057 | touch other person's pocket | 55/72 | 76.39% | A053 pat on back of other person (12), A052 pushing other person (1), A111 step on foot (1) |
| 86 | A099 | running on the spot | 110/144 | 76.39% | A100 butt kicks (kick backward) (6), A102 side kick (6), A066 juggling table tennis balls (5) |
| 87 | A111 | step on foot | 110/144 | 76.39% | A116 follow other person (8), A120 finger-guessing game (rock-paper-scissors) (7), A107 wield knife towards other person (4) |
| 88 | A115 | take a photo of other person | 112/144 | 77.78% | A113 cheers and drink (9), A112 high-five (6), A110 shoot at other person with a gun (4) |
| 89 | A064 | bounce ball | 113/144 | 78.47% | A073 staple book (3), A105 blow nose (3), A066 juggling table tennis balls (3) |
| 90 | A088 | take off bag | 113/144 | 78.47% | A090 take something out of a bag (10), A087 put on bag (7), A062 take off headphone (3) |
| 91 | A013 | tear up paper | 58/72 | 80.56% | A010 clapping (3), A011 reading (3), A002 eat meal/snack (2) |
| 92 | A033 | check time (from watch) | 58/72 | 80.56% | A034 rub two hands together (3), A031 pointing to something with finger (3), A047 touch neck (neckache) (2) |
| 93 | A118 | exchange things with other person | 116/144 | 80.56% | A109 grab other person's stuff (14), A120 finger-guessing game (rock-paper-scissors) (5), A113 cheers and drink (4) |
| 94 | A087 | put on bag | 117/144 | 81.25% | A089 put something into a bag (3), A088 take off bag (3), A061 put on headphone (3) |
| 95 | A006 | pickup | 59/72 | 81.94% | A016 wear a shoe (8), A043 falling (1), A024 kicking something (1) |
| 96 | A119 | support somebody with hand | 119/144 | 82.64% | A116 follow other person (12), A108 knock over other person (hit with body) (4), A114 carry something with other person (3) |
| 97 | A024 | kicking something | 60/72 | 83.33% | A042 staggering (3), A007 throw (3), A026 hopping (one foot jumping) (3) |
| 98 | A101 | cross toe touch | 121/144 | 84.03% | A064 bounce ball (11), A092 move heavy objects (9), A065 tennis bat swing (1) |
| 99 | A113 | cheers and drink | 121/144 | 84.03% | A115 take a photo of other person (6), A112 high-five (5), A120 finger-guessing game (rock-paper-scissors) (3) |
| 100 | A052 | pushing other person | 61/72 | 84.72% | A050 punching/slapping other person (4), A058 handshaking (2), A057 touch other person's pocket (2) |
| 101 | A015 | take off jacket | 62/72 | 86.11% | A046 touch back (backache) (2), A023 hand waving (2), A022 cheer up (1) |
| 102 | A022 | cheer up | 62/72 | 86.11% | A023 hand waving (5), A038 salute (1), A019 take off glasses (1) |
| 103 | A054 | point finger at the other person | 62/72 | 86.11% | A053 pat on back of other person (5), A056 giving something to other person (3), A005 drop (1) |
| 104 | A116 | follow other person | 126/144 | 87.50% | A119 support somebody with hand (12), A111 step on foot (2), A108 knock over other person (hit with body) (2) |
| 105 | A062 | take off headphone | 127/144 | 88.19% | A061 put on headphone (4), A068 flick hair (2), A097 arm circles (2) |
| 106 | A092 | move heavy objects | 127/144 | 88.19% | A101 cross toe touch (13), A080 squat down (2), A100 butt kicks (kick backward) (1) |
| 107 | A036 | shake head | 64/72 | 88.89% | A025 reach into pocket (3), A046 touch back (backache) (1), A053 pat on back of other person (1) |
| 108 | A026 | hopping (one foot jumping) | 65/72 | 90.28% | A027 jump up (4), A042 staggering (2), A024 kicking something (1) |
| 109 | A053 | pat on back of other person | 65/72 | 90.28% | A054 point finger at the other person (5), A056 giving something to other person (1), A057 touch other person's pocket (1) |
| 110 | A020 | put on a hat/cap | 67/72 | 93.06% | A018 wear on glasses (2), A038 salute (1), A019 take off glasses (1) |
| 111 | A014 | wear jacket | 68/72 | 94.44% | A007 throw (1), A027 jump up (1), A015 take off jacket (1) |
| 112 | A021 | take off a hat/cap | 68/72 | 94.44% | A019 take off glasses (2), A022 cheer up (1), A032 taking a selfie (1) |
| 113 | A080 | squat down | 139/144 | 96.53% | A064 bounce ball (2), A102 side kick (2), A088 take off bag (1) |
| 114 | A043 | falling | 70/72 | 97.22% | A008 sitting down (2) |
| 115 | A055 | hugging other person | 70/72 | 97.22% | A050 punching/slapping other person (1), A053 pat on back of other person (1) |
| 116 | A008 | sitting down | 71/72 | 98.61% | A017 take off a shoe (1) |
| 117 | A009 | standing up (from sitting position) | 71/72 | 98.61% | A017 take off a shoe (1) |
| 118 | A059 | walking towards each other | 71/72 | 98.61% | A056 giving something to other person (1) |
| 119 | A114 | carry something with other person | 142/144 | 98.61% | A111 step on foot (1), A119 support somebody with hand (1) |
| 120 | A060 | walking apart from each other | 72/72 | 100.00% | — |
