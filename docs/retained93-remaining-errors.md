# 保留 93 類：殘餘錯誤

依 whole＋zoom 的錯誤率由高到低排列。「兩個 view 都錯」指同一樣本 whole 與 zoom 均判錯，不代表跨訓練輪次一直錯。排除類別仍可出現在錯誤預測中，這些預測照樣算錯。

| 類別 | 名稱 | 合併仍錯／樣本 | 錯誤率 | 兩個 view 都錯 | 最常混淆（筆數） |
| --- | --- | --- | --- | --- | --- |
| A047 | touch neck (neckache) | 48 / 72 | 66.7% | 41 | A037 wipe face (16)；A045 touch chest (stomachache/heart pain) (5)；A041 sneeze/cough (4) |
| A067 | hush (quiet) | 88 / 144 | 61.1% | 62 | A105 blow nose (25)；A103 yawn (17)；A079 sniff (smell) (12) |
| A010 | clapping | 43 / 72 | 59.7% | 38 | A034 rub two hands together (21)；A039 put the palms together (11)；A013 tear up paper (4) |
| A107 | wield knife towards other person | 85 / 144 | 59.0% | 61 | A109 grab other person's stuff (17)；A120 finger-guessing game (rock-paper-scissors) (13)；A111 step on foot (9) |
| A049 | use a fan (with hand or paper)/feeling warm | 39 / 72 | 54.2% | 35 | A037 wipe face (10)；A003 brushing teeth (4)；A039 put the palms together (4) |
| A106 | hit other person with something | 78 / 144 | 54.2% | 61 | A107 wield knife towards other person (25)；A109 grab other person's stuff (12)；A112 high-five (8) |
| A002 | eat meal/snack | 37 / 72 | 51.4% | 27 | A019 take off glasses (8)；A041 sneeze/cough (6)；A013 tear up paper (5) |
| A041 | sneeze/cough | 35 / 72 | 48.6% | 27 | A037 wipe face (9)；A033 check time (from watch) (4)；A046 touch back (backache) (4) |
| A023 | hand waving | 33 / 72 | 45.8% | 30 | A049 use a fan (with hand or paper)/feeling warm (7)；A003 brushing teeth (5)；A022 cheer up (4) |
| A032 | taking a selfie | 32 / 72 | 44.4% | 27 | A031 pointing to something with finger (22)；A001 drink water (3)；A047 touch neck (neckache) (2) |
| A079 | sniff (smell) | 64 / 144 | 44.4% | 56 | A105 blow nose (16)；A067 hush (quiet) (6)；A075 cutting nails (5) |
| A075 | cutting nails | 62 / 144 | 43.1% | 48 | A076 cutting paper (using scissors) (14)；A078 open bottle (12)；A073 staple book (9) |
| A108 | knock over other person (hit with body) | 62 / 144 | 43.1% | 46 | A117 whisper in other person's ear (17)；A111 step on foot (12)；A116 follow other person (8) |
| A003 | brushing teeth | 30 / 72 | 41.7% | 18 | A001 drink water (5)；A037 wipe face (5)；A031 pointing to something with finger (3) |
| A068 | flick hair | 59 / 144 | 41.0% | 46 | A061 put on headphone (11)；A105 blow nose (11)；A062 take off headphone (6) |
| A028 | make a phone call/answer phone | 29 / 72 | 40.3% | 25 | A003 brushing teeth (8)；A018 wear on glasses (2)；A036 shake head (2) |
| A037 | wipe face | 29 / 72 | 40.3% | 23 | A018 wear on glasses (5)；A049 use a fan (with hand or paper)/feeling warm (5)；A044 touch head (headache) (4) |
| A103 | yawn | 58 / 144 | 40.3% | 46 | A105 blow nose (17)；A068 flick hair (11)；A067 hush (quiet) (8) |
| A007 | throw | 25 / 72 | 34.7% | 19 | A042 staggering (10)；A023 hand waving (7)；A019 take off glasses (2) |
| A031 | pointing to something with finger | 25 / 72 | 34.7% | 21 | A049 use a fan (with hand or paper)/feeling warm (5)；A033 check time (from watch) (4)；A001 drink water (3) |
| A045 | touch chest (stomachache/heart pain) | 25 / 72 | 34.7% | 18 | A048 nausea or vomiting condition (12)；A025 reach into pocket (3)；A016 wear a shoe (2) |
| A063 | shoot at the basket | 49 / 144 | 34.0% | 36 | A093 shake fist (8)；A104 stretch oneself (6)；A068 flick hair (5) |
| A029 | playing with phone/tablet | 24 / 72 | 33.3% | 15 | A033 check time (from watch) (8)；A012 writing (7)；A045 touch chest (stomachache/heart pain) (2) |
| A104 | stretch oneself | 48 / 144 | 33.3% | 36 | A097 arm circles (20)；A063 shoot at the basket (10)；A095 hands up (both hands) (5) |
| A105 | blow nose | 45 / 144 | 31.2% | 33 | A103 yawn (11)；A067 hush (quiet) (7)；A079 sniff (smell) (4) |
| A044 | touch head (headache) | 22 / 72 | 30.6% | 16 | A037 wipe face (4)；A047 touch neck (neckache) (4)；A001 drink water (3) |
| A048 | nausea or vomiting condition | 22 / 72 | 30.6% | 14 | A045 touch chest (stomachache/heart pain) (13)；A016 wear a shoe (4)；A017 take off a shoe (4) |
| A056 | giving something to other person | 21 / 72 | 29.2% | 14 | A058 handshaking (6)；A054 point finger at the other person (5)；A053 pat on back of other person (4) |
| A085 | apply cream on face | 42 / 144 | 29.2% | 35 | A105 blow nose (11)；A083 ball up paper (5)；A061 put on headphone (4) |
| A102 | side kick | 41 / 144 | 28.5% | 32 | A100 butt kicks (kick backward) (30)；A105 blow nose (3)；A099 running on the spot (2) |
| A038 | salute | 20 / 72 | 27.8% | 14 | A003 brushing teeth (4)；A004 brushing hair (3)；A023 hand waving (3) |
| A109 | grab other person's stuff | 40 / 144 | 27.8% | 31 | A118 exchange things with other person (8)；A111 step on foot (6)；A117 whisper in other person's ear (5) |
| A117 | whisper in other person's ear | 40 / 144 | 27.8% | 28 | A119 support somebody with hand (7)；A111 step on foot (6)；A112 high-five (5) |
| A042 | staggering | 19 / 72 | 26.4% | 17 | A005 drop (5)；A007 throw (4)；A025 reach into pocket (3) |
| A050 | punching/slapping other person | 19 / 72 | 26.4% | 15 | A052 pushing other person (8)；A053 pat on back of other person (5)；A057 touch other person's pocket (4) |
| A058 | handshaking | 18 / 72 | 25.0% | 9 | A054 point finger at the other person (7)；A056 giving something to other person (6)；A053 pat on back of other person (2) |
| A016 | wear a shoe | 17 / 72 | 23.6% | 9 | A006 pickup (11)；A017 take off a shoe (4)；A043 falling (1) |
| A046 | touch back (backache) | 17 / 72 | 23.6% | 11 | A033 check time (from watch) (3)；A036 shake head (3)；A003 brushing teeth (2) |
| A025 | reach into pocket | 16 / 72 | 22.2% | 15 | A036 shake head (7)；A005 drop (3)；A028 make a phone call/answer phone (2) |
| A039 | put the palms together | 16 / 72 | 22.2% | 11 | A049 use a fan (with hand or paper)/feeling warm (5)；A040 cross hands in front (say stop) (4)；A037 wipe face (3) |
| A100 | butt kicks (kick backward) | 32 / 144 | 22.2% | 25 | A102 side kick (19)；A116 follow other person (4)；A080 squat down (2) |
| A111 | step on foot | 32 / 144 | 22.2% | 23 | A116 follow other person (9)；A107 wield knife towards other person (6)；A120 finger-guessing game (rock-paper-scissors) (4) |
| A097 | arm circles | 31 / 144 | 21.5% | 27 | A104 stretch oneself (7)；A098 arm swings (5)；A096 cross arms (3) |
| A005 | drop | 15 / 72 | 20.8% | 12 | A031 pointing to something with finger (5)；A025 reach into pocket (4)；A036 shake head (3) |
| A027 | jump up | 15 / 72 | 20.8% | 13 | A042 staggering (9)；A006 pickup (2)；A035 nod head/bow (2) |
| A095 | hands up (both hands) | 29 / 144 | 20.1% | 18 | A104 stretch oneself (12)；A097 arm circles (5)；A071 make ok sign (4) |
| A017 | take off a shoe | 14 / 72 | 19.4% | 9 | A016 wear a shoe (9)；A006 pickup (3)；A014 wear jacket (1) |
| A035 | nod head/bow | 14 / 72 | 19.4% | 10 | A041 sneeze/cough (3)；A016 wear a shoe (2)；A025 reach into pocket (2) |
| A115 | take a photo of other person | 27 / 144 | 18.8% | 17 | A112 high-five (8)；A113 cheers and drink (5)；A110 shoot at other person with a gun (4) |
| A019 | take off glasses | 13 / 72 | 18.1% | 9 | A018 wear on glasses (7)；A037 wipe face (3)；A025 reach into pocket (2) |
| A057 | touch other person's pocket | 13 / 72 | 18.1% | 7 | A053 pat on back of other person (11)；A050 punching/slapping other person (1)；A058 handshaking (1) |
| A064 | bounce ball | 26 / 144 | 18.1% | 16 | A005 drop (5)；A066 juggling table tennis balls (2)；A089 put something into a bag (2) |
| A096 | cross arms | 25 / 144 | 17.4% | 16 | A098 arm swings (6)；A079 sniff (smell) (3)；A086 apply cream on hand back (2) |
| A018 | wear on glasses | 11 / 72 | 15.3% | 7 | A019 take off glasses (2)；A025 reach into pocket (2)；A028 make a phone call/answer phone (2) |
| A051 | kicking other person | 11 / 72 | 15.3% | 4 | A050 punching/slapping other person (3)；A024 kicking something (2)；A057 touch other person's pocket (2) |
| A061 | put on headphone | 22 / 144 | 15.3% | 17 | A062 take off headphone (7)；A068 flick hair (3)；A105 blow nose (3) |
| A120 | finger-guessing game (rock-paper-scissors) | 22 / 144 | 15.3% | 15 | A112 high-five (6)；A111 step on foot (4)；A118 exchange things with other person (4) |
| A099 | running on the spot | 21 / 144 | 14.6% | 19 | A100 butt kicks (kick backward) (5)；A102 side kick (3)；A064 bounce ball (2) |
| A006 | pickup | 10 / 72 | 13.9% | 5 | A016 wear a shoe (6)；A035 nod head/bow (2)；A017 take off a shoe (1) |
| A040 | cross hands in front (say stop) | 10 / 72 | 13.9% | 7 | A049 use a fan (with hand or paper)/feeling warm (5)；A039 put the palms together (3)；A033 check time (from watch) (1) |
| A118 | exchange things with other person | 18 / 144 | 12.5% | 14 | A109 grab other person's stuff (8)；A120 finger-guessing game (rock-paper-scissors) (5)；A110 shoot at other person with a gun (2) |
| A119 | support somebody with hand | 18 / 144 | 12.5% | 14 | A116 follow other person (13)；A108 knock over other person (hit with body) (2)；A114 carry something with other person (2) |
| A015 | take off jacket | 8 / 72 | 11.1% | 7 | A046 touch back (backache) (2)；A016 wear a shoe (1)；A020 put on a hat/cap (1) |
| A024 | kicking something | 8 / 72 | 11.1% | 5 | A042 staggering (3)；A026 hopping (one foot jumping) (2)；A005 drop (1) |
| A033 | check time (from watch) | 8 / 72 | 11.1% | 5 | A047 touch neck (neckache) (3)；A031 pointing to something with finger (1)；A038 salute (1) |
| A052 | pushing other person | 8 / 72 | 11.1% | 6 | A050 punching/slapping other person (6)；A057 touch other person's pocket (1)；A058 handshaking (1) |
| A053 | pat on back of other person | 8 / 72 | 11.1% | 4 | A054 point finger at the other person (4)；A058 handshaking (2)；A050 punching/slapping other person (1) |
| A101 | cross toe touch | 16 / 144 | 11.1% | 11 | A092 move heavy objects (9)；A064 bounce ball (7) |
| A088 | take off bag | 15 / 144 | 10.4% | 13 | A090 take something out of a bag (6)；A087 put on bag (4)；A062 take off headphone (2) |
| A112 | high-five | 15 / 144 | 10.4% | 9 | A115 take a photo of other person (6)；A120 finger-guessing game (rock-paper-scissors) (5)；A113 cheers and drink (3) |
| A113 | cheers and drink | 15 / 144 | 10.4% | 9 | A120 finger-guessing game (rock-paper-scissors) (6)；A112 high-five (2)；A115 take a photo of other person (2) |
| A036 | shake head | 7 / 72 | 9.7% | 3 | A053 pat on back of other person (2)；A005 drop (1)；A010 clapping (1) |
| A098 | arm swings | 14 / 144 | 9.7% | 13 | A097 arm circles (6)；A096 cross arms (4)；A066 juggling table tennis balls (1) |
| A022 | cheer up | 6 / 72 | 8.3% | 4 | A023 hand waving (4)；A004 brushing hair (1)；A019 take off glasses (1) |
| A026 | hopping (one foot jumping) | 6 / 72 | 8.3% | 2 | A024 kicking something (2)；A027 jump up (2)；A042 staggering (2) |
| A054 | point finger at the other person | 6 / 72 | 8.3% | 6 | A053 pat on back of other person (4)；A005 drop (1)；A056 giving something to other person (1) |
| A087 | put on bag | 12 / 144 | 8.3% | 7 | A088 take off bag (3)；A089 put something into a bag (2)；A104 stretch oneself (2) |
| A092 | move heavy objects | 11 / 144 | 7.6% | 4 | A101 cross toe touch (7)；A114 carry something with other person (2)；A090 take something out of a bag (1) |
| A116 | follow other person | 11 / 144 | 7.6% | 8 | A119 support somebody with hand (7)；A111 step on foot (2)；A107 wield knife towards other person (1) |
| A001 | drink water | 5 / 72 | 6.9% | 3 | A003 brushing teeth (2)；A040 cross hands in front (say stop) (2)；A002 eat meal/snack (1) |
| A013 | tear up paper | 5 / 72 | 6.9% | 2 | A001 drink water (1)；A002 eat meal/snack (1)；A011 reading (1) |
| A062 | take off headphone | 7 / 144 | 4.9% | 6 | A068 flick hair (1)；A087 put on bag (1)；A091 open a box (1) |
| A014 | wear jacket | 2 / 72 | 2.8% | 2 | A016 wear a shoe (1)；A022 cheer up (1) |
| A020 | put on a hat/cap | 2 / 72 | 2.8% | 1 | A019 take off glasses (1)；A038 salute (1) |
| A043 | falling | 2 / 72 | 2.8% | 1 | A008 sitting down (1)；A016 wear a shoe (1) |
| A009 | standing up (from sitting position) | 1 / 72 | 1.4% | 0 | A017 take off a shoe (1) |
| A021 | take off a hat/cap | 1 / 72 | 1.4% | 1 | A019 take off glasses (1) |
| A080 | squat down | 2 / 144 | 1.4% | 0 | A008 sitting down (1)；A064 bounce ball (1) |
| A114 | carry something with other person | 1 / 144 | 0.7% | 1 | A116 follow other person (1) |
| A008 | sitting down | 0 / 72 | 0.0% | 0 |  |
| A055 | hugging other person | 0 / 72 | 0.0% | 0 |  |
| A059 | walking towards each other | 0 / 72 | 0.0% | 0 |  |
| A060 | walking apart from each other | 0 / 72 | 0.0% | 0 |  |
