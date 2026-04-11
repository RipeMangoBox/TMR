# D0: HumanML3D-E Corrected Real-Data Audit

- This report supersedes the earlier D0 report that was generated from the wrong HumanML3D-E source.
- Report type: **data audit / launch gate only**
- This report is computed directly from the six trusted HumanML3D-E `.npy` files and does not use any training checkpoint.
- Reliability boundary: the statistics below remain valid even if D1/D1.5/D2a/D2b have not finished full training epochs.
- Scope boundary: D0 only determines whether corrected real data justify launching D1; it is not evidence that Stage4.1 already won.
- Canonical dataset root: `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- Total entries across train/val/test: **30722**
- Total captions across train/val/test: **83347**
- Valid event extractions: **83347**
- Event extraction coverage: **100.00%**
- Canonical `text[].decomposed` coverage: **100.00%**
- Invalid captions: **0**

## Split Entry Counts

| Split | Entries |
|---|---:|
| train | 24546 |
| val | 1530 |
| test | 4646 |

## Motion-Level Text Count Distribution

| #texts per motion | Count |
|---:|---:|
| 1 | 2952 |
| 2 | 2941 |
| 3 | 24803 |
| 4 | 26 |

## Caption Structure Summary

- Plain natural-language captions: **83347**
- `action i:` marker captions: **0**
- Event source breakdown: **{'decomposed': 83347}**

## Conditioned Test Files

| File | Entries | Overlap with `data_test.npy` |
|---|---:|---:|
| condition2 | 2622 | 2446 |
| condition3 | 927 | 874 |
| condition4 | 260 | 251 |

## K Distribution

| K | Count | Ratio over valid |
|---:|---:|---:|
| 1 | 42261 | 50.70% |
| 2 | 28859 | 34.63% |
| 3 | 8794 | 10.55% |
| 4 | 2493 | 2.99% |
| 5 | 670 | 0.80% |
| 6 | 186 | 0.22% |
| 7 | 58 | 0.07% |
| 8 | 20 | 0.02% |
| 9 | 3 | 0.00% |
| 10 | 3 | 0.00% |

## Key Ratios for D0 Gates

- `K>=2` ratio: **49.30%** (41086/83347)
- `K=1` ratio: **50.70%** (42261/83347)
- Rule-based overlap ratio: **1.81%** (1508/83347)

## Event Length Summary

- Caption-level average event length (words): mean **7.9648**, median **7.0**, p25/p75 **5.5 / 9.5**
- Caption-level average event length (frames proxy = motion_length / K): mean **101.5603**, median **93.0**, p25/p75 **60.0 / 133.0**

## D0 Quantitative Gates

1. Gate-1 (`K>=2` ratio < 40%): **NOT_TRIGGERED** (value=49.30%)
2. Gate-2 (`K=1` ratio > 60%): **NOT_TRIGGERED** (value=50.70%)
3. Gate-3 (rule-based overlap ratio > 15%): **NOT_TRIGGERED** (value=1.81%)

- Recommendation: **DATA-GATE GO: corrected real data support launching D1 frozen minimal event-time head.**

## Rule-Based Overlap Examples

- [train] `000004` (cues=while): a person bends forward at the waist while both hands are tucked inside their armpits and elbows move up and down.
- [train] `000094` (cues=while): while standing, a person makes a series of small gestures near their face using both hands.
- [train] `000102` (cues=simultaneous): the person walks to the left and stops, they raise their hands up above shoulders. then moving both hands together to the left simultaneously and back. again but only with left hand,
- [train] `000169` (cues=while): a person moves their right hand from the right side of their body towards the left side and makes a petting or digging motion while the left arm hangs at their side.
- [train] `000251` (cues=while): a person takes a giant leap forward while his arms flail out to his sides.
- [train] `000242` (cues=while): a person slightly wobbly walks foreward while grabbing onto railings at both sides.
- [train] `000280` (cues=simultaneous): raise both arms simultaneously with little spirit, keep them there for a moment and drop them lifelessly
- [train] `000368` (cues=while): a person jumps up while simultanesouly moving their arms/hands inward & upward towards their armpits, after they land they move both arms out at 45 degree angles away from their sides and does a slight jog serveral steps forward.
- [train] `000394` (cues=while): a person climbs stairs while using the railing
- [train] `000441` (cues=while): a person shrugs shoulders while leaving both arms flaccid and then shrugs shoulders while bending elbows and making hand gesture.
- [train] `000446` (cues=at_same_time): person swings arms back whilst bending slightly at the knee and then leaps forward with both feet at the same time.
- [train] `000455` (cues=while): a person turns clockwise while standing in place then slowly stretches arms outward to their side.
- [train] `000471` (cues=while): while a person is standing, they bend their knees and jump forward.
- [train] `000518` (cues=while): stand on the left foot while right leg makes a wide circle out and back down to the left foot.
- [train] `000542` (cues=while): a person does a light skip while walking
- [train] `000555` (cues=while): putting the arms to the chest with a slight blend to the knees, then bringing the left arm to the hip while the right goes in a circle.
- [train] `000560` (cues=while): person starts with arms held out to the sides, steps forward with right foot, bends down, reaches for an item with right hand, retrieves item and holds with both hands, turns left, steadies item with right hand on top, walks forward, turns right and bends down holding item with left hand while right hand reaches up and makes repeated stroking motions towards the item held in left hand.
- [train] `000583` (cues=while): someone is doing arm exercises and dancing a little while they are doing so.
- [train] `000606` (cues=while): walks forward and moves to the left slitghtly while walkng
- [train] `000676` (cues=at_same_time): a person slowly hops up and down on both legs at the same time.

## Caveat

This corrected rerun uses canonical `text[].decomposed[].caption` as the primary event structure and only keeps caption parsing as a fallback. Training completeness matters for D1-D3 model comparisons, but not for this D0 data-audit report.
