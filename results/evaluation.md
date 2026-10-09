# The final robot, in detail

Ladder, run 1 (my 7 videos, then 5 rounds). The exam is the same 70 attempts as its last exam in experiment.py.

## 1. Exam, spot by spot

| exam video | cm from robot base | robot could copy my video there | success | what went wrong |
|---|---|---|---|---|
| IMG_8697 | 63 | yes | 10/10 | - |
| IMG_8684 | 31 | yes | 10/10 | - |
| IMG_8693 | 71 | yes | 10/10 | - |
| IMG_8689 | 47 | yes | 10/10 | - |
| IMG_8694 | 28 | yes | 1/10 | missed the coaster |
| IMG_8704 | 44 | yes | 10/10 | - |
| IMG_8710 | 27 | no | 2/10 | missed the coaster |

Overall 76%. Episodes where the arm or the carried can touched another object: 0 of 70 (a touch makes the episode fail).

## 2. The same exam, finding the can with its own camera

At every decision the robot takes a colour + depth picture with a camera on the far side of the table, keeps the light-blue pixels and fits a circle of the can's radius to their 3D points. That estimate replaces the simulator's answer. Its own gripper position and finger opening come from its joints.

| where the can is, according to | exam success | error of that estimate |
|---|---|---|
| the simulator | 76% | 0 |
| its own camera | 71% | 0.61 cm on average, 95% of the time under 0.80 cm |

## 3. The 7 spots it learned from

Does it still do them, and how far does its can stray from my path while carrying it? For comparison, the robot copying my video directly (retarget.py).

| my video | success | gap to my path (learned policy) | gap to my path (direct copy) |
|---|---|---|---|
| IMG_8682 | 3/3 | 1.3 cm | 0.3 cm |
| IMG_8688 | 3/3 | 1.2 cm | 1.3 cm |
| IMG_8690 | 3/3 | 4.0 cm | 0.5 cm |
| IMG_8703 | 3/3 | 1.2 cm | 0.2 cm |
| IMG_8707 | 3/3 | 1.3 cm | 0.3 cm |
| IMG_8718 | 3/3 | 2.1 cm | 0.3 cm |
| IMG_8723 | 3/3 | 0.8 cm | 0.2 cm |

Success at the spots it learned from: 100%. Average gap to my path: 1.7 cm (direct copy: 0.4 cm).

## 4. Every strategy, spot by spot (all runs)

| exam video | cm from robot base | start (7 videos) | imagine > retry > request (mine) | request only | retry only (RL) | random videos | all videos at once |
|---|---|---|---|---|---|---|---|
| IMG_8697 | 63 | 93% | 100% | 100% | 100% | 100% | 93% |
| IMG_8684 | 31 | 63% | 93% | 100% | 93% | 67% | 100% |
| IMG_8693 | 71 | 100% | 100% | 73% | 100% | 97% | 100% |
| IMG_8689 | 47 | 70% | 97% | 100% | 100% | 100% | 100% |
| IMG_8694 | 28 | 0% | 17% | 13% | 7% | 10% | 30% |
| IMG_8704 | 44 | 90% | 97% | 100% | 97% | 100% | 100% |
| IMG_8710 | 27 | 40% | 17% | 20% | 20% | 10% | 20% |
| **all 7** | | **65%** | **74%** | **72%** | **74%** | **69%** | **78%** |

None of the videos the robot can learn from starts closer than 32 cm to its base (IMG_8703).
Exam spots closer than that: IMG_8694, IMG_8710. On the other 5: start 83% -> imagine > retry > request (mine) 97%, request only 95%, retry only (RL) 98%, random videos 93%, all videos 99%.
