#!/usr/bin/env bash
# Everything after the videos have been read (read_demos.py) and split (make_splits.py).
# About an hour on a 16-thread laptop. Every step prints what it found; results go to sim/, results/ and docs/.
set -e
python retarget.py                       # my videos -> robot demos, and the transfer map          (~3 min)
python retarget.py --base 0.35 -0.10     # the same robot standing at the right side of the table   (~3 min)
python experiment.py                     # the four strategies + the all-videos upper bound        (~40 min)
python evaluate.py                       # the final robot in detail, incl. its own camera          (~5 min)
python desk_test.py                      # the final robots at 100 random spots on the desk         (~3 min)
python speed.py                          # the same policy, much faster                             (~5 min)
python watch.py IMG_8707 --copy --save --no-window      # my video | the robot copying it
python watch.py --exam --save --no-window               # my video | the trained robot, at the 7 exam spots
python watch.py --train --save --no-window              # my video | the trained robot, at the 7 spots it learned from
python make_gif.py results/watch_IMG_8707_copy.mp4 docs/robot_copies_my_video.gif
python make_gif.py results/watch_IMG_8697.mp4 docs/exam_spot.gif
python make_gif.py results/imagined_vs_real.mp4 docs/imagined_vs_real.gif
if [ -f out/IMG_8707/annotated.mp4 ]; then                # what read_demos.py sees in one of my videos
  python make_gif.py out/IMG_8707/annotated.mp4 docs/what_the_computer_sees.gif
fi
echo "All done."
