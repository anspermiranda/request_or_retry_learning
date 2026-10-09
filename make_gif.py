#!/usr/bin/env python3
"""
make_gif.py: turn a result video into a small GIF for the README (GitHub plays GIFs inside a README, not .mp4 files).

    python make_gif.py sim/IMG_8707.mp4 docs/robot_copies_my_video.gif
    python make_gif.py results/imagined_vs_real.mp4 docs/imagined_vs_real.gif
Options: --width (pixels, default 960), --fps (default 10), --start and --seconds (to cut a part of the video).
"""
import argparse
import os

import cv2
from PIL import Image


def main():
    p = argparse.ArgumentParser(description="Video -> small GIF for the README.")
    p.add_argument("video")
    p.add_argument("gif")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--start", type=float, default=0.0, help="start at this second")
    p.add_argument("--seconds", type=float, default=None, help="length in seconds (default: to the end)")
    a = p.parse_args()
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Can't open {a.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i, next_t = [], 0, a.start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / src_fps
        i += 1
        if a.seconds is not None and t > a.start + a.seconds:
            break
        if t + 1e-6 < next_t:
            continue
        next_t += 1.0 / a.fps
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (a.width, round(h * a.width / w)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise SystemExit("No frames read: check the video path and --start.")
    palette = frames[len(frames) // 2].quantize(colors=255)      # one shared palette: smaller file, no flicker
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    os.makedirs(os.path.dirname(a.gif) or ".", exist_ok=True)
    frames[0].save(a.gif, save_all=True, append_images=frames[1:], duration=round(1000 / a.fps), loop=0, optimize=True)
    mb = os.path.getsize(a.gif) / 1e6
    print(f"Saved {a.gif}: {len(frames)} frames, {frames[0].width}x{frames[0].height}, {mb:.1f} MB"
          + ("  (over 10 MB: try --width 720 or --fps 8)" if mb > 10 else ""))


if __name__ == "__main__":
    main()
