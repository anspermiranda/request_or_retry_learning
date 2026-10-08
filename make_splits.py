#!/usr/bin/env python3
"""
Split the demo videos into three groups and draw them on the table.

  start : the videos the robot learns from first (your picks)
  exam  : 7 videos it never trains on, chosen to cover the whole table
  pile  : everything else, handed over only when the robot asks

Run it after read_demos.py:
    python make_splits.py --start IMG_8682 IMG_8688 IMG_8690 IMG_8703 IMG_8707 IMG_8718 IMG_8723
Writes data/splits.json and out/splits_map.jpg.
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

from read_demos import TABLE_X, TABLE_Y


def spread_pick(names, xy, k, seed):
    """k clips that represent the whole table: group the start spots into k areas (k-means)
    and take the clip closest to the middle of each area."""
    rng = np.random.default_rng(seed)
    centres = xy[rng.choice(len(xy), k, replace=False)]
    for _ in range(100):
        groups = np.argmin(((xy[:, None] - centres[None]) ** 2).sum(-1), 1)
        new = np.array([xy[groups == g].mean(0) if np.any(groups == g) else centres[g] for g in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    chosen = []
    for c in centres:
        order = np.argsort(((xy - c) ** 2).sum(1))
        chosen.append(next(names[i] for i in order if names[i] not in chosen))
    return chosen


def main():
    p = argparse.ArgumentParser(description="Split demo videos into start, exam and pile groups.")
    p.add_argument("--start", nargs="+", required=True, help="clip names for the starting set")
    p.add_argument("--exam-size", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out", help="folder with read_demos.py results (default out)")
    args = p.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.out, "summary.csv"))))
    def moved(r):   # a real demo moves the can at least 8 cm (IMG_8715 lifts it and puts it back on the coaster)
        return ((float(r["start_x"]) - float(r["end_x"])) ** 2 + (float(r["start_y"]) - float(r["end_y"])) ** 2) ** 0.5 >= 0.08
    def reliable(r):   # leave out clips whose 3D measurement the reader itself flagged
        w = r.get("warnings") or ""
        return not any(k in w for k in ("size-based distance disagrees", "covers the can's known shape poorly"))
    usable = [r for r in rows if r.get("status") in ("ok", "check") and r.get("start_x") and moved(r) and reliable(r)]
    skipped = [r["clip"] for r in rows if r not in usable]
    names = [os.path.splitext(r["clip"])[0] for r in usable]
    xy = np.array([[float(r["start_x"]), float(r["start_y"])] for r in usable])

    start = [os.path.splitext(s)[0] for s in args.start]
    missing = [s for s in start if s not in names]
    if missing:
        sys.exit(f"These starting clips were not read successfully: {missing}")

    rest = [n for n in names if n not in start]
    rest_xy = np.array([xy[names.index(n)] for n in rest])
    exam = spread_pick(rest, rest_xy, args.exam_size, args.seed)
    pile = [n for n in rest if n not in exam]

    # Fairness check: two random starting sets of the same size, drawn from everything except the exam.
    rng = np.random.default_rng(args.seed + 1)
    trainable = start + pile
    fairness = []
    for _ in range(2):
        s = sorted(rng.choice(trainable, len(start), replace=False).tolist())
        fairness.append({"start": s, "pile": [n for n in trainable if n not in s]})

    splits = {"start": start, "exam": exam, "pile": pile, "fairness_runs": fairness,
              "seed": args.seed, "skipped": skipped,
              "note": "exam clips are never used for training; pile clips are handed over only when the robot asks"}
    os.makedirs("data", exist_ok=True)
    with open("data/splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Map: every clip's start spot on the bird's-eye view of the table.
    first = os.path.join(args.out, names[0], "birdseye.jpg")
    img = cv2.imread(first) if os.path.exists(first) else np.full((850, 1150, 3), 60, np.uint8)
    demo = json.load(open(os.path.join(args.out, names[0], "demo.json")))
    ppm = img.shape[1] / (TABLE_X[1] - TABLE_X[0])

    def px(x, y):
        return int((x - TABLE_X[0]) * ppm), int((TABLE_Y[1] - y) * ppm)

    if demo.get("coaster"):
        c = demo["coaster"]
        cv2.circle(img, px(*c["center"]), int(c["diameter"] / 2 * ppm), (60, 200, 60), 3)
    colours = {"start": (255, 140, 30), "exam": (40, 40, 230), "pile": (235, 235, 235)}
    for group, members in (("pile", pile), ("exam", exam), ("start", start)):
        for n in members:
            x, y = xy[names.index(n)]
            cv2.circle(img, px(x, y), 11, colours[group], -1)
            cv2.circle(img, px(x, y), 11, (0, 0, 0), 2)
            cv2.putText(img, n[-4:], (px(x, y)[0] + 13, px(x, y)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)
    for k, (group, label) in enumerate([("start", "start (your 7)"), ("exam", "exam (never trained on)"),
                                        ("pile", "pile (given when asked)")]):
        cv2.circle(img, (25, 30 + 30 * k), 10, colours[group], -1)
        cv2.putText(img, label, (45, 37 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.out, "splits_map.jpg"), img)

    print(f"start ({len(start)}): {' '.join(start)}")
    print(f"exam  ({len(exam)}): {' '.join(exam)}")
    print(f"pile  ({len(pile)}): {' '.join(pile)}")
    if skipped:
        print(f"left out (failed, the can never really moved, or an unreliable 3D measurement): {' '.join(skipped)}")
    d = np.sqrt(((xy[[names.index(s) for s in start]][:, None] - xy[None]) ** 2).sum(-1))
    print(f"Your 7 starting spots span {100 * np.ptp(xy[[names.index(s) for s in start]][:, 0]):.0f} cm left-right "
          f"and {100 * np.ptp(xy[[names.index(s) for s in start]][:, 1]):.0f} cm front-back.")
    print("Saved data/splits.json and out/splits_map.jpg")


if __name__ == "__main__":
    main()
