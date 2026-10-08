#!/usr/bin/env python3
"""
Calibrate a phone camera from a checkerboard video.

Usage:
    python calibrate_camera.py path/to/checkerboard_video.MOV

Default board: 9 x 6 squares (8 x 5 inner corners), 25 mm squares.
Writes calibration.json and a few check images to calib_check/.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate a phone camera from a checkerboard video.")
    p.add_argument("video", help="path to the checkerboard video")
    p.add_argument("--cols", type=int, default=8, help="inner corners along the long side (default 8)")
    p.add_argument("--rows", type=int, default=5, help="inner corners along the short side (default 5)")
    p.add_argument("--square-mm", type=float, default=25.0, help="square size in mm (default 25)")
    p.add_argument("--candidates", type=int, default=300, help="frames to examine (default 300)")
    p.add_argument("--max-views", type=int, default=60, help="max frames used for calibration (default 60)")
    p.add_argument("--out", default="calibration.json", help="output file (default calibration.json)")
    return p.parse_args()


def detect_board(gray, pattern):
    """Fast detection at half resolution, then sub-pixel refinement at full resolution."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, pattern, flags)
    if not ok:
        return None
    corners = corners * 2.0 + 0.5
    grid = corners.reshape(pattern[1], pattern[0], 2)
    spacing = min(np.linalg.norm(np.diff(grid, axis=1), axis=2).min(),
                  np.linalg.norm(np.diff(grid, axis=0), axis=2).min())
    win = int(np.clip(spacing * 0.4, 3, 11))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    return cv2.cornerSubPix(gray, corners.astype(np.float32), (win, win), (-1, -1), crit)


def sharpness(gray, corners):
    x, y, w, h = cv2.boundingRect(corners.reshape(-1, 2).astype(np.float32))
    roi = gray[max(y, 0):y + h, max(x, 0):x + w]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0


def calibrate(obj, img_points, size):
    flags = cv2.CALIB_FIX_K3  # a simple lens model is enough for a phone's main camera
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(img_points), img_points, size, None, None, flags=flags)
    errors = []
    for pts, rv, tv in zip(img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts.reshape(-1, 2)) ** 2, axis=1)))))
    return rms, K, dist, np.array(errors)


def main():
    args = parse_args()
    pattern = (args.cols, args.rows)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // args.candidates)
    print(f"Video has about {total} frames. Checking 1 in every {step} frames for the checkerboard...")

    found, size, frame_idx, checked = [], None, -1, 0
    while True:
        if not cap.grab():
            break
        frame_idx += 1
        if frame_idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        checked += 1
        corners = detect_board(gray, pattern)
        if corners is not None:
            found.append({"idx": frame_idx, "corners": corners, "sharp": sharpness(gray, corners)})
        if checked % 50 == 0:
            print(f"  checked {checked} frames, board found in {len(found)}")
    cap.release()

    if size is None:
        sys.exit("No frames could be read from the video.")
    print(f"Checked {checked} frames; board found in {len(found)}. Frame size: {size[0]} x {size[1]}")
    if size[1] > size[0]:
        print("WARNING: this video is portrait. Your demos are landscape, so please re-record in landscape.")
    if len(found) < 15:
        sys.exit("Too few frames with the full board visible (need at least 15). "
                 "Re-record more slowly, keeping the whole board in view.")

    # Drop the blurriest 20%, then spread the chosen frames evenly through the video.
    cutoff = np.percentile([f["sharp"] for f in found], 20)
    sharp = [f for f in found if f["sharp"] >= cutoff]
    pick = np.unique(np.linspace(0, len(sharp) - 1, min(args.max_views, len(sharp))).round().astype(int))
    views = [sharp[i] for i in pick]

    obj = np.zeros((args.rows * args.cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    img_points = [v["corners"] for v in views]

    rms, K, dist, errors = calibrate(obj, img_points, size)
    keep = errors <= max(1.0, 2.5 * np.median(errors))
    if keep.sum() < len(views) and keep.sum() >= 15:
        views = [v for v, k in zip(views, keep) if k]
        img_points = [v["corners"] for v in views]
        rms, K, dist, errors = calibrate(obj, img_points, size)

    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    result = {
        "image_width": w, "image_height": h,
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "dist_coeffs": [float(d) for d in dist.ravel()],
        "rms_reprojection_error_px": float(rms),
        "views_used": len(views),
        "horizontal_fov_deg": float(np.degrees(2 * np.arctan(w / (2 * fx)))),
        "vertical_fov_deg": float(np.degrees(2 * np.arctan(h / (2 * fy)))),
        "source_video": os.path.basename(args.video),
        "board": {"inner_corners": [args.cols, args.rows], "square_mm": args.square_mm},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # Check images: detected corners on 4 frames, plus one undistorted frame (second pass, low memory).
    os.makedirs("calib_check", exist_ok=True)
    samples = views[:: max(1, len(views) // 4)][:4]
    wanted = {v["idx"]: v for v in samples}
    cap = cv2.VideoCapture(args.video)
    frame_idx, saved = -1, 0
    while wanted and cap.grab():
        frame_idx += 1
        if frame_idx in wanted:
            ok, frame = cap.retrieve()
            if not ok:
                continue
            if saved == 0:
                cv2.imwrite("calib_check/undistorted_example.jpg", cv2.undistort(frame, K, dist))
            saved += 1
            cv2.drawChessboardCorners(frame, pattern, wanted.pop(frame_idx)["corners"], True)
            cv2.imwrite(f"calib_check/corners_{saved}.jpg", frame)
    cap.release()

    print("\n===== CALIBRATION RESULT =====")
    print(f"fx = {fx:.1f}   fy = {fy:.1f}   cx = {cx:.1f}   cy = {cy:.1f}")
    print("distortion = " + ", ".join(f"{d:.4f}" for d in dist.ravel()))
    print(f"field of view = {result['horizontal_fov_deg']:.1f} x {result['vertical_fov_deg']:.1f} degrees")
    print(f"frames used = {len(views)}   average error = {rms:.3f} px")
    notes = []
    if abs(fx - fy) / fx > 0.03:
        notes.append("fx and fy differ by more than 3%")
    if abs(cx - w / 2) > 0.08 * w or abs(cy - h / 2) > 0.08 * h:
        notes.append("the image centre estimate is far from the middle of the frame")
    if rms < 0.5 and not notes:
        verdict = "EXCELLENT"
    elif rms < 1.0 and not notes:
        verdict = "GOOD"
    else:
        verdict = "CHECK NEEDED"
        if rms >= 1.0:
            notes.append("average error is 1 px or more (blurry frames or a bent board?)")
    print(f"verdict = {verdict}" + ("" if not notes else "  (" + "; ".join(notes) + ")"))
    print(f"Saved {args.out} and check images in calib_check/")


if __name__ == "__main__":
    main()
