#!/usr/bin/env python3
"""
read_demos.py: turn phone demo videos into robot-ready data and annotated videos.

Perception uses real models, not colour rules:
  - YOLOE (open-vocabulary detector) recognises the objects on the table by name in the first frame:
    the can, the coaster, and the other objects (saved for building the simulator).
  - SAM 2 (Segment Anything 2) turns the can and coaster boxes into pixel-accurate masks and tracks
    them through the whole video, including when the hand covers part of the can.
  - MediaPipe tracks the hand (21 points per frame).
Geometry:
  - The ArUco marker gives the camera position (the phone is fixed, so once per clip).
  - While the can rests on the table, its position comes from where its mask touches the table.
  - While it is carried, its distance comes from how wide its mask looks (its real diameter is known),
    calibrated against the table just before it is lifted and just after it is put down.

Coordinates are metres in the marker's frame: origin at the marker centre, x right, y away from the
camera, z up from the table.

Usage:
    python read_demos.py data/demos                  # every clip in a folder
    python read_demos.py data/demos/IMG_8682.MOV     # one clip
Outputs: out/<clip>/demo.json, scene.json, annotated.mp4, birdseye.jpg, and out/summary.csv
"""
import argparse
import csv
import json
import os
import pickle
import sys
import urllib.request

os.environ.setdefault("GLOG_minloglevel", "2")        # quieter MediaPipe / TensorFlow Lite messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

VERSION = "4.7 (7 Oct): final fixes - camera re-measured, coaster occlusion, clean calibration windows"

# ---------------------------------------------------------------- settings
MARKER_ID = 0
MARKER_SIZE = 0.0955                  # m, outer edge of the black square
CAN_HEIGHT = 0.175                    # m, measured with a ruler
CAN_DIAMETER = 0.063                  # m, body
CAN_END_DIAMETER = 0.059              # m, tapered ends
COASTER_THICKNESS = 0.005             # m
DETECTOR = "yoloe-11s-seg.pt"         # downloaded automatically on first use
SEGMENTER = "sam2.1_b.pt"             # downloaded automatically on first use
CAN_NAMES = ["energy drink can", "drink can", "can"]     # no "cup"/"bottle": they competed for the same object
COASTER_NAMES = ["coaster", "drink coaster", "round black coaster"]
OTHER_NAMES = ["laptop", "book", "watch", "keys", "charger", "glasses case", "power socket", "paper"]
SCENE_CLASSES = CAN_NAMES + COASTER_NAMES + OTHER_NAMES
TABLE_X, TABLE_Y = (-0.80, 0.35), (-0.40, 0.45)            # area shown in the bird's-eye view
WORKSPACE_X, WORKSPACE_Y = (-0.68, 0.25), (-0.35, 0.25)    # where the can may stand
MOVE_START = 0.015                    # m, the can counts as moved once this far from its rest spot
LIFTED = 0.01                         # m, the can counts as lifted above this height
CAN_HSV_LOW, CAN_HSV_HIGH = (90, 80, 60), (108, 255, 255)   # only used if the detector misses the can
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]
FINGERTIPS = [4, 8, 12, 16, 20]
BLUE, GREEN, PINK, PINK_DOT = (255, 170, 60), (60, 200, 60), (180, 105, 255), (147, 20, 255)
YELLOW, WHITE, GREY = (0, 220, 255), (255, 255, 255), (170, 170, 170)


LOG_PATH = None


def log(msg):
    print(msg)
    if LOG_PATH:
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")


def pt(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


# ---------------------------------------------------------------- camera geometry
class Camera:
    def __init__(self, calib_path):
        c = json.load(open(calib_path))
        self.K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)
        self.D = np.array(c["dist_coeffs"], float)
        self.size = (c["image_width"], c["image_height"])
        self.rvec = self.tvec = self.R = self.C = None

    def set_pose(self, rvec, tvec):
        self.rvec, self.tvec = rvec.reshape(3, 1), tvec.reshape(3, 1)
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.C = (-self.R.T @ self.tvec).ravel()          # camera position in table coordinates

    def project(self, pts):
        uv, _ = cv2.projectPoints(np.asarray(pts, float).reshape(-1, 3), self.rvec, self.tvec, self.K, self.D)
        return uv.reshape(-1, 2)

    def rays(self, uv):
        n = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        return (self.R.T @ np.column_stack([n, np.ones(len(n))]).T).T

    def to_plane(self, uv, z=0.0):
        d = self.rays(uv)
        return self.C + d * ((z - self.C[2]) / d[:, 2])[:, None]

    def at_depth(self, uv, depth):
        n = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        return (self.R.T @ (np.column_stack([n * depth, np.full(len(n), depth)]).T - self.tvec)).T


def find_marker_pose(path, cam, n_frames=15):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                  cv2.aruco.DetectorParameters())
    cap, found = cv2.VideoCapture(path), []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None or MARKER_ID not in ids.ravel():
            continue
        c = corners[list(ids.ravel()).index(MARKER_ID)].reshape(-1, 1, 2).astype(np.float32)
        c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
        found.append(c.reshape(-1, 2))
    cap.release()
    if len(found) < 3:
        raise RuntimeError("marker not found in the first frames")
    corners = np.median(np.array(found), 0)
    h = MARKER_SIZE / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, cam.K, cam.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    cam.set_pose(rvec, tvec)
    return corners, float(np.abs(cam.project(obj) - corners).max())


class BirdsEye:
    """Top-down view of the table, made by re-projecting the camera image onto the table plane."""

    def __init__(self, cam, ppm=1000):
        self.ppm = ppm
        self.w, self.h = int((TABLE_X[1] - TABLE_X[0]) * ppm), int((TABLE_Y[1] - TABLE_Y[0]) * ppm)
        gx, gy = np.meshgrid(TABLE_X[0] + (np.arange(self.w) + 0.5) / ppm, TABLE_Y[1] - (np.arange(self.h) + 0.5) / ppm)
        uv = cam.project(np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1))
        self.mx = uv[:, 0].reshape(self.h, self.w).astype(np.float32)
        self.my = uv[:, 1].reshape(self.h, self.w).astype(np.float32)

    def warp(self, frame):
        return cv2.remap(frame, self.mx, self.my, cv2.INTER_LINEAR, borderValue=(35, 35, 35))

    def to_px(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.column_stack([(xy[:, 0] - TABLE_X[0]) * self.ppm, (TABLE_Y[1] - xy[:, 1]) * self.ppm])


def in_workspace(p):
    return WORKSPACE_X[0] < p[0] < WORKSPACE_X[1] and WORKSPACE_Y[0] < p[1] < WORKSPACE_Y[1]


# ---------------------------------------------------------------- recognition (YOLOE)
def load_models():
    try:
        from ultralytics import YOLOE
        from ultralytics.models.sam import SAM2VideoPredictor
    except ImportError:
        sys.exit("Ultralytics is not installed. Run:  pip install ultralytics")
    detector = YOLOE(DETECTOR)
    detector.set_classes(SCENE_CLASSES)
    return detector, SAM2VideoPredictor


def detect_scene(frame, detector):
    """Every object YOLOE recognises in the frame, by name, with its box and outline."""
    res = detector.predict(frame, conf=0.05, verbose=False)[0]
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    polys = res.masks.xy if res.masks is not None else [None] * len(res.boxes)
    for box, cls, conf, poly in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                                    res.boxes.conf.cpu().numpy(), polys):
        out.append({"name": res.names[int(cls)], "box": [float(v) for v in box], "conf": float(conf),
                    "polygon": None if poly is None or len(poly) == 0 else np.round(poly).astype(int).tolist()})
    return out


def base_on_table(cam, bottom_uv, z_table=0.0):
    """Centre of the can's base, from the lowest point of its mask (where it touches the surface)."""
    front = cam.to_plane(bottom_uv, z_table)[0]
    away = (front[:2] - cam.C[:2]) / np.linalg.norm(front[:2] - cam.C[:2])
    return np.array([front[0] + CAN_DIAMETER / 2 * away[0], front[1] + CAN_DIAMETER / 2 * away[1], z_table])


def hull_width_at(poly, y):
    xs = []
    for k in range(len(poly)):
        (x0, y0), (x1, y1) = poly[k], poly[(k + 1) % len(poly)]
        if y0 != y1 and (y0 - y) * (y1 - y) <= 0:
            xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    return max(xs) - min(xs) if len(xs) >= 2 else 0.0


def base_from_width(cam, bottom_uv, width_px, width_row):
    """Where the can is while carried: slide a real-size can along the ray through the mask's lowest point until
    its projected outline is exactly as wide as the mask at the measured row (fitting the 3D can to the mask)."""
    if width_px < 8:
        return None
    d = cam.rays(bottom_uv)[0]

    def can_at(t):
        p = cam.C + t * d
        away = (p[:2] - cam.C[:2]) / np.linalg.norm(p[:2] - cam.C[:2])
        return np.array([p[0] + CAN_DIAMETER / 2 * away[0], p[1] + CAN_DIAMETER / 2 * away[1], p[2]])

    lo, hi = 0.15, 2.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if hull_width_at(can_silhouette(cam, can_at(mid)), width_row) > width_px:
            lo = mid                                     # outline too wide: the can is farther away
        else:
            hi = mid
    return can_at((lo + hi) / 2)


def pick_can(dets, cam):
    best = None
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        if d["name"] not in CAN_NAMES or (y1 - y0) < 1.3 * (x1 - x0):
            continue
        if in_workspace(base_on_table(cam, [(x0 + x1) / 2, y1])) and (best is None or d["conf"] > best["conf"]):
            best = d
    return best


def pick_coaster(dets, cam):
    best = None
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        if d["name"] not in COASTER_NAMES or (x1 - x0) < 1.2 * (y1 - y0):
            continue
        centre = cam.to_plane([[(x0 + x1) / 2, (y0 + y1) / 2]], COASTER_THICKNESS)[0]
        if in_workspace(centre) and (best is None or d["conf"] > best["conf"]):
            best = d
    return best


def colour_can_box(frame, cam):
    """Fallback if the detector misses the can: the largest upright light-blue blob standing on the table."""
    small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), CAN_HSV_LOW, CAN_HSV_HIGH)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i] * np.array([2, 2, 2, 2, 4])
        if area < 3000 or h < 1.3 * w or not in_workspace(base_on_table(cam, [x + w / 2, y + h])):
            continue
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    if best is None:
        return None
    x, y, w, h, _ = best
    return [float(x - 0.15 * w), float(y - 0.15 * h), float(x + 1.15 * w), float(y + h + 4)]


KNOWN_COASTERS = []        # coaster centres found in earlier clips of this run (the coaster never moves)


def coaster_from_birdseye(frame, be):
    """The coaster's centre and size on the table: the dark round shape in the bird's-eye view."""
    hsv = cv2.cvtColor(be.warp(frame), cv2.COLOR_BGR2HSV)
    dark = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 0), (180, 255, 60)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in contours:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        roundness = cv2.contourArea(c) / (np.pi * r * r) if r > 0 else 0
        if r > 10 and roundness > 0.7 and 0.08 < 2 * r / be.ppm < 0.15 and (best is None or roundness > best[0]):
            best = (roundness, cx, cy, r)
    if best is None:
        return None
    _, cx, cy, r = best
    return {"center": [TABLE_X[0] + cx / be.ppm, TABLE_Y[1] - cy / be.ppm], "diameter": 2 * r / be.ppm}


def coaster_image_box(cam, coaster):
    a = np.linspace(0, 2 * np.pi, 48)
    r = coaster["diameter"] / 2
    ring = cam.project(np.column_stack([coaster["center"][0] + r * np.cos(a), coaster["center"][1] + r * np.sin(a),
                                        np.full(48, COASTER_THICKNESS)]))
    return [float(v) for v in [*ring.min(0) - 4, *ring.max(0) + 4]]


def coaster_box_from_birdseye(frame, cam, be):
    """Fallback if the detector misses the coaster: the dark round shape in the bird's-eye view."""
    hsv = cv2.cvtColor(be.warp(frame), cv2.COLOR_BGR2HSV)
    dark = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 0), (180, 255, 60)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r > 10 and cv2.contourArea(c) / (np.pi * r * r) > 0.7 and 0.08 < 2 * r / be.ppm < 0.15:
            centre = [TABLE_X[0] + cx / be.ppm, TABLE_Y[1] - cy / be.ppm]
            a = np.linspace(0, 2 * np.pi, 48)
            ring = cam.project(np.column_stack([centre[0] + r / be.ppm * np.cos(a), centre[1] + r / be.ppm * np.sin(a),
                                                np.full(48, COASTER_THICKNESS)]))
            return [float(v) for v in [*ring.min(0) - 4, *ring.max(0) + 4]]
    return None


def can_silhouette(cam, base):
    """Image outline of the real can shape (tapered ends) standing at `base`."""
    a = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    profile = [(0.0, CAN_END_DIAMETER / 2), (0.008, CAN_DIAMETER / 2), (0.155, CAN_DIAMETER / 2),
               (0.170, CAN_END_DIAMETER / 2), (CAN_HEIGHT, CAN_END_DIAMETER / 2)]
    pts = [np.column_stack([base[0] + r * np.cos(a), base[1] + r * np.sin(a), np.full(36, base[2] + h)]) for h, r in profile]
    return cv2.convexHull(cam.project(np.vstack(pts)).astype(np.float32)).reshape(-1, 2)


def can_prompt(cam, frame, det_box):
    """Prompt for SAM 2 that covers the WHOLE can, silver top included, and nothing else.
    Box: the detector's box, widened towards (or, if it runs far above the can, trimmed to) the outline of a
    real-size can standing where the detection touches the table.
    Points: three points on actual can pixels (light blue inside that outline): bottom, middle and top."""
    x0, y0, x1, y1 = det_box
    sil = can_silhouette(cam, base_on_table(cam, [(x0 + x1) / 2, y1]))
    sx0, sy0, sx1, sy1 = sil[:, 0].min(), sil[:, 1].min(), sil[:, 0].max(), sil[:, 1].max()
    mx, my = (x1 - x0) / 3, (y1 - y0) / 3
    top = sy0 - 0.1 * (sy1 - sy0) if y0 < sy0 - 0.15 * (sy1 - sy0) else max(min(y0, sy0), y0 - my)
    box = [max(min(x0, sx0), x0 - mx) - 6, top - 6, min(max(x1, sx1), x1 + mx) + 6, min(max(y1, sy1), y1 + my) + 6]
    h, w = frame.shape[:2]
    inside = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inside, np.round(sil).astype(np.int32), 1)
    inside = cv2.dilate(inside, np.ones((15, 15), np.uint8))
    bx0, by0, bx1, by1 = int(max(box[0], 0)), int(max(box[1], 0)), int(min(box[2], w)), int(min(box[3], h))
    blue = cv2.inRange(cv2.cvtColor(frame[by0:by1, bx0:bx1], cv2.COLOR_BGR2HSV), CAN_HSV_LOW, CAN_HSV_HIGH) > 0
    ys, xs = np.nonzero(blue & (inside[by0:by1, bx0:bx1] > 0))
    if len(xs) < 50:
        axis = cam.project(np.array([[*base_on_table(cam, [(x0 + x1) / 2, y1])[:2], z] for z in (0.03, 0.09, 0.15)]))
        points = axis.tolist()
    else:
        parts = np.array_split(np.argsort(ys), 3)
        points = [[float(np.median(xs[q]) + bx0), float(np.median(ys[q]) + by0)] for q in parts]
    box = [min(max(box[0], 0), w - 2), min(max(box[1], 0), h - 2), min(max(box[2], 1), w - 1), min(max(box[3], 1), h - 1)]
    points = [[min(max(px, 0), w - 1), min(max(py, 0), h - 1)] for px, py in points]
    return [float(v) for v in box], points


def coaster_prompt(box):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [[cx, cy], [cx - 0.25 * (x1 - x0), cy], [cx + 0.25 * (x1 - x0), cy]]


def mask_iou_poly(outline_a, poly_b, shape):
    a, b = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    if outline_a is None or len(outline_a) < 3:
        return 0.0
    cv2.fillPoly(a, [outline_a.reshape(-1, 1, 2)], 1)
    cv2.fillConvexPoly(b, np.round(poly_b).astype(np.int32), 1)
    return float((a & b).sum() / max((a | b).sum(), 1))


# ---------------------------------------------------------------- segmentation and tracking (SAM 2)
def sam_track(path, boxes, points, sam_class):
    """Pixel masks for each prompted object in every frame, in prompt order (None when not visible).
    Each object is prompted with a box plus positive points on the first frame."""
    predictor = sam_class(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=SEGMENTER,
                                         save=False, verbose=False))
    prev = [None] * len(boxes)
    prompts = dict(bboxes=boxes) if points is None else dict(bboxes=boxes, points=points, labels=[[1] * len(p) for p in points])
    for r in predictor(source=path, stream=True, **prompts):
        frame = r.orig_img
        h, w = frame.shape[:2]
        masks = []
        if r.masks is not None and len(r.masks.data):
            for m in r.masks.data.cpu().numpy() > 0.5:
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                masks.append(m)
        assigned = match_masks(masks, prev, boxes, (h, w))
        prev = [a if a is not None else p for a, p in zip(assigned, prev)]
        yield frame, assigned


def match_masks(masks, prev, boxes, shape):
    """Keep each object's identity from frame to frame (SAM drops empty masks, which shifts the order)."""
    refs = []
    for p, b in zip(prev, boxes):
        if p is not None:
            refs.append(p[::4, ::4])
        else:
            r = np.zeros(shape, bool)
            r[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = True
            refs.append(r[::4, ::4])
    small = [m[::4, ::4] for m in masks]
    scores = np.array([[(a & r).sum() / max((a | r).sum(), 1) for r in refs] for a in small]).reshape(len(small), len(refs))
    out = [None] * len(refs)
    while scores.size and scores.max() > 0.05:
        i, k = np.unravel_index(scores.argmax(), scores.shape)
        out[k] = masks[i]
        scores[i, :] = -1
        scores[:, k] = -1
    return out


def measure_can(mask):
    """Outline, lowest point, centre, and body width (rows below the hand, above the bottom taper)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 300:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bottom = ys >= y1 - 4
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    has = crop.any(1)
    left = crop.argmax(1)
    right = crop.shape[1] - 1 - crop[:, ::-1].argmax(1)
    widths = (right - left + 1).astype(float)
    hgt = y1 - y0 + 1
    rows = np.arange(hgt)
    body = has & (rows >= hgt - 0.55 * hgt) & (rows <= hgt - 0.08 * hgt)
    width, row = 0.0, float(y1)
    if body.any():
        wb, rb = widths[body], rows[body]
        keep = np.abs(wb - np.median(wb)) <= 0.15 * np.median(wb)    # ignore rows widened by a hand or cut by an arm
        width, row = float(np.median(wb[keep])), float(y0 + np.median(rb[keep]))
    return {"bottom": [float(np.median(xs[bottom])), float(y1)], "centre": [float(xs.mean()), float(ys.mean())],
            "width": width, "width_row": row,
            "box": [float(x0), float(y0), float(x1), float(y1)], "outline": outline(mask)}


def outline(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(np.int32)


def skin(frame):
    """Skin-coloured pixels (YCrCb rule): the hand and forearm. The can's light blue and silver never pass."""
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (40, 135, 80), (255, 180, 135))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0


def marker_pose_in(frame, calib):
    """Camera position from the marker in one frame (None if the marker is hidden, e.g. by your arm)."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or MARKER_ID not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(MARKER_ID)].reshape(-1, 1, 2).astype(np.float32)
    c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)).reshape(-1, 2)
    h = MARKER_SIZE / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, c, calib.K, calib.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec, c) if ok else None


def cameras_per_frame(calib_path, cam0, corners0, poses, n):
    """One camera per frame. If the phone was bumped, the marker shows it, and every later frame uses the new
    camera position, so a bump no longer shifts the measurements."""
    cams, current, current_corners, k = [], cam0, corners0, 0
    for t in range(n):
        while k < len(poses) and poses[k][0] <= t:
            _, rvec, tvec, corners = poses[k]
            if np.abs(np.asarray(corners) - current_corners).max() > 1.5:    # moved for real, not detection noise
                current = Camera(calib_path)
                current.set_pose(np.asarray(rvec), np.asarray(tvec))
                current_corners = np.asarray(corners)
            k += 1
        cams.append(current)
    return cams


def coaster_ring(cam, coaster, k=72):
    a = np.linspace(0, 2 * np.pi, k)
    r = coaster["diameter"] / 2
    return cam.project(np.column_stack([coaster["center"][0] + r * np.cos(a), coaster["center"][1] + r * np.sin(a),
                                        np.full(k, COASTER_THICKNESS)])).astype(np.int32)


def tint(img, mask, color, alpha):
    img[mask] = (img[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def clean_can_masks(path, can_m, hand_pts):
    """SAM 2 sometimes includes the fingers holding the can in the can's mask, which makes the can look wider,
    so closer and higher than it is. Remove skin-coloured pixels and MediaPipe's hand outline from every can
    mask, then measure it again. Only reads the video; the saved SAM 2 masks are reused."""
    cap, out = cv2.VideoCapture(path), []
    for m, hp in zip(can_m, hand_pts):
        ok, frame = cap.read()
        if not ok:
            break
        if m is None or m.get("outline") is None:
            out.append(m)
            continue
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [m["outline"].reshape(-1, 1, 2)], 1)
        remove = skin(frame)
        if hp is not None:
            hand = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillConvexPoly(hand, cv2.convexHull(hp.astype(np.int32)), 1)
            remove |= cv2.dilate(hand, np.ones((25, 25), np.uint8)) > 0
        out.append(measure_can((mask > 0) & ~remove) or m)
    cap.release()
    return out + list(can_m[len(out):])


def rolling_median(x, k=9):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k // 2):i + k // 2 + 1]
        if np.isfinite(w).any():
            out[i] = np.nanmedian(w)
    return out


# ---------------------------------------------------------------- hand (MediaPipe)
class HandTracker:
    """MediaPipe hands: the Tasks API (MediaPipe 1.x) or the older solutions API if present."""

    def __init__(self, model_dir="models"):
        import mediapipe as mp
        self.mp = mp
        if hasattr(mp, "solutions"):
            self.kind = "solutions"
            self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            os.makedirs(model_dir, exist_ok=True)
            path = os.path.join(model_dir, "hand_landmarker.task")
            if not os.path.exists(path):
                print("  downloading the MediaPipe hand model (once)...")
                urllib.request.urlretrieve(HAND_MODEL_URL, path)
            self.kind = "tasks"
            opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=path),
                                                running_mode=vision.RunningMode.VIDEO, num_hands=1,
                                                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                                                min_tracking_confidence=0.5)
            self.hands = vision.HandLandmarker.create_from_options(opts)

    def __call__(self, frame_bgr, t_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if self.kind == "solutions":
            res = self.hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            lm = res.multi_hand_landmarks[0].landmark
        else:
            res = self.hands.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), int(t_ms))
            if not res.hand_landmarks:
                return None
            lm = res.hand_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm])

    def close(self):
        if hasattr(self.hands, "close"):
            self.hands.close()


# ---------------------------------------------------------------- helpers
def smooth(x, k=7):
    if len(x) < k:
        return x.copy()
    pad = np.pad(x, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in range(x.shape[1])])


def fill_gaps(arr, valid, max_gap=20):
    out, idx, good = arr.copy(), np.arange(len(arr)), np.flatnonzero(valid)
    if len(good) < 2:
        return out, valid.copy()
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(idx, good, arr[good, j])
    filled = valid.copy()
    for g in np.flatnonzero(~valid):
        before, after = good[good < g], good[good > g]
        filled[g] = len(before) > 0 and len(after) > 0 and after[0] - before[-1] <= max_gap
    return out, filled


def fit_coaster(cam, coaster_outline):
    """Coaster centre and diameter on the table, from its mask outline in the first frame."""
    if coaster_outline is None or len(coaster_outline) < 8:
        return None
    pts = cam.to_plane(coaster_outline.astype(float), COASTER_THICKNESS)[:, :2]
    (cx, cy), r = cv2.minEnclosingCircle((pts * 1000).astype(np.float32))
    return {"center": [cx / 1000, cy / 1000], "diameter": 2 * r / 1000}


# ---------------------------------------------------------------- one clip
def process_clip(path, calib, out_root, models, hands_on=True, video_on=True, show=False, retrack=False):
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    cam = Camera(calib)
    corners, marker_err = find_marker_pose(path, cam)
    be = BirdsEye(cam)
    warnings = []
    detector, sam_class = models

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read the video")
    top_first = be.warp(first)
    cv2.imwrite(os.path.join(out_dir, "birdseye.jpg"), top_first)

    # 1. recognition: what is on the table, by name
    log("   1/3 recognising the objects on the table (YOLOE)...")
    dets = detect_scene(first, detector)
    can_det, coaster_det = pick_can(dets, cam), pick_coaster(dets, cam)
    seen = sorted(dets, key=lambda d: -d["conf"])
    log("      YOLOE saw: " + (", ".join(f"{d['name']} {100 * d['conf']:.0f}%" for d in seen[:8]) or "nothing"))
    can_box = can_det["box"] if can_det else colour_can_box(first, cam)
    # coaster: its round shape on the bird's-eye view is checked in centimetres, so a round black desk hub or a
    # watch can't pass; YOLOE's "coaster" only counts when it agrees with that shape
    coaster_geo = coaster_from_birdseye(first, be)
    if coaster_geo is None and KNOWN_COASTERS:
        coaster_geo = {"center": np.median(np.array([c["center"] for c in KNOWN_COASTERS]), 0).tolist(),
                       "diameter": float(np.median([c["diameter"] for c in KNOWN_COASTERS]))}
        coaster_how = "remembered from earlier clips"
    else:
        coaster_how = "round shape on the bird's-eye view"
    if coaster_geo is not None:
        if coaster_how.startswith("round"):
            KNOWN_COASTERS.append(coaster_geo)
        coaster_box = coaster_image_box(cam, coaster_geo)
        if coaster_det and np.linalg.norm(cam.to_plane([[(coaster_det["box"][0] + coaster_det["box"][2]) / 2,
                                                         (coaster_det["box"][1] + coaster_det["box"][3]) / 2]],
                                                       COASTER_THICKNESS)[0, :2] - np.array(coaster_geo["center"])) < 0.04:
            coaster_how = "YOLOE + " + coaster_how
    else:
        coaster_box = coaster_det["box"] if coaster_det else None
        coaster_how = "YOLOE" if coaster_det else "not found"
    perception = {"detector": DETECTOR, "segmenter": SEGMENTER,
                  "can_found_by": "YOLOE" if can_det else "colour fallback",
                  "can_label": can_det["name"] if can_det else None,
                  "can_confidence": can_det["conf"] if can_det else None,
                  "coaster_found_by": coaster_how,
                  "coaster_confidence": coaster_det["conf"] if coaster_det else None}
    if can_box is None:
        raise RuntimeError("could not find the can in the first frame")
    if not can_det:
        warnings.append("YOLOE missed the can in the first frame; used the colour fallback to prompt SAM 2")
    can_box, can_points = can_prompt(cam, first, can_box)
    scene = [{**d, "footprint": None if d["polygon"] is None else
              np.round(cam.to_plane(np.array(d["polygon"], float))[:, :2], 4).tolist()} for d in dets]
    json.dump({"clip": name, "objects": scene}, open(os.path.join(out_dir, "scene.json"), "w"))

    # 2. segmentation and tracking, plus the hand, in one pass over the video (cached in tracks.pkl)
    cache = os.path.join(out_dir, "tracks.pkl")
    if os.path.exists(cache) and not retrack:
        log("   2/3 using the saved masks and hand points (tracks.pkl); --retrack to redo them")
        t = pickle.load(open(cache, "rb"))
        can_m, hand_pts, poses, perception = t["can_m"], t["hand_pts"], t["poses"], t["perception"]
    else:
        log("   2/3 segmenting and tracking the can (SAM 2), the hand (MediaPipe) and the camera (marker)...")
        tracker = HandTracker() if hands_on else None
        can_m, hand_pts, poses, i = [], [], [], 0
        for frame, masks in sam_track(path, [can_box], [can_points], sam_class):
            can_m.append(measure_can(masks[0]) if masks[0] is not None else None)
            hand_pts.append(tracker(frame, i * 1000.0 / fps) if tracker else None)
            if i % 15 == 0:
                pose = marker_pose_in(frame, cam)
                if pose is not None:
                    poses.append((i, pose[0].tolist(), pose[1].tolist(), pose[2].tolist()))
            i += 1
        if tracker:
            tracker.close()
        if sum(m is not None for m in can_m) < 0.5 * len(can_m):
            log("   the can's mask was lost; trying SAM 2 again with the can's box alone...")
            retry = [measure_can(m[0]) if m[0] is not None else None for _, m in sam_track(path, [can_box], None, sam_class)]
            if sum(m is not None for m in retry) > sum(m is not None for m in can_m):
                can_m = retry
                perception["can_prompt"] = "box only (retry)"
        if can_m and can_m[0] is not None:   # does SAM's first mask match a real-size can standing where it touches the table?
            perception["can_mask_vs_known_shape_iou"] = mask_iou_poly(can_m[0]["outline"], can_silhouette(cam, base_on_table(cam, can_m[0]["bottom"])),
                                                                      first.shape[:2])
        pickle.dump({"can_m": can_m, "hand_pts": hand_pts, "poses": poses, "perception": perception}, open(cache, "wb"))
    n = len(can_m)
    cams = cameras_per_frame(calib, cam, corners, poses, n)
    drift = max([float(np.abs(np.asarray(p[3]) - corners).max()) for p in poses] or [0.0])
    moves = len({id(c) for c in cams}) - 1
    if len(poses) < 0.3 * (n / 15):
        warnings.append("the marker was hidden for most of the clip, so a bumped phone could not be corrected")
    iou = perception.get("can_mask_vs_known_shape_iou")
    if iou is not None and iou < 0.5:
        warnings.append(f"first can mask covers the can's known shape poorly (overlap {iou:.2f})")
    coaster = coaster_geo or (fit_coaster(cam, np.array(coaster_det["polygon"])) if coaster_det and coaster_det["polygon"] else None)
    if coaster is None:
        warnings.append("no coaster outline in the first frame")

    # 3. the can in 3D: on the table while resting, from its apparent width while carried
    log("   3/3 removing the hand from the can's mask and measuring the can in 3D...")
    can_m = clean_can_masks(path, can_m, hand_pts)
    can_ok = np.array([m is not None for m in can_m])
    if can_ok.mean() < 0.5:
        warnings.append(f"can segmented in only {100 * can_ok.mean():.0f}% of frames")
    if can_ok.sum() < 2:
        raise RuntimeError("the can was not segmented")
    raw = np.array([[*m["bottom"], m["width"], m["width_row"]] if m else [np.nan] * 4 for m in can_m], float)
    jump = np.abs(raw[:, 2] - rolling_median(raw[:, 2])) > 0.15 * rolling_median(raw[:, 2])   # one-frame width spikes
    raw[jump] = np.nan
    can_ok = can_ok & ~jump
    filled, can_valid = fill_gaps(raw, can_ok)
    bottom, width, width_row = smooth(filled[:, :2], 5), smooth(filled[:, 2:3], 5)[:, 0], smooth(filled[:, 3:4], 5)[:, 0]
    sec = int(fps)
    rest_start = np.array([base_on_table(cams[t], b) for t, b in enumerate(bottom)])
    end_z = COASTER_THICKNESS if coaster else 0.0
    rest_end = np.array([base_on_table(cams[t], b, end_z) for t, b in enumerate(bottom)])
    from_width = np.array([(base_from_width(cams[t], b, w_, r_) if w_ > 8 else None)
                           for t, (b, w_, r_) in enumerate(zip(bottom, width, width_row))], dtype=object)
    from_width = np.array([f_ if f_ is not None else [np.nan] * 3 for f_ in from_width], float)
    start = np.median(rest_start[:sec][can_ok[:sec]], 0) if can_ok[:sec].any() else rest_start[0]
    end = np.median(rest_end[-sec:][can_ok[-sec:]], 0) if can_ok[-sec:].any() else rest_end[-1]

    d_start = np.linalg.norm(rest_start[:, :2] - start[:2], axis=1)
    d_end = np.linalg.norm(rest_end[:, :2] - end[:2], axis=1)
    moved = np.flatnonzero(d_start > MOVE_START)
    can_s = np.repeat(start[None], n, 0)
    ev = dict.fromkeys(["grasp", "lift", "set_down", "settled", "release"])
    depth_check = None
    if len(moved) == 0:
        warnings.append("the can never moved")
    else:
        g = int(moved[0])
        while g > 0 and d_start[g - 1] > 0.004:
            g -= 1
        away = np.flatnonzero(d_end > MOVE_START)
        s = int(away[-1]) + 1 if len(away) else n - 1
        while s < n - 1 and d_end[s] > 0.004:
            s += 1
        # calibrate the size-based distance against the table in the first and last second of the clip, when your
        # hand is away from the can (frames right next to the grasp often have fingers on it)
        before, after = slice(0, max(1, min(sec, g))), slice(max(s, n - sec), n)
        bias0 = np.nanmean(rest_start[before] - from_width[before], 0) if g > 0 else np.zeros(3)
        bias1 = np.nanmean(rest_end[after] - from_width[after], 0) if s < n else bias0
        bias0, bias1 = np.nan_to_num(bias0), np.nan_to_num(bias1)
        depth_check = float(np.nanmean(np.linalg.norm(np.vstack([rest_start[before] - from_width[before],
                                                                 rest_end[after] - from_width[after]]), axis=1)))
        for j in range(g, s):
            a = (j - g) / max(s - g, 1)
            p = from_width[j] + (1 - a) * bias0 + a * bias1
            can_s[j] = p if np.all(np.isfinite(p)) else can_s[j - 1]
        can_s[s:] = end
        if s - g > 9:
            can_s[g:s] = smooth(can_s[g:s], 9)
        if can_s[:, 2].max() > 0.30:
            warnings.append(f"carried height reached {100 * can_s[:, 2].max():.0f} cm; check the annotated video")
        can_s[:, 2] = np.clip(can_s[:, 2], 0.0, 0.35)
        up = np.flatnonzero(can_s[g:s, 2] > LIFTED)
        ev.update(grasp=g, settled=s, lift=int(g + up[0]) if len(up) else g, set_down=int(g + up[-1] + 1) if len(up) else s)
        top_px = cams[-1].project(np.array([[end[0], end[1], end[2] + CAN_HEIGHT]]))[0]
        width_px = np.nanmedian(width[can_ok])
        far = 0
        for j in range(ev["set_down"], n):
            hp = hand_pts[j]
            gone = hp is None or np.min(np.linalg.norm(hp[FINGERTIPS] - top_px, axis=1)) > 1.0 * width_px
            far = far + 1 if gone else 0
            if far == 3:
                ev["release"] = j - 2
                break
        if ev["release"] is None:
            ev["release"] = min(s + int(0.3 * fps), n - 1)

    if coaster and np.linalg.norm(start[:2] - np.array(coaster["center"])) < coaster["diameter"] / 2:
        warnings.append("the can starts on the coaster, so this clip is not a move from elsewhere")
    if depth_check is not None and depth_check > 0.03:
        warnings.append(f"size-based distance disagrees with the table by {100 * depth_check:.1f} cm at rest; "
                        "carried heights in this clip are less reliable")
    end_offset = float(np.linalg.norm(end[:2] - np.array(coaster["center"]))) if coaster else None
    if end_offset is not None and end_offset > coaster["diameter"] / 2:
        warnings.append(f"can ended {100 * end_offset:.1f} cm from the coaster centre")
    result = {
        "clip": name, "reader_version": VERSION, "fps": fps, "n_frames": n, "image_size": list(cam.size),
        "perception": perception,
        "camera": {"rvec": cam.rvec.ravel().tolist(), "tvec": cam.tvec.ravel().tolist(), "position": cam.C.tolist(),
                   "marker_fit_error_px": marker_err, "max_drift_px": drift, "camera_moves_corrected": moves},
        "coaster": coaster,
        "can": {"start": start.tolist(), "end": end.tolist(), "found_fraction": float(can_ok.mean()),
                "end_offset_from_coaster": end_offset, "max_lift_height": float(can_s[:, 2].max()),
                "height": CAN_HEIGHT, "diameter": CAN_DIAMETER, "end_diameter": CAN_END_DIAMETER,
                "size_based_distance_error_at_rest": depth_check},
        "events": {**{f"{k}_frame": v for k, v in ev.items()},
                   **{f"{k}_time": (None if v is None else round(v / fps, 3)) for k, v in ev.items()}},
        "frames": {"t": (np.arange(n) / fps).round(4).tolist(), "can_xyz": np.round(can_s, 4).tolist(),
                   "can_found": can_ok.tolist(),
                   "hand_px": [None if h is None else np.round(h, 1).tolist() for h in hand_pts]},
        "warnings": warnings,
    }
    with open(os.path.join(out_dir, "demo.json"), "w") as f:
        json.dump(result, f)
    if video_on:
        log("   drawing the annotated video...")
        render(path, os.path.join(out_dir, "annotated.mp4"), cams, be, top_first, coaster, can_s, can_m,
               hand_pts, ev, end_offset, perception, fps, show)
    return result


# ---------------------------------------------------------------- annotated video
def draw_label(img, lines, xy, color, scale=0.75):
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 2
    w = max(cv2.getTextSize(t, font, scale, th)[0][0] for t in lines) + 20
    lh = int(32 * scale) + 6
    h = lh * len(lines) + 12
    x = int(min(max(xy[0], 5), img.shape[1] - w - 5))
    y = int(min(max(xy[1], 5), img.shape[0] - h - 5))
    roi = img[y:y + h, x:x + w]
    roi[:] = (roi * 0.35 + np.array([25, 25, 25]) * 0.65).astype(np.uint8)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (x + 10, y + lh * (k + 1)), font, scale, color if k == 0 else WHITE, th, cv2.LINE_AA)


def fill(img, poly, color, alpha):
    if poly is None or len(poly) < 3:
        return
    layer = img.copy()
    cv2.fillPoly(layer, [poly.reshape(-1, 1, 2)], color)
    x, y, w, h = cv2.boundingRect(poly.reshape(-1, 1, 2))
    img[y:y + h, x:x + w] = cv2.addWeighted(layer[y:y + h, x:x + w], alpha, img[y:y + h, x:x + w], 1 - alpha, 0)
    cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)


def render(path, out_path, cams, be, top_first, coaster, can_s, can_m, hand_pts, ev, end_offset,
           perception, fps, show=False):
    n = len(can_s)
    W, H = 1280, 720
    tw = int(be.w * H / be.h)
    plot_h = 110
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + tw, H + plot_h))
    k_top = H / be.h
    top_base = cv2.resize(top_first, (tw, H))
    if coaster:
        cv2.circle(top_base, pt(be.to_px([coaster["center"]])[0] * k_top), int(coaster["diameter"] / 2 * be.ppm * k_top), GREEN, 3)
    path_top = be.to_px(can_s[:, :2]) * k_top
    centres = np.array([m["centre"] if m else [np.nan, np.nan] for m in can_m], float)
    centres_s, centre_ok = fill_gaps(centres, ~np.isnan(centres[:, 0]))
    centres_s = smooth(centres_s, 5)
    who = f"found by {perception['can_found_by']}" + (f" {100 * perception['can_confidence']:.0f}%" if perception["can_confidence"] else "")
    phases = ["reach", "grasp", "lift & carry", "set down", "release"]
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    i = 0
    while ok and i < n:
        view = frame.copy()
        phase = sum(i >= ev[k] for k in ["grasp", "lift", "set_down", "release"]) if ev["grasp"] is not None else 0
        placed = ev["settled"] is not None and i >= ev["settled"]
        m = can_m[i]
        if coaster is not None:
            ring = coaster_ring(cams[i], coaster)
            disc = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(disc, [ring], 1)
            cover = skin(frame)
            if m is not None and m.get("outline") is not None:
                can_fill = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillPoly(can_fill, [m["outline"].reshape(-1, 1, 2)], 1)
                cover |= can_fill > 0
            if hand_pts[i] is not None:
                hand = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillConvexPoly(hand, cv2.convexHull(hand_pts[i].astype(np.int32)), 1)
                cover |= cv2.dilate(hand, np.ones((25, 25), np.uint8)) > 0
            tint(view, (disc > 0) & ~cover, GREEN, 0.45)
            label = (["PLACED ON TARGET", f"{100 * end_offset:.1f} cm from coaster centre"] if placed and end_offset is not None
                     else ["TARGET: coaster", f"diameter {100 * coaster['diameter']:.1f} cm"])
            draw_label(view, label, (ring[:, 0].max() + 15, ring[:, 1].max() - 40), GREEN)
        if m is not None:
            fill(view, m["outline"], BLUE, 0.45)
            b = can_s[i]
            draw_label(view, ["TARGET OBJECT: energy drink can", who,
                              f"cylinder, tapered ends  {100 * CAN_DIAMETER:.1f} / {100 * CAN_END_DIAMETER:.1f} x {100 * CAN_HEIGHT:.1f} cm",
                              "colour  light blue + silver",
                              f"x {100 * b[0]:6.1f}  y {100 * b[1]:6.1f}  z {100 * b[2]:5.1f} cm"],
                       (m["box"][2] + 15, m["box"][1]), BLUE)
        sel = centre_ok[:i + 1]
        if sel.sum() > 1:
            cv2.polylines(view, [np.round(centres_s[:i + 1][sel]).astype(np.int32)], False, YELLOW, 4, cv2.LINE_AA)
        hp = hand_pts[i]
        if hp is not None:
            for a_, b_ in HAND_LINKS:
                cv2.line(view, pt(hp[a_]), pt(hp[b_]), PINK, 3, cv2.LINE_AA)
            for q in hp:
                cv2.circle(view, pt(q), 6, PINK_DOT, -1, cv2.LINE_AA)
            draw_label(view, ["HAND", "21 tracked points"], (hp[:, 0].min() - 230, hp[:, 1].max() + 10), PINK)

        view = cv2.resize(view, (W, H))
        top = top_base.copy()
        if i > 0:
            cv2.polylines(top, [np.round(path_top[:i + 1]).astype(np.int32)], False, YELLOW, 4, cv2.LINE_AA)
        cv2.circle(top, pt(path_top[0]), 9, BLUE, -1)
        cv2.circle(top, pt(path_top[i]), 9, (0, 0, 255), -1)
        cv2.putText(top, "bird's-eye view: the can's path on the table", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    WHITE, 2, cv2.LINE_AA)

        strip = np.full((plot_h, W + tw, 3), 28, np.uint8)
        x0, x1, y0, y1 = 300, W + tw - 20, 15, plot_h - 15
        zmax = max(0.05, float(can_s[:, 2].max()))
        xs_plot = x0 + np.arange(n) / max(n - 1, 1) * (x1 - x0)
        cv2.polylines(strip, [np.column_stack([xs_plot, y1 - can_s[:, 2] / zmax * (y1 - y0)]).astype(np.int32)], False,
                      YELLOW, 2, cv2.LINE_AA)
        cv2.line(strip, pt((xs_plot[i], y0)), pt((xs_plot[i], y1)), WHITE, 2)
        cv2.putText(strip, f"can height {100 * can_s[i, 2]:4.1f} cm   (max {100 * zmax:.1f})", (x0 + 10, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1, cv2.LINE_AA)
        for k, name in enumerate(phases):
            cv2.putText(strip, f"{k + 1}. {name}", (12, 22 + 19 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 165, 255) if k == phase else GREY, 2 if k == phase else 1, cv2.LINE_AA)
        cv2.putText(strip, f"{i / fps:5.2f} s", (190, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        out = np.vstack([np.hstack([view, top]), strip])
        writer.write(out)
        if show:
            cv2.imshow("request_or_retry_learning: demo reader", cv2.resize(out, (out.shape[1] * 3 // 4, out.shape[0] * 3 // 4)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                show = False
                cv2.destroyAllWindows()
        ok, frame = cap.read()
        i += 1
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------- main
def summary_row(r):
    s, e, ev, c, pc = r["can"]["start"], r["can"]["end"], r["events"], r["can"], r["perception"]
    err, iou = c["size_based_distance_error_at_rest"], pc.get("can_mask_vs_known_shape_iou")
    return {"clip": r["clip"], "status": "ok" if not r["warnings"] else "check",
            "start_x": round(s[0], 4), "start_y": round(s[1], 4), "end_x": round(e[0], 4), "end_y": round(e[1], 4),
            **{k: ev[k] for k in ["grasp_time", "lift_time", "set_down_time", "release_time"]},
            "max_lift_height": round(c["max_lift_height"], 4),
            "end_offset_from_coaster": None if c["end_offset_from_coaster"] is None else round(c["end_offset_from_coaster"], 4),
            "coaster_diameter": None if r["coaster"] is None else round(r["coaster"]["diameter"], 4),
            "can_found_by": pc["can_found_by"], "can_confidence": pc["can_confidence"] and round(pc["can_confidence"], 3),
            "mask_vs_real_shape": None if iou is None else round(iou, 3),
            "distance_check_cm": None if err is None else round(100 * err, 2),
            "can_found_fraction": round(c["found_fraction"], 3), "warnings": "; ".join(r["warnings"])}


def describe(r):
    s, e, ev, c, pc = r["can"]["start"], r["can"]["end"], r["events"], r["can"], r["perception"]
    def fmt(x):
        return "-" if x is None else f"{x:.2f}s"
    err, iou = c["size_based_distance_error_at_rest"], pc.get("can_mask_vs_known_shape_iou")
    return (f"   can {pc['can_found_by']} (mask vs real shape {0 if iou is None else iou:.2f}), coaster {pc['coaster_found_by']}"
            f" | start ({100 * s[0]:.1f}, {100 * s[1]:.1f}) cm -> end ({100 * e[0]:.1f}, {100 * e[1]:.1f}) cm"
            f" | grasp {fmt(ev['grasp_time'])} lift {fmt(ev['lift_time'])} set down {fmt(ev['set_down_time'])}"
            f" release {fmt(ev['release_time'])} | max height {100 * c['max_lift_height']:.1f} cm"
            + ("" if c["end_offset_from_coaster"] is None else f" | {100 * c['end_offset_from_coaster']:.1f} cm from coaster centre")
            + ("" if err is None else f" | size-based distance check {100 * err:.1f} cm"))


FIELDS = ["clip", "status", "start_x", "start_y", "end_x", "end_y", "grasp_time", "lift_time", "set_down_time",
          "release_time", "max_lift_height", "end_offset_from_coaster", "coaster_diameter", "can_found_by",
          "can_confidence", "mask_vs_real_shape", "distance_check_cm", "can_found_fraction", "warnings"]


def main():
    global LOG_PATH
    p = argparse.ArgumentParser(description="Turn phone demo videos into robot-ready data.")
    p.add_argument("input", help="a video file or a folder of videos")
    p.add_argument("--calib", default="calibration.json", help="camera calibration file (default calibration.json)")
    p.add_argument("--out", default="out", help="output folder (default out)")
    p.add_argument("--no-hands", action="store_true", help="skip hand tracking")
    p.add_argument("--no-video", action="store_true", help="skip the annotated videos (faster)")
    p.add_argument("--show", action="store_true", help="play each annotated video live while it is made")
    p.add_argument("--redo", action="store_true", help="process every clip again, even ones already finished")
    p.add_argument("--retrack", action="store_true", help="redo detection, SAM 2 and hand tracking instead of using tracks.pkl")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    LOG_PATH = os.path.join(args.out, "log.txt")
    log(f"read_demos.py version {VERSION}")
    clips = (sorted(os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith((".mov", ".mp4", ".m4v")))
             if os.path.isdir(args.input) else [args.input])
    if not clips:
        sys.exit("No videos found.")
    models = None
    rows = []
    for k, path in enumerate(clips, 1):
        name = os.path.basename(path)
        log(f"[{k}/{len(clips)}] {name}")
        done = os.path.join(args.out, os.path.splitext(name)[0], "demo.json")
        r = None
        if not args.redo and os.path.exists(done):
            old = json.load(open(done))
            if old.get("reader_version") == VERSION:
                r = old
                log("   already done with this version, skipping (use --redo to process it again)")
        if r is None:
            models = models or load_models()
            try:
                r = process_clip(path, args.calib, args.out, models, hands_on=not args.no_hands,
                                 video_on=not args.no_video, show=args.show, retrack=args.retrack)
            except Exception as e:
                log(f"   FAILED: {e}")
                rows.append({"clip": name, "status": "failed", "warnings": str(e)})
                r = None
        if r is not None:
            log(describe(r))
            for w in r["warnings"]:
                log(f"   WARNING: {w}")
            rows.append(summary_row(r))
        with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:     # saved after every clip
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    flagged = [r for r in rows if r.get("status") != "ok"]
    if flagged:
        log("\nClips with a note (the rest are clean):")
        for r in flagged:
            log(f"   {r['clip']}: {r.get('warnings', '')}")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    log(f"\nDone: {ok}/{len(rows)} clips clean. Results in {args.out}/ (summary.csv, log.txt, and per clip: demo.json, "
        "scene.json, annotated.mp4)")


if __name__ == "__main__":
    main()
