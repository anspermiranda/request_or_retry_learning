#!/usr/bin/env python3
"""
Collect every clip's results (out/<clip>/demo.json) into one table, with the quality scores.

    python collect_results.py            # reads out/, writes out/results_table.csv and prints the table
"""
import csv
import glob
import json
import os
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "out"
rows = []
for path in sorted(glob.glob(os.path.join(out, "*", "demo.json"))):
    r = json.load(open(path))
    c, pc, ev = r["can"], r.get("perception", {}), r["events"]
    iou, err = pc.get("can_mask_vs_known_shape_iou"), c.get("size_based_distance_error_at_rest")
    rows.append({"clip": r["clip"], "found_by": pc.get("can_found_by", "-"),
                 "yoloe_conf": "" if pc.get("can_confidence") is None else round(pc["can_confidence"], 2),
                 "mask_vs_shape": "" if iou is None else round(iou, 2),
                 "distance_check_cm": "" if err is None else round(100 * err, 1),
                 "start_x_cm": round(100 * c["start"][0], 1), "start_y_cm": round(100 * c["start"][1], 1),
                 "max_height_cm": round(100 * c["max_lift_height"], 1),
                 "from_coaster_centre_cm": "" if c.get("end_offset_from_coaster") is None else round(100 * c["end_offset_from_coaster"], 1),
                 "grasp_s": ev.get("grasp_time"), "release_s": ev.get("release_time"),
                 "warnings": len(r.get("warnings", []))})
if not rows:
    sys.exit(f"No results found in {out}/")
with open(os.path.join(out, "results_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

cols = ["clip", "found_by", "yoloe_conf", "mask_vs_shape", "distance_check_cm", "max_height_cm", "from_coaster_centre_cm", "warnings"]
width = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in cols}
print("  ".join(k.ljust(width[k]) for k in cols))
for r in rows:
    print("  ".join(str(r[k]).ljust(width[k]) for k in cols))
ious = [r["mask_vs_shape"] for r in rows if r["mask_vs_shape"] != ""]
yoloe = sum(1 for r in rows if r["found_by"] == "YOLOE")
print(f"\n{len(rows)} clips | can recognised by YOLOE in {yoloe}, colour fallback in {len(rows) - yoloe}"
      + (f" | mask vs real shape: average {sum(ious) / len(ious):.2f}, {sum(i >= 0.7 for i in ious)} clips at 0.7 or better" if ious else "")
      + f" | clips with warnings: {sum(1 for r in rows if r['warnings'])}")
print(f"Saved {out}/results_table.csv")
