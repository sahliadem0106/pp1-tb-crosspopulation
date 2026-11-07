#!/usr/bin/env python3
# PP5 Part 1 RECON - NIH Infiltration bounding-box subset
# Run in a Kaggle notebook (CPU is fine; CSV-only, no images).
# Downloads BBox_List_2017.csv, filters the infiltrate class, reports the count
# and the RUL vs non-RUL split. This locks the localization design BEFORE we
# write any pointing-game/IoU code.
#
# Note: the bbox CSV labels the class "Infiltrate" (the finding CSV calls it
# "Infiltration"). We match the shared substring to be safe.

import pandas as pd

# --- 1. Download BBox_List_2017.csv directly (small; only the CSV is needed for recon) ---
url = ("https://raw.githubusercontent.com/gregwchase/nih-chest-xray/master/"
       "data/BBox_List_2017.csv")
df = pd.read_csv(url, header=None, skiprows=1,
                 names=["Image Index", "Finding Label", "x", "y", "w", "h"])
print("Total bbox rows:", len(df))
print("Label counts:\n", df["Finding Label"].value_counts())

# --- 2. Filter the infiltrate class ---
inf = df[df["Finding Label"].str.lower().str.contains("infiltrat")].copy()
print("\n=== INFILTRATE bboxes ===")
print("Infiltrate bbox rows:", len(inf))
print("Infiltrate unique images:", inf["Image Index"].nunique())

# --- 3. RUL / non-RUL split ---
# NIH bboxes are [x,y,w,h] top-left, in original pixel space (~1024x1024).
# Frontal CXR: patient's RIGHT appears on the LEFT of the image (viewer's left).
# RUL = box center in the UPPER half AND the patient's-RIGHT (viewer-left) half.
cx = inf["x"] + inf["w"] / 2.0
cy = inf["y"] + inf["h"] / 2.0
is_upper = cy < 512.0
is_patient_right = cx < 512.0   # viewer-left = patient's right
rul = is_upper & is_patient_right

inf["is_upper"] = is_upper
inf["is_patient_right"] = is_patient_right
inf["is_RUL"] = rul

print("\nRUL (upper + patient-right) bboxes:", int(rul.sum()))
print("non-RUL bboxes:", int((~rul).sum()))
print("RUL proportion: %.3f" % rul.mean())
print("\nQuadrant breakdown (rows):")
inf["quadrant"] = (
    inf["is_upper"].map({True: "Upper", False: "Lower"}) + " " +
    inf["is_patient_right"].map({True: "Patient-Right", False: "Patient-Left"})
)
print(inf["quadrant"].value_counts())

# --- 4. Save the filtered list for downstream Part 1 ---
out = "infiltration_bboxes.csv"
inf.to_csv(out, index=False)
print("\nSaved:", out)
print("DONE - design is now locked on real numbers, not assumptions.")
