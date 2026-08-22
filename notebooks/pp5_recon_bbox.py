#!/usr/bin/env python3
# PP5 Part 1 RECON - NIH Infiltration bounding-box subset
# Run in a Kaggle notebook (GPU/TPU off is fine; this is CSV-only, no images).
# Pulls BBox_List_2017.csv, filters Infiltration, reports count + RUL/non-RUL split.
# This locks the localization design BEFORE we write any pointing-game/IoU code.

import kagglehub
import pandas as pd
import glob, os

# --- 1. Download the NIH dataset and locate BBox_List_2017.csv ---
path = kagglehub.dataset_download("khanfashee/nih-chest-x-ray-14-224x224-resized")
print("Dataset root:", path)

cands = glob.glob(os.path.join(path, "**", "BBox_List_2017.csv"), recursive=True)
if not cands:
    # fallback: the original NIH ChestX-ray14 dataset (full res) if the resized one lacks bboxes
    print("BBox_List_2017.csv not in resized set; trying full NIH dataset...")
    path = kagglehub.dataset_download("nih-chest-xray-images")
    cands = glob.glob(os.path.join(path, "**", "BBox_List_2017.csv"), recursive=True)

assert cands, "Could not locate BBox_List_2017.csv in either dataset"
bbox_csv = cands[0]
print("Found bbox CSV:", bbox_csv)

# --- 2. Load and filter Infiltration ---
df = pd.read_csv(bbox_csv)
df.columns = [c.strip() for c in df.columns]
print("Total rows in BBox_List_2017.csv:", len(df))
print("Label counts:\n", df["Finding Label"].value_counts())

inf = df[df["Finding Label"].str.strip().str.lower() == "infiltration"].copy()
print("\n=== INFLITRATION bboxes ===")
print("Infiltration bbox rows:", len(inf))
print("Infiltration unique images:", inf["Image Index"].nunique())

# --- 3. RUL / non-RUL split ---
# NIH bboxes are [x,y,w,h] top-left, in original pixel space (~1024x1024).
# Frontal CXR: patient's RIGHT appears on the LEFT of the image (viewer's left).
# RUL = box center in the UPPER half AND the patient's-RIGHT (viewer-left) half.
# We use the raw pixel coords from the CSV (no resizing needed for the split).
cx = inf["x"] + inf["w"] / 2.0
cy = inf["y"] + inf["h"] / 2.0
# image is 1024x1024 in the annotation space
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
    inf["is_upper"].map({True:"Upper",False:"Lower"}) + " " +
    inf["is_patient_right"].map({True:"Patient-Right",False:"Patient-Left"})
)
print(inf["quadrant"].value_counts())

# --- 4. Save the filtered Infiltration bbox list for downstream Part 1 ---
out = "infiltration_bboxes.csv"
inf.to_csv(out, index=False)
print("\nSaved:", out)
print("DONE - design is now locked on real numbers, not assumptions.")
