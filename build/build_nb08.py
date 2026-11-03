# Builder of Notebook 08 of the course "AI Technologies (Basic Course)" (AIT-BC).
# Topic 5, Lessons 5.1 (lecture) and 5.2 (group class). All exercises use real open data sets.
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n"), False))


def code(text, form=False):
    CELLS.append(("code", text.strip("\n"), form))


# =====================================================================================
md(r"""
# AI Technologies (Basic Course) · Notebook 08
## Topic 5. Neural-network detection, segmentation and tracking of objects in the video of UAVs and UGVs

| Lesson | Type, duration | Title |
|---|---|---|
| **5.1** | Lecture, 2 h | Object detection by YOLO neural networks: IoU, confidence, non-maximum suppression, mAP and the detection of small targets |
| **5.2** | Group class, 2 h | Segmentation of the traversable area for a ground robotic platform, tracking of objects (ByteTrack), the video-analysis pipeline and the requirements of real time |

Course: *AI Technologies (Basic Course)*, AIT-BC · Content module 2 · Kruty Heroes Military Institute of Telecommunications and Information Technology, Kyiv, 2026.

**How to work with this notebook**

1. Run the cells from top to bottom: *Runtime → Run all*, or `Shift+Enter` cell by cell. Section 0 is always run first; after a restart of the runtime it is run again.
2. Every block begins with a text cell: the purpose, the formulas, the parameters of the form, the expected result and the control question. The parameters are changed in the form of the cell, and the cell is run again.
3. The notebook runs on the processor of Colab in about 10–15 minutes; with a GPU (*Runtime → Change runtime type → T4 GPU*) the detection blocks are several times faster.
4. The results are saved in the course project on Google Drive (`outputs`, `models`, `project.json`), created in Notebook 02.

**Data.** Only real images and videos are used, with the annotation of the authors of the data sets:
- **DUT Anti-UAV** — 10 000 images of drones taken by ground cameras against the sky, buildings and trees (counter-UAV task), with boxes of the class UAV;
- **M3OT** — video of DJI Mavic 3T drones at 100–120 m over roads and crossroads with tracks of vehicles;
- **GOOSE** — front camera of a ground vehicle in unstructured terrain (field tracks, forest, a training area) with pixel labels of 64 classes.

The detector is **YOLO11** of the library Ultralytics, trained by its authors on the data set COCO (80 everyday classes). In this notebook it is used as it is; its training for a new class is the mini-project of Lesson 5.4 (Notebook 09).

**Contents**

- **0. Preparation** — 0.1 runtime, course library, YOLO and data sets · 0.2 course project
- **Lesson 5.1 (lecture)**
  - 5.1.1 The task of detection, drones in the images of a ground camera and the output grid of YOLO
  - 5.1.2 Intersection over union (IoU)
  - 5.1.3 Confidence of a detection and non-maximum suppression
  - 5.1.4 Precision, recall, AP and mAP
  - 5.1.5 Small targets: size of the input, slicing of the frame and the cost in time
- **Lesson 5.2 (group class)**
  - 5.2.1 Terrain classes and the traversable area of a ground robotic platform
  - 5.2.2 Segmentation network LR-ASPP MobileNetV3: transfer learning and quality
  - 5.2.3 Tracking of objects by detection: Kalman filter, association, ByteTrack
  - 5.2.4 Video-analysis pipeline and the requirements of real time
- **Summary**: self-check, data sources and licences
""")

# =====================================================================================
md(r"""
---
# 0. Preparation

### Block 0.1. Runtime, course library, YOLO and data sets

**Purpose.** To prepare the runtime: the libraries, the library **Ultralytics** (models YOLO; it is installed in the session if it is absent), the course library `aitbc` (version 2.7.0 adds the data set DUT Anti-UAV, the metrics of detection and tracking, slicing of frames, the tracker ByteTrack, the segmentation network and a viewing window `crop_view`) and the data sets of the lessons.

**Parameters.** `CACHE_ON_DRIVE` — keep the downloaded files on Google Drive (they remain after the end of the session); otherwise they are kept on the disk of the session.

**Output.** A table of versions, the computing device and the size of the cache. The weights of the YOLO models (5–20 MB) are downloaded from the repository of Ultralytics when they are first used.

*Licence note.* Ultralytics and the weights of YOLO11 are distributed under AGPL-3.0; they are installed from PyPI in the session and are not part of the course materials.
""")

code(r"""
#@title 0.1 Runtime, course library, YOLO and data sets (run once) { display-mode: "form" }
CACHE_ON_DRIVE = False #@param {type:"boolean"}

import sys, os, platform, time, json, math, importlib, shutil, warnings, subprocess, collections
import numpy as np
import pandas as pd
import torch
import torchvision
import cv2
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
pd.set_option("display.max_colwidth", 200)

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
WORK_DIR = "/content" if IN_COLAB else os.getcwd()

try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    import ultralytics
from ultralytics import YOLO
from ultralytics import settings as yolo_settings
yolo_settings.update({"sync": False})

LIB_SOURCE = r'''%%LIB%%'''

with open(os.path.join(WORK_DIR, "aitbc.py"), "w", encoding="utf-8") as fh:
    fh.write(LIB_SOURCE)
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)
import aitbc
aitbc = importlib.reload(aitbc)
aitbc.setup_plots()

if os.environ.get("AITBC_DATA"):
    cache_dir = os.environ["AITBC_DATA"]
elif CACHE_ON_DRIVE and IN_COLAB:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    cache_dir = "/content/drive/MyDrive/AITBC/data_cache"
else:
    cache_dir = os.path.join(WORK_DIR, "aitbc_data")
aitbc.set_data_root(cache_dir)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
WEIGHTS_DIR = os.environ.get("AITBC_WEIGHTS", os.path.join(WORK_DIR, "yolo_weights"))
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def load_yolo(name):
    # model YOLO by the name of its weights file; the file is downloaded once into WEIGHTS_DIR
    return YOLO(os.path.join(WEIGHTS_DIR, name))


t0 = time.time()
dut = aitbc.DUTAntiUAV()       # drones seen from the ground (counter-UAV task)
m3 = aitbc.M3OT()              # video of UAVs over roads with tracks of vehicles
goose = aitbc.GOOSE()          # camera of a ground vehicle with pixel labels of the terrain
VERSIONS = {"Python": platform.python_version(), "NumPy": np.__version__, "OpenCV": cv2.__version__,
            "pandas": pd.__version__, "PyTorch": torch.__version__, "torchvision": torchvision.__version__,
            "Ultralytics": ultralytics.__version__, "aitbc": aitbc.__version__}
n_files, size_mb = aitbc.cache_size()
dev = torch.cuda.get_device_name(0) if DEVICE == "cuda" else f"CPU, {os.cpu_count()} logical cores"
display(pd.DataFrame({"Value": list(VERSIONS.values()) + ["Google Colab" if IN_COLAB else "local Jupyter", dev,
                                                           f"{aitbc.DATA_ROOT}: {n_files} files, {size_mb:.0f} MB",
                                                           f"{time.time() - t0:.0f} s"]},
                     index=list(VERSIONS) + ["environment", "computing device", "cache of the data sets", "preparation time"]))
""", form=True)

md(r"""
### Block 0.2. Course project

**Purpose.** To connect the course project created in Notebook 02. The results of this notebook are saved in it: figures and tables in `outputs`, the trained output layers of the segmentation network in `models`, the video with the tracks in `outputs/video` and the parameters of the lessons in `project.json`.

**Parameters.** `USE_DRIVE` — the project is on Google Drive (Colab asks for access to the Google account); otherwise it is on the disk of the session. `PROJECT_PATH` — folder of the project; the same as in Notebooks 02–07.

**Output.** The folders of the project and the lessons already recorded in `project.json`.
""")

code(r"""
#@title 0.2 Course project { display-mode: "form" }
USE_DRIVE = True #@param {type:"boolean"}
PROJECT_PATH = "AITBC/AITBC_project" #@param {type:"string"}

if USE_DRIVE and IN_COLAB:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    PROJECT_ROOT = os.path.join("/content/drive/MyDrive", PROJECT_PATH)
else:
    PROJECT_ROOT = os.path.join(WORK_DIR, PROJECT_PATH)
    if USE_DRIVE:
        print("Google Drive is available only in Google Colab; the project is used in", PROJECT_ROOT)
for d in ["data/images", "data/video", "data/thermal", "data/annotations", "frames/visible", "frames/thermal",
          "models", "outputs/figures", "outputs/video", "outputs/reports"]:
    os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "outputs", "video")


def update_project_json(lesson, extra=None):
    path = os.path.join(PROJECT_ROOT, "project.json")
    meta = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"project": os.path.basename(PROJECT_ROOT)}
    meta.setdefault("lessons", [])
    if lesson not in meta["lessons"]:
        meta["lessons"].append(lesson)
    meta["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["versions"] = VERSIONS
    meta.update(extra or {})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    print(f"project.json updated: lesson {lesson}" + (f", keys: {', '.join(extra)}" if extra else ""))


meta = json.load(open(os.path.join(PROJECT_ROOT, "project.json"), encoding="utf-8")) if os.path.exists(os.path.join(PROJECT_ROOT, "project.json")) else {}
print("Project:", PROJECT_ROOT)
print("Lessons recorded in project.json:", ", ".join(meta.get("lessons", [])) or "none (new project)")
""", form=True)

# =====================================================================================
md(r"""
---
# Lesson 5.1 (Lecture). Object detection by YOLO neural networks

**Learning questions**
1. The task of object detection and the one-stage detector YOLO.
2. Intersection over union (IoU) of two boxes.
3. Confidence of a detection and non-maximum suppression (NMS).
4. Precision, recall, average precision (AP) and mAP.
5. Detection of small targets: size of the input and slicing of the frame.

**Situation of the lesson.** A post of air defence observes the sky with a daylight camera. A small drone at a distance of several hundred metres occupies 10–40 pixels of a frame of 1920 × 1080. A detector must find it in every frame, give its box and a confidence, and not raise alarms on birds, clouds and branches. The lecture shows how a neural detector of the family YOLO solves this task, how its result is measured and why small targets are the main difficulty.
""")

md(r"""
## Block 5.1.1. The task of detection, drones in the images of a ground camera and the output grid of YOLO

**Detection** gives for an image a list of objects, each described by a box $(x_1, y_1, x_2, y_2)$, a class and a confidence $s \in [0, 1]$. A classifier (Topic 4) answers *what* is in the image; a detector answers *what* and *where*, for any number of objects.

**YOLO** (*You Only Look Once*) is a one-stage detector: the image is passed through the network once, and every cell of three output grids predicts a box and the scores of the classes. For the input $S \times S$ the grids have the steps (strides) $8$, $16$ and $32$ pixels, i.e. $\frac{S}{8}\times\frac{S}{8}$, $\frac{S}{16}\times\frac{S}{16}$ and $\frac{S}{32}\times\frac{S}{32}$ cells; the fine grid (stride 8) is responsible for small objects. YOLO11 predicts for a cell the distances from its centre to the four sides of the box and the scores of the 80 classes of COCO; the overlapping predictions are removed by non-maximum suppression (block 5.1.3).

**Data.** The detection part of **DUT Anti-UAV** has the official parts train (5200 images), val (2600) and test (2200). The images of drones were taken by ground cameras from different distances, against the sky, clouds, buildings and trees. The annotation gives one box per drone.

**Parameters.** `N_SHOW` — images in the gallery; `SEED` — choice of the images; `GRID_IMAGE` — number of the image in the gallery for which the output grids are drawn (−1 — the image with the smallest drone).

**Expected result.** The table of the parts of the data set with the sizes of the boxes; a gallery of test images with the enlarged fragments of the drones; the distribution of the sizes of the boxes; the grids of the strides 8, 16 and 32 over a drone in the input 640 × 640.

**What to observe.** In the training and validation parts about half of the drones are smaller than 32 px (median side 31 px in frames that are mostly 1920 × 1080); the test part has larger drones (median 41 px, 38 % smaller than 32 px). Most drones are dark spots on the sky; near buildings and trees they are hard to see even for a person. The grid of the smallest drone of the gallery shows the main problem: a drone of 8 × 7 px in the input 640 × 640 covers one cell of the stride 8 and only a quarter of a cell of the stride 32.

**Control question 1.** How many cells of each output grid of YOLO cover a drone of 20 × 20 pixels in a frame of 1920 × 1080 reduced to the input 640 × 640, and which grid can find it?
""")

code(r"""
#@title 5.1.1 Drones in the images of a ground camera and the output grid of YOLO { display-mode: "form" }
N_SHOW = 6 #@param {type:"slider", min:3, max:9, step:3}
SEED = 0 #@param {type:"integer"}
GRID_IMAGE = -1 #@param {type:"integer"}

ann = {s: dut.annotations(s) for s in ("train", "val", "test")}


def box_sides(boxes_list):
    return np.array([math.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0)) for bs in boxes_list for b in bs])


rows = []
SIDES = {}
for s, tb in ann.items():
    sd = box_sides(tb.boxes)
    SIDES[s] = sd
    rows.append({"part": s, "images": len(tb), "boxes": int(tb.n.sum()), "images without a drone": int((tb.n == 0).sum()),
                 "frames 1920 x 1080": f"{np.mean((tb.width == 1920) & (tb.height == 1080)):.0%}",
                 "median side of a box, px": round(float(np.median(sd)), 1),
                 "side < 32 px": f"{np.mean(sd < 32):.0%}", "32-96 px": f"{np.mean((sd >= 32) & (sd < 96)):.0%}",
                 ">= 96 px": f"{np.mean(sd >= 96):.0%}"})
display(pd.DataFrame(rows).set_index("part"))

rng = np.random.default_rng(SEED)
test = ann["test"]
show = rng.choice(np.where(test.n > 0)[0], N_SHOW, replace=False)
paths = dut.image_paths("test", test.image_id.values[show])
fig, axes = plt.subplots(2, N_SHOW, figsize=(2.7 * N_SHOW, 4.4), gridspec_kw={"height_ratios": [1.2, 1]})
for k, (i, p) in enumerate(zip(show, paths)):
    img = aitbc.to_rgb(cv2.imread(p)[..., ::-1])
    bs = test.boxes[i]
    axes[0, k].imshow(aitbc.draw_boxes(img, [{"box": b, "cls": "UAV"} for b in bs], color=(255, 40, 40),
                                       labels=False, thickness=max(2, img.shape[1] // 400)))
    axes[0, k].set_title(f"{test.image_id[i]}: {img.shape[1]}x{img.shape[0]}", fontsize=8)
    b = bs[0]
    side = max(b[2] - b[0], b[3] - b[1])
    axes[1, k].imshow(aitbc.crop_view(img, b, scale=4, min_side=48))
    axes[1, k].set_title(f"drone {b[2] - b[0]:.0f} x {b[3] - b[1]:.0f} px", fontsize=8)
for a in axes.ravel():
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB08_511_gallery.png"), dpi=110)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
bins = np.logspace(np.log10(4), np.log10(1000), 40)
for s, c in zip(SIDES, ("#1565c0", "#ef6c00", "#2e7d32")):
    ax[0].hist(SIDES[s], bins=bins, histtype="step", lw=1.6, color=c, label=f"{s} ({len(SIDES[s])} boxes)")
for v in (32, 96):
    ax[0].axvline(v, color="grey", ls="--", lw=1)
ax[0].set_xscale("log")
ax[0].set_xlabel("side of the box sqrt(w*h), px (frame of the data set)")
ax[0].set_ylabel("boxes")
ax[0].legend(fontsize=8)
ax[0].set_title("Sizes of the drones: small < 32 px, medium 32-96 px, large >= 96 px", fontsize=9)

sides_show = [box_sides([test.boxes[j]]).min() for j in show]
gi = int(np.argmin(sides_show)) if GRID_IMAGE < 0 else GRID_IMAGE % len(show)
i = show[gi]
img = cv2.imread(paths[gi])[..., ::-1]
lb, r, pad = aitbc.letterbox(img, 640)
b = aitbc.boxes_to_input([test.boxes[i][0]], r, pad)[0]
cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
half = 48
x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
crop = lb[y0:y0 + 2 * half, x0:x0 + 2 * half]
ax[1].imshow(crop, extent=(x0, x0 + crop.shape[1], y0 + crop.shape[0], y0))
for stride, col, lw in ((8, "#2e7d32", 0.6), (16, "#ef6c00", 1.0), (32, "#c62828", 1.8)):
    for g in range(0, 641, stride):
        ax[1].axvline(g, color=col, lw=lw, alpha=0.8)
        ax[1].axhline(g, color=col, lw=lw, alpha=0.8)
ax[1].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="cyan", lw=2))
ax[1].set_xlim(x0, x0 + crop.shape[1]); ax[1].set_ylim(y0 + crop.shape[0], y0)
ax[1].set_title(f"input 640 x 640 (r = {r:.3f}): drone {b[2] - b[0]:.1f} x {b[3] - b[1]:.1f} px; "
                f"grids: stride 8 green, 16 orange, 32 red", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB08_511_sizes_grid.png"), dpi=110)
plt.show()
w_in, h_in = b[2] - b[0], b[3] - b[1]
print(f"Drone of the image {test.image_id[i]} in the input 640 x 640: {w_in:.1f} x {h_in:.1f} px = "
      + ", ".join(f"{w_in / s:.2f} x {h_in / s:.2f} cells of stride {s}" for s in (8, 16, 32)))
""", form=True)

md(r"""
## Block 5.1.2. Intersection over union (IoU)

**Definition.** For a predicted box $B_p$ and a true (annotated) box $B_t$
$$\mathrm{IoU}(B_p, B_t) = \frac{|B_p \cap B_t|}{|B_p \cup B_t|} = \frac{|B_p \cap B_t|}{|B_p| + |B_t| - |B_p \cap B_t|}\in[0,1].$$
A detection is **correct** (true positive) if its IoU with a free true box is not less than the threshold $\tau$: $\tau = 0.5$ in the measure AP50 (Pascal VOC), $\tau = 0.5, 0.55, \dots, 0.95$ in the measure AP50-95 of COCO.

For two squares of side $a$ shifted by $d$ pixels along one axis $\mathrm{IoU} = \frac{a-d}{a+d}$; the shift that gives IoU $= 0.5$ is $d = a/3$. For a drone of 12 px this is 4 px, for a vehicle of 120 px it is 40 px: **the same error of a few pixels is negligible for a large object and fatal for a small one.**

**Parameters.** `SHIFT_X_PX`, `SHIFT_Y_PX` — shift of the predicted box relative to the true box of a real drone; `SCALE` — ratio of the sizes of the predicted and the true box; `IMAGE_NUMBER` — number of the test image of the gallery of block 5.1.1.

**Expected result.** The true and the predicted box on the enlarged fragment with the value of IoU; the curves of IoU against the shift for squares of 8, 16, 32, 64 and 128 px and the thresholds 0.5 and 0.75.

**What to observe.** For the default real drone (a box of about 42 px) a shift of 4 px gives IoU = 0.83, and the shift that gives IoU = 0.5 is 14 px. On the curves, a shift of 4 px brings a square of 8 px to IoU = 0.33 (an error at the threshold 0.5), while a square of 128 px keeps IoU = 0.94. A scale error acts in the same way: a box twice as large as the true one and with the same centre gives IoU = 0.25.

**Control question 2.** Why does COCO average AP over the thresholds 0.5–0.95, and why is the threshold 0.5 often used for small targets?
""")

code(r"""
#@title 5.1.2 IoU of a predicted and a true box { run: "auto", display-mode: "form" }
SHIFT_X_PX = 4 #@param {type:"slider", min:-30, max:30, step:1}
SHIFT_Y_PX = 0 #@param {type:"slider", min:-30, max:30, step:1}
SCALE = 1.0 #@param {type:"slider", min:0.5, max:2.0, step:0.05}
IMAGE_NUMBER = 0 #@param {type:"integer"}

i = show[IMAGE_NUMBER % len(show)]
img = cv2.imread(paths[IMAGE_NUMBER % len(show)])[..., ::-1]
t_box = np.array(test.boxes[i][0], float)
cx, cy = (t_box[0] + t_box[2]) / 2 + SHIFT_X_PX, (t_box[1] + t_box[3]) / 2 + SHIFT_Y_PX
w, h = (t_box[2] - t_box[0]) * SCALE, (t_box[3] - t_box[1]) * SCALE
p_box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
iou = aitbc.box_iou_matrix([p_box], [t_box])[0, 0]

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
side = max(t_box[2] - t_box[0], t_box[3] - t_box[1])
c = ((t_box[0] + t_box[2]) / 2, (t_box[1] + t_box[3]) / 2)
half = max(2.5 * side, 24)
ax[0].imshow(img)
ax[0].add_patch(plt.Rectangle(t_box[:2], *(t_box[2:] - t_box[:2]), fill=False, ec="lime", lw=2, label="true box"))
ax[0].add_patch(plt.Rectangle(p_box[:2], *(p_box[2:] - p_box[:2]), fill=False, ec="red", lw=2, ls="--", label="predicted box"))
ax[0].set_xlim(max(0, c[0] - half), min(img.shape[1], c[0] + half)); ax[0].set_ylim(min(img.shape[0], c[1] + half), max(0, c[1] - half))
ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_title(f"true {t_box[2] - t_box[0]:.0f} x {t_box[3] - t_box[1]:.0f} px; IoU = {iou:.3f} "
                f"({'correct' if iou >= 0.5 else 'error'} at 0.5)", fontsize=9)
d = np.arange(0, 61)
for a in (8, 16, 32, 64, 128):
    ax[1].plot(d, np.clip((a - d) / (a + d), 0, 1), lw=1.8, label=f"square {a} px")
for t in (0.5, 0.75):
    ax[1].axhline(t, color="grey", ls="--", lw=1)
ax[1].set_xlabel("shift of the predicted box along one axis, px")
ax[1].set_ylabel("IoU")
ax[1].legend(fontsize=8)
ax[1].set_title("IoU = (a - d) / (a + d): the same shift in pixels against the size of the object", fontsize=9)
plt.tight_layout()
plt.show()
print(f"IoU = {iou:.3f}; shift giving IoU = 0.5 for this box along x: {(t_box[2] - t_box[0]) / 3:.1f} px")
""", form=True)

md(r"""
## Block 5.1.3. Confidence of a detection and non-maximum suppression

**Confidence.** For every cell and class YOLO gives a score $s \in [0,1]$; a box is kept if $s \geq$ `CONF`. A low threshold keeps weak true detections and many false ones; a high threshold removes false detections and misses weak targets.

**Non-maximum suppression (NMS).** Neighbouring cells predict almost the same box of one object. NMS sorts the boxes by score and repeats: the box with the highest score is kept, the boxes with IoU $>$ `NMS_IOU` with it are removed. A low `NMS_IOU` removes the duplicates but can merge two close objects (a group of drones); a high `NMS_IOU` keeps the duplicates.

**Classes of COCO.** The YOLO11 model trained on COCO has no class *drone*. The images of DUT Anti-UAV show what such a model does with a drone: it assigns the nearest classes it knows — *airplane*, *bird*, *kite*. The boxes of these three classes are taken in this lecture as detections of a drone (class-agnostic evaluation).

**Parameters.** `MODEL` — YOLO11n (2.6 M parameters) or YOLO11s (9.4 M); `IMAGE_NUMBER` — test image of the gallery (−1 — the image where the drone gets the highest score); `CONF` — threshold of the confidence; `NMS_IOU` — threshold of NMS; the raw predictions are obtained with the confidence 0.01 and NMS switched off.

**Expected result.** The raw boxes and the boxes after NMS on the fragment of the drone, the table of the kept boxes with their classes and scores.

**What to observe.** In the image with the highest score of a drone (a large drone close to the camera), YOLO11n gives 17 raw boxes, 10 of them with a confidence of at least 0.25. NMS with the threshold 0.5 leaves one box: the class *airplane* with the confidence 0.87 and IoU 0.98 with the true box. The model has no class drone, so it calls the drone by the nearest class of COCO. Set `NMS_IOU` = 0.9 and then `CONF` = 0.05 and count the boxes again.

**Control question 3.** Why does NMS reduce the number of boxes on one object to one, and in which situation of a group of targets is it harmful?
""")

code(r"""
#@title 5.1.3 Raw predictions, threshold of the confidence and NMS { display-mode: "form" }
MODEL = "yolo11n.pt" #@param ["yolo11n.pt", "yolo11s.pt"]
IMAGE_NUMBER = -1 #@param {type:"integer"}
CONF = 0.25 #@param {type:"slider", min:0.01, max:0.9, step:0.01}
NMS_IOU = 0.5 #@param {type:"slider", min:0.1, max:0.95, step:0.05}

COCO_UAV = [4, 14, 33]  # airplane, bird, kite: the nearest classes of COCO for a drone
yolo = load_yolo(MODEL)
RAW = []
for k, p in enumerate(paths):
    im = cv2.imread(p)[..., ::-1].copy()
    rb, rs, rc, _ = aitbc.yolo_predict(yolo, im, imgsz=1280, conf=0.01, iou=0.999, max_det=1000)
    iou_t = aitbc.box_iou_matrix(rb, [test.boxes[show[k]][0]])[:, 0] if len(rb) else np.zeros(0)
    RAW.append((im, rb, rs, rc, float(rs[iou_t >= 0.3].max()) if np.any(iou_t >= 0.3) else 0.0))
k_img = int(np.argmax([r[4] for r in RAW])) if IMAGE_NUMBER < 0 else IMAGE_NUMBER % len(show)
i = show[k_img]
img, raw_b, raw_s, raw_c, _ = RAW[k_img]
keep_conf = raw_s >= CONF
k = aitbc.nms(raw_b[keep_conf], raw_s[keep_conf], NMS_IOU)
fin_b, fin_s, fin_c = raw_b[keep_conf][k], raw_s[keep_conf][k], raw_c[keep_conf][k]
t_box = np.array(test.boxes[i][0], float)
side = max(t_box[2] - t_box[0], t_box[3] - t_box[1])
c = ((t_box[0] + t_box[2]) / 2, (t_box[1] + t_box[3]) / 2)
half = max(4 * side, 40)
near = lambda bb: (np.abs((bb[:, 0] + bb[:, 2]) / 2 - c[0]) < half) & (np.abs((bb[:, 1] + bb[:, 3]) / 2 - c[1]) < half)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, (bb, ss, title) in zip(ax, ((raw_b, raw_s, f"raw predictions (conf >= 0.01, without NMS): {near(raw_b).sum()} boxes near the drone"),
                                   (fin_b, fin_s, f"conf >= {CONF}, NMS IoU {NMS_IOU}: {near(fin_b).sum() if len(fin_b) else 0} boxes"))):
    a.imshow(img)
    for b, s in zip(bb, ss):
        a.add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=plt.cm.autumn(1 - s), lw=0.6 + 2 * s))
    a.add_patch(plt.Rectangle(t_box[:2], *(t_box[2:] - t_box[:2]), fill=False, ec="lime", lw=1.5, ls="--"))
    a.set_xlim(max(0, c[0] - half), min(img.shape[1], c[0] + half))
    a.set_ylim(min(img.shape[0], c[1] + half), max(0, c[1] - half))
    a.set_title(title, fontsize=9)
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB08_513_nms.png"), dpi=110)
plt.show()
rows = []
for b, s, cl in zip(fin_b, fin_s, fin_c):
    rows.append({"class of COCO": yolo.names[int(cl)], "confidence": round(float(s), 3),
                 "IoU with the drone": round(float(aitbc.box_iou_matrix([b], [t_box])[0, 0]), 3),
                 "size, px": f"{b[2] - b[0]:.0f} x {b[3] - b[1]:.0f}"})
print(f"All raw boxes: {len(raw_b)}; with conf >= {CONF}: {keep_conf.sum()}; after NMS: {len(fin_b)}")
display(pd.DataFrame(rows).sort_values("confidence", ascending=False).head(10) if rows else pd.DataFrame({"result": ["no boxes"]}))
""", form=True)

md(r"""
## Block 5.1.4. Precision, recall, AP and mAP

**Counting.** The detections of all test images are sorted by confidence. Going down the list, every detection is matched with a free true box of its image with IoU $\geq \tau$ (true positive, TP) or not (false positive, FP); the true boxes left without a detection are false negatives (FN). For the threshold of the confidence $s$
$$P(s) = \frac{TP(s)}{TP(s) + FP(s)}, \qquad R(s) = \frac{TP(s)}{N_{\text{true}}} .$$

**Average precision.** AP is the area under the curve $P(R)$ made non-increasing (the envelope $\tilde P(R) = \max_{R' \geq R} P(R')$); COCO averages $\tilde P$ over 101 values of the recall $0, 0.01, \dots, 1$:
$$\mathrm{AP}_\tau = \frac{1}{101}\sum_{R \in \{0, 0.01, \dots, 1\}} \tilde P(R), \qquad \mathrm{AP}_{50\text{-}95} = \frac{1}{10}\sum_{\tau=0.50}^{0.95}\mathrm{AP}_\tau .$$
**mAP** is the mean of AP over the classes; with one class (drone) mAP = AP. The function `ap_per_class` of Ultralytics integrates the same curve by trapezoids and gives a slightly different number; both are shown.

**Parameters.** `N_EVAL` — random test images; `MODEL`; `IMG_SIZE` — size of the input; `CLASSES` — which detections are taken as a drone (the three nearest classes of COCO or all 80 classes); `CONF_WORK` — working threshold for the table of precision and recall.

**Expected result.** The table AP50, AP50-95, precision and recall at the working threshold, the threshold of the maximum F1; the precision-recall curve; the precision and the recall against the threshold.

**What to observe.** On 150 test images (152 drones) YOLO11n at the input 640 reaches AP50 = 0.517 and AP50-95 = 0.325 (the function of Ultralytics gives 0.516 and 0.324). At the threshold 0.25 the precision is high (0.88) but the recall is only 0.37: almost every reported drone is real, but two of three drones are missed. The maximum F1 = 0.59 is reached at the much lower threshold 0.13. For an air-defence post a threshold below 0.25 is chosen, accepting more false alarms.

**Control question 4.** Why is AP independent of the working threshold, and how is the working threshold chosen for an air-defence post where a missed drone is more dangerous than a false alarm?
""")

code(r"""
#@title 5.1.4 Evaluation of the YOLO model trained on COCO on the drones of DUT Anti-UAV { display-mode: "form" }
N_EVAL = 150 #@param {type:"slider", min:50, max:500, step:50}
MODEL = "yolo11n.pt" #@param ["yolo11n.pt", "yolo11s.pt"]
IMG_SIZE = 640 #@param [640, 1280] {type:"raw"}
CLASSES = "airplane, bird, kite" #@param ["airplane, bird, kite", "all 80 classes"]
CONF_WORK = 0.25 #@param {type:"slider", min:0.05, max:0.8, step:0.05}

from ultralytics.utils.metrics import ap_per_class
yolo = load_yolo(MODEL)
cls_sel = COCO_UAV if CLASSES.startswith("airplane") else None
rng = np.random.default_rng(1)
EVAL_IDX = np.sort(rng.choice(np.where(test.n > 0)[0], N_EVAL, replace=False))
EVAL_PATHS = dut.image_paths("test", test.image_id.values[EVAL_IDX])
EVAL_GT = [np.array(test.boxes[i], float) for i in EVAL_IDX]


def run_detector(model, imgsz, classes, conf=0.01):
    out, t = [], 0.0
    for p in EVAL_PATHS:
        img = cv2.imread(p)[..., ::-1]
        t1 = time.time()
        b, s, c, _ = aitbc.yolo_predict(model, img, imgsz=imgsz, conf=conf, classes=classes)
        t += time.time() - t1
        out.append((b, s))
    return out, 1000 * t / len(EVAL_PATHS)


PRED_514, MS_514 = run_detector(yolo, IMG_SIZE, cls_sel)
THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
ev = aitbc.evaluate_detections(PRED_514, EVAL_GT, iou_thrs=THRS, conf=CONF_WORK)
conf_f1, f1 = aitbc.f1_threshold(ev["curve"])
# the same curve by the function of Ultralytics
tp = np.concatenate([np.stack([aitbc.match_greedy(b, s, g, t)[0] for t in THRS], 1) for (b, s), g in zip(PRED_514, EVAL_GT)])
sc = np.concatenate([s for _, s in PRED_514])
res = ap_per_class(tp.astype(float), sc, np.zeros(len(sc)), np.zeros(sum(len(g) for g in EVAL_GT)))
ap_ul = res[5][0]
display(pd.DataFrame({"value": [len(EVAL_PATHS), ev["n_gt"], f"{ev['AP50']:.3f}", f"{ev['AP']:.3f}",
                                f"{ap_ul[0]:.3f} / {ap_ul.mean():.3f}", f"{ev['precision']:.3f}", f"{ev['recall']:.3f}",
                                f"{ev['false_per_image']:.2f}", f"{conf_f1:.2f} (F1 = {f1:.3f})", f"{MS_514:.0f} ms"]},
                     index=["test images", "drones (true boxes)", "AP50 (COCO, 101 points)", "AP50-95 (COCO)",
                            "AP50 / AP50-95 by ap_per_class of Ultralytics", f"precision at conf >= {CONF_WORK}",
                            f"recall at conf >= {CONF_WORK}", f"false detections per image at conf >= {CONF_WORK}",
                            "threshold of the maximum F1", f"time per image ({DEVICE}, input {IMG_SIZE})"]))
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4))
cv_ = ev["curve"]
ax[0].plot(cv_["recall"], cv_["precision"], color="#bbbbbb", lw=1, label="P(R)")
ax[0].plot(cv_["recall"], np.maximum.accumulate(cv_["precision"][::-1])[::-1], color="#1565c0", lw=2, label="envelope")
j = np.searchsorted(-cv_["scores"], -CONF_WORK, side="right") - 1
if j >= 0:
    ax[0].plot(cv_["recall"][j], cv_["precision"][j], "o", color="#c62828", label=f"conf = {CONF_WORK}")
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.02)
ax[0].legend(fontsize=8)
ax[0].set_title(f"{MODEL} trained on COCO, input {IMG_SIZE}: AP50 = {ev['AP50']:.3f}", fontsize=9)
ax[1].plot(cv_["scores"], cv_["precision"], label="precision", color="#1565c0")
ax[1].plot(cv_["scores"], cv_["recall"], label="recall", color="#ef6c00")
ax[1].axvline(CONF_WORK, color="#c62828", ls="--", lw=1)
ax[1].set_xlabel("threshold of the confidence"); ax[1].set_xlim(0, 1); ax[1].legend(fontsize=8)
ax[1].set_title("precision and recall against the threshold", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB08_514_precision_recall.png"), dpi=110)
plt.show()
RESULT_51 = {"model": MODEL, "img_size": IMG_SIZE, "classes": CLASSES, "n_images": int(N_EVAL),
             "AP50": round(ev["AP50"], 4), "AP50_95": round(ev["AP"], 4), "precision": round(ev["precision"], 4),
             "recall": round(ev["recall"], 4), "conf_work": CONF_WORK}
""", form=True)

md(r"""
## Block 5.1.5. Small targets: size of the input, slicing of the frame and the cost in time

**Why small targets are lost.** A frame of 1920 × 1080 reduced to the input 640 × 640 is scaled by $r = 1/3$: a drone of 24 px becomes 8 px, i.e. one cell of the finest grid (stride 8). Its features are mixed with the background in the first layers, and the detector misses it.

**Two remedies.**
1. **Larger input** $S$ (1280 instead of 640): the drone keeps more pixels; the computation grows as $S^2$.
2. **Slicing** (tiling, SAHI): the frame is cut into windows $T \times T$ with the overlap $o$; every window is processed at its own resolution, the boxes are shifted back to the frame and merged by NMS. The number of runs of the network is
$$n = \left\lceil\frac{W - oT}{(1-o)T}\right\rceil\cdot\left\lceil\frac{H - oT}{(1-o)T}\right\rceil \ (+1 \text{ for the whole frame}).$$

**Parameters.** `TILE` — size of a window; `OVERLAP` — overlap of the windows; `WITH_FULL_FRAME` — also process the whole frame (for large targets close to the camera). The model, the images and the classes are those of block 5.1.4.

**Expected result.** The table of AP50, recall at the working threshold by size groups and the time per image for three variants (input 640, input 1280, slicing); the bar chart of the recall by sizes; a small drone found only by the slicing.

**What to observe.** The input 1280 raises AP50 from 0.517 to 0.577 and the recall at 0.25 from 0.37 to 0.42 at almost three times the time (40 → 109 ms per image on the processor of the test computer). Slicing with windows of 640 px (8.3 runs of the network per image) gives the highest recall, 0.59, and AP50 = 0.583, but the precision falls from 0.88 to 0.73 (0.23 false detections per image instead of 0.05) and the time grows about 10 times (421 ms per image). The gain is in the small drones: on the bar chart the recall of the 64 drones smaller than 32 px grows from about 0.1 (input 640) to about 0.3 (input 1280) and 0.46 (slicing), while for the drones of 96 px and more it grows only from about 0.66 to 0.77.

**Control question 5.** Which variant would you choose for a post with a computer of 10 TOPS that must process 25 frames per second, and what would you lose?
""")

code(r"""
#@title 5.1.5 Input 640, input 1280 and slicing of the frame { display-mode: "form" }
TILE = 640 #@param [480, 640, 800] {type:"raw"}
OVERLAP = 0.2 #@param {type:"slider", min:0.0, max:0.4, step:0.05}
WITH_FULL_FRAME = True #@param {type:"boolean"}

variants = {}
for name, imgsz in (("input 640", 640), ("input 1280", 1280)):
    if name == f"input {IMG_SIZE}":
        variants[name] = (PRED_514, MS_514, 1)
    else:
        variants[name] = run_detector(yolo, imgsz, cls_sel) + (1,)
out, t, runs = [], 0.0, 0
for p in EVAL_PATHS:
    img = cv2.imread(p)[..., ::-1]
    t1 = time.time()
    b, s, c, n = aitbc.predict_tiled(yolo, img, tile=TILE, overlap=OVERLAP, imgsz=TILE, conf=0.01, classes=cls_sel,
                                     full_frame=WITH_FULL_FRAME)
    t += time.time() - t1
    runs += n
    out.append((b, s))
variants[f"slicing {TILE} px, overlap {OVERLAP}"] = (out, 1000 * t / len(EVAL_PATHS), runs / len(EVAL_PATHS))

rows, by_size = [], {}
for name, (pred, ms, n) in variants.items():
    e = aitbc.evaluate_detections(pred, EVAL_GT, iou_thrs=(0.5,), conf=CONF_WORK)
    by_size[name] = e["recall_by_size"]
    rows.append({"variant": name, "runs of the network per image": round(n, 1), "AP50": round(e["AP50"], 3),
                 f"recall at conf >= {CONF_WORK}": round(e["recall"], 3), "precision": round(e["precision"], 3),
                 "false detections per image": round(e["false_per_image"], 2), f"time per image, ms ({DEVICE})": round(ms)})
TAB_515 = pd.DataFrame(rows).set_index("variant")
display(TAB_515)
groups = list(next(iter(by_size.values())).keys())
fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.3, 1]})
wbar = 0.8 / len(by_size)
for k, (name, g) in enumerate(by_size.items()):
    ax[0].bar(np.arange(len(groups)) + k * wbar, [g[x][0] for x in groups], wbar, label=name)
ax[0].set_xticks(np.arange(len(groups)) + wbar * (len(by_size) - 1) / 2)
ax[0].set_xticklabels([f"{x}\n({next(iter(by_size.values()))[x][1]} drones)" for x in groups])
ax[0].set_ylabel(f"recall at conf >= {CONF_WORK}"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
ax[0].set_title("Recall of the drones by the size of the box (frame of the data set)", fontsize=9)
# a small drone found by the slicing and missed at the input 640
names = list(variants)
shown = False
for (b0, s0), (b2, s2), g, p in zip(variants[names[0]][0], variants[names[2]][0], EVAL_GT, EVAL_PATHS):
    for tb in g:
        side = math.sqrt((tb[2] - tb[0]) * (tb[3] - tb[1]))
        if side >= 32:
            continue
        hit0 = len(b0) and (aitbc.box_iou_matrix([tb], b0[s0 >= CONF_WORK]).max(initial=0) >= 0.5)
        hit2 = len(b2) and (aitbc.box_iou_matrix([tb], b2[s2 >= CONF_WORK]).max(initial=0) >= 0.5)
        if hit2 and not hit0:
            img = cv2.imread(p)[..., ::-1]
            ax[1].imshow(aitbc.crop_view(img, tb, scale=6, min_side=64))
            ax[1].set_title(f"drone {tb[2] - tb[0]:.0f} x {tb[3] - tb[1]:.0f} px in {img.shape[1]}x{img.shape[0]}: "
                            f"missed at input 640, found by slicing", fontsize=9)
            shown = True
            break
    if shown:
        break
ax[1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB08_515_small_targets.png"), dpi=110)
plt.show()
RESULT_51["small_targets"] = TAB_515.reset_index().to_dict("records")
update_project_json("5.1", {"lesson_5_1": RESULT_51})
""", form=True)

md(r"""
### Self-check of Lesson 5.1

1. What does a detector give for an image, and how does a one-stage detector differ from a classifier applied to windows?
2. Write the formula of IoU and compute it for two squares of 12 px shifted by 3 px.
3. What do the threshold of the confidence and the threshold of NMS change in the list of detections?
4. How are precision, recall and AP computed, and why is AP50-95 lower than AP50?
5. Name two ways to detect small targets and their cost in computation.
""")

# =====================================================================================
md(r"""
---
# Lesson 5.2 (Group class). Segmentation of the traversable area, tracking of objects and the video-analysis pipeline

**Learning questions**
1. Semantic segmentation of the terrain and the traversable area of a ground robotic platform (UGV).
2. Segmentation network LR-ASPP MobileNetV3 and its transfer learning.
3. Tracking of objects by detection: Kalman filter, association by IoU, ByteTrack.
4. Measures of tracking: MOTA, IDF1, identity switches.
5. Video-analysis pipeline and the requirements of real time.

**Situation of the lesson.** A ground robotic platform moves along field tracks and across a training area; its front camera must show in every frame where it can drive. A reconnaissance UAV over a crossroads must keep the identity of every vehicle from frame to frame. Both tasks run on an onboard computer in real time. The group class builds the segmentation of the traversable area, compares two trackers and measures the time of every stage of the video pipeline.
""")

md(r"""
## Block 5.2.1. Terrain classes and the traversable area of a ground robotic platform

**Semantic segmentation** assigns a class to every pixel of the frame. The annotation of **GOOSE** has 64 classes of an unstructured environment (asphalt, gravel, soil, low and high grass, bushes, trees, buildings, vehicles, persons, sky, …). For the motion of a UGV they are joined into three groups of **traversability**:
- **traversable** — asphalt, cobble, gravel, soil, low grass, moss, leaves, snow, markings, sidewalks;
- **conditionally traversable** — high grass, crops, debris, tree roots (the speed must be reduced);
- **obstacle** — everything else (trees, bushes, walls, poles, vehicles, persons, water); the sky and the own vehicle are not evaluated.

**Data and parts.** The validation part of GOOSE has 962 frames from 8 recording sessions (sequences). Consecutive frames of one session are almost the same; therefore the test frames are taken from **other sessions** than the training frames (the lesson of Topic 4 on leakage): by default the test part is the session on a **military training area** (Garching *Übungsplatz*) and a session **in rain**.

**Parameters.** `N_TRAIN_PER_SEQUENCE` — training frames per training session; `N_TEST_PER_SEQUENCE` — test frames per test session; `TEST_SEQUENCES` — sessions of the test part; `SEED_SEG` — choice of the frames. Frames are reduced to 512 × 256 px.

**Expected result.** The table of the sessions with the shares of the three groups, examples of the frames with the groups of traversability.

**What to observe.** The shares of the groups differ strongly between the sessions: the field tracks of Siegertsbrunn have 53 % traversable and 27 % conditionally traversable terrain, the forest and hills of Aying more than 50 % obstacles. The conditionally traversable group (high grass, crops) is frequent only in two training sessions (22–27 %); in the test sessions on the training area and in rain it is 13 % and 3 %.

**Control question 6.** Why is a frame of the training area not used in the training if the network must work on a training area?
""")

code(r"""
#@title 5.2.1 Terrain classes of GOOSE and the groups of traversability { display-mode: "form" }
N_TRAIN_PER_SEQUENCE = 12 #@param {type:"slider", min:4, max:40, step:4}
N_TEST_PER_SEQUENCE = 15 #@param {type:"slider", min:5, max:40, step:5}
TEST_SEQUENCES = "training area + rain" #@param ["training area + rain", "training area", "hills + rain"]
SEED_SEG = 0 #@param {type:"integer"}

SEG_SIZE = (512, 256)
TRAV_NAMES = ["traversable", "conditionally traversable", "obstacle"]
TRAV_COLORS = np.array([[46, 160, 67], [240, 180, 40], [200, 40, 40]], np.uint8)
TEST_SETS = {"training area + rain": ["2022-09-21_garching_uebungsplatz_2", "2023-05-15_neubiberg_rain"],
             "training area": ["2022-09-21_garching_uebungsplatz_2"],
             "hills + rain": ["2022-12-07_aying_hills", "2023-05-15_neubiberg_rain"]}
test_seqs = TEST_SETS[TEST_SEQUENCES]
by_seq = collections.defaultdict(list)
for m in goose.frames():
    by_seq[m.split("/")[2]].append(m)
rng = np.random.default_rng(SEED_SEG)


def pick(lst, n):
    idx = np.sort(rng.choice(len(lst), min(n, len(lst)), replace=False))
    return [lst[i] for i in idx]


seg_train = [m for s in sorted(by_seq) if s not in test_seqs for m in pick(by_seq[s], N_TRAIN_PER_SEQUENCE)]
seg_test = [m for s in test_seqs for m in pick(by_seq[s], N_TEST_PER_SEQUENCE)]
t0 = time.time()
goose.prefetch(seg_train + seg_test)


def load_seg(member):
    img, lab = goose.load(member, scale=0.25)
    img = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)
    lab = cv2.resize(lab, SEG_SIZE, interpolation=cv2.INTER_NEAREST)
    t = goose.traversability(lab)                       # 0 not evaluated, 1, 2, 3
    return img, np.where(t == 0, 255, t - 1).astype(np.uint8)


SEG_TR = [load_seg(m) for m in seg_train]
SEG_TE = [load_seg(m) for m in seg_test]
print(f"{len(SEG_TR)} training and {len(SEG_TE)} test frames prepared in {time.time() - t0:.0f} s")
rows = []
for s in sorted(by_seq):
    part = "test" if s in test_seqs else "training"
    mem = seg_test if part == "test" else seg_train
    labs = [y for m, (x, y) in zip(mem, SEG_TE if part == "test" else SEG_TR) if m.split("/")[2] == s]
    v = np.concatenate([y[y != 255] for y in labs]) if labs else np.zeros(0)
    rows.append({"session": s, "frames in GOOSE val": len(by_seq[s]), "part": part, "frames used": len(labs),
                 **{n: f"{np.mean(v == k):.0%}" if len(v) else "-" for k, n in enumerate(TRAV_NAMES)}})
display(pd.DataFrame(rows).set_index("session"))


def trav_overlay(img, y, alpha=0.45):
    col = np.zeros_like(img)
    m = y != 255
    col[m] = TRAV_COLORS[y[m]]
    out = img.copy()
    out[m] = (img[m] * (1 - alpha) + col[m] * alpha).astype(np.uint8)
    return out


ex = [SEG_TR[0], SEG_TE[0], SEG_TE[-1]]
ex_names = [seg_train[0].split("/")[2], seg_test[0].split("/")[2], seg_test[-1].split("/")[2]]
fig, axes = plt.subplots(2, 3, figsize=(15, 5))
for k, ((x, y), n) in enumerate(zip(ex, ex_names)):
    axes[0, k].imshow(x); axes[0, k].set_title(n, fontsize=9)
    axes[1, k].imshow(trav_overlay(x, y))
for a in axes.ravel():
    a.axis("off")
aitbc.legend(axes[1, 2], TRAV_NAMES, TRAV_COLORS, loc="lower right", fontsize=8)   # colours 0...255
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB08_521_traversability.png"), dpi=110)
plt.show()
""", form=True)

md(r"""
## Block 5.2.2. Segmentation network LR-ASPP MobileNetV3: transfer learning and quality

**Network.** LR-ASPP (*Lite Reduced Atrous Spatial Pyramid Pooling*, Howard et al., 2019) is the segmentation head of MobileNetV3: the features of the stride 16 (960 channels) are weighted by a global attention branch, joined with the features of the stride 8 (40 channels) and converted by $1\times1$ convolutions into the scores of the classes; the scores are enlarged to the size of the input. The backbone MobileNetV3-Large and the head were trained by torchvision on COCO with the 21 classes of Pascal VOC.

**Transfer learning** as in Lesson 4.4: the backbone is frozen, its features of the training frames are computed once, the two output convolutions are replaced for 3 groups of traversability, and the head is trained by the cross-entropy over the evaluated pixels (the pixels of the sky and of the own vehicle are ignored).

**Quality.** For every group $c$: $\mathrm{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}$ over the pixels of the test frames; mIoU is their mean.

**Decision of motion.** In front of the platform the **corridor** of the next metres of motion is taken (a trapezoid in the bottom part of the frame). If at least 80 % of its pixels are traversable — *go*; if at least 80 % are traversable or conditionally traversable — *slow*; otherwise — *stop*. The decision by the network is compared with the decision by the annotation.

**Parameters.** `EPOCHS_SEG` — epochs of the training of the head; `LR_SEG` — learning rate; `SHOW_TEST` — number of the test frame shown.

**Expected result.** The curve of the loss, the IoU of the groups on the training and the test part, the time of one frame, the agreement of the decisions of motion, three test frames with the annotation and the prediction.

**What to observe.** Only the head (246 526 parameters) is trained, in about 20 s on the processor. On the test sessions the IoU of the traversable terrain is 0.79 and of the obstacles 0.89, but the conditionally traversable group almost fails (IoU 0.04 against 0.61 in the training part): it is rare in the test sessions and is confused with the neighbouring groups; mIoU falls from 0.82 to 0.57. The decision of motion in the corridor agrees with the annotation in 87 % of the 30 test frames (by the annotation 26 go, 1 slow, 3 stop). The whole network takes about 25 ms per frame of 512 × 256 on the processor.

**Control question 7.** Which error of the segmentation is dangerous for a UGV, and which is only costly in time?
""")

code(r"""
#@title 5.2.2 LR-ASPP MobileNetV3: training of the head and the IoU of the groups { display-mode: "form" }
EPOCHS_SEG = 40 #@param {type:"slider", min:5, max:100, step:5}
LR_SEG = 0.003 #@param {type:"number"}
SHOW_TEST = 0 #@param {type:"integer"}

import torch.nn.functional as F
torch.manual_seed(0)
seg_model = aitbc.lraspp_traversability(3).to(DEVICE).eval()
MEAN = torch.tensor(aitbc.IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(aitbc.IMAGENET_STD).view(1, 3, 1, 1)


def to_batch(imgs):
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
    return ((x - MEAN) / STD).to(DEVICE)


@torch.no_grad()
def backbone_features(data, batch=8):
    low, high = [], []
    for k in range(0, len(data), batch):
        f = seg_model.backbone(to_batch([x for x, _ in data[k:k + batch]]))
        low.append(f["low"].half().cpu()); high.append(f["high"].half().cpu())
    return torch.cat(low), torch.cat(high)


t0 = time.time()
F_TR = backbone_features(SEG_TR)
F_TE = backbone_features(SEG_TE)
Y_TR = torch.from_numpy(np.stack([y for _, y in SEG_TR])).long()
Y_TE = torch.from_numpy(np.stack([y for _, y in SEG_TE])).long()
t_feat = time.time() - t0
head = seg_model.classifier
opt = torch.optim.Adam(head.parameters(), lr=LR_SEG)
w = torch.tensor([1.0, 1.5, 1.0], device=DEVICE)       # conditional terrain is the rarest group
hist = []
t0 = time.time()
for ep in range(EPOCHS_SEG):
    head.train()
    perm = torch.randperm(len(Y_TR))
    tot = 0.0
    for k in range(0, len(perm), 8):
        j = perm[k:k + 8]
        logits = head({"low": F_TR[0][j].float().to(DEVICE), "high": F_TR[1][j].float().to(DEVICE)})
        logits = F.interpolate(logits, size=Y_TR.shape[1:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(logits, Y_TR[j].to(DEVICE), weight=w, ignore_index=255)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(j)
    hist.append(tot / len(perm))
t_train = time.time() - t0


@torch.no_grad()
def predict_feats(feats, n):
    head.eval()
    out = []
    for k in range(0, n, 8):
        lg = head({"low": feats[0][k:k + 8].float().to(DEVICE), "high": feats[1][k:k + 8].float().to(DEVICE)})
        lg = F.interpolate(lg, size=(SEG_SIZE[1], SEG_SIZE[0]), mode="bilinear", align_corners=False)
        out.append(lg.argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out)


P_TR, P_TE = predict_feats(F_TR, len(Y_TR)), predict_feats(F_TE, len(Y_TE))
cm_tr, iou_tr = aitbc.segmentation_iou(P_TR, Y_TR.numpy(), 3)
cm_te, iou_te = aitbc.segmentation_iou(P_TE, Y_TE.numpy(), 3)
seg_model.eval()
x1 = to_batch([SEG_TE[0][0]])
with torch.no_grad():
    ms_frame = aitbc.time_call(lambda: seg_model(x1)["out"], repeats=5)
corr = aitbc.corridor_mask((SEG_SIZE[1], SEG_SIZE[0]))


def decision(y):
    v = y[corr & (y != 255)]
    if v.size == 0:
        return "not evaluated"
    if np.mean(v == 0) >= 0.8:
        return "go"
    return "slow" if np.mean(v <= 1) >= 0.8 else "stop"


dec_true = [decision(y) for _, y in SEG_TE]
dec_pred = [decision(np.where(y == 255, 255, p)) for (_, y), p in zip(SEG_TE, P_TE)]
agree = np.mean([a == b for a, b in zip(dec_true, dec_pred)])
display(pd.DataFrame({"training part": [f"{v:.3f}" for v in iou_tr] + [f"{iou_tr.mean():.3f}"],
                      "test part (other sessions)": [f"{v:.3f}" for v in iou_te] + [f"{iou_te.mean():.3f}"]},
                     index=[f"IoU: {n}" for n in TRAV_NAMES] + ["mIoU"]))
display(pd.DataFrame({"value": [len(SEG_TR), len(SEG_TE), sum(p.numel() for p in head.parameters()), f"{t_feat:.0f} s",
                                f"{t_train:.0f} s ({EPOCHS_SEG} epochs)", f"{ms_frame:.0f} ms ({DEVICE})",
                                f"{agree:.0%} of {len(dec_true)} test frames",
                                ", ".join(f"{d}: {dec_true.count(d)}" for d in ("go", "slow", "stop"))]},
                     index=["training frames", "test frames", "trained parameters (head)", "features of the backbone",
                            "training of the head", "one frame 512 x 256, whole network",
                            "decision of motion: agreement with the annotation", "decisions by the annotation"]))
fig, axes = plt.subplots(3, 3, figsize=(15, 7.2))
for r_, k in enumerate([SHOW_TEST % len(SEG_TE), (SHOW_TEST + len(SEG_TE) // 2) % len(SEG_TE), len(SEG_TE) - 1]):
    x, y = SEG_TE[k]
    p = np.where(y == 255, 255, P_TE[k])
    axes[r_, 0].imshow(x); axes[r_, 0].set_title(seg_test[k].split("/")[2], fontsize=8)
    axes[r_, 1].imshow(trav_overlay(x, y)); axes[r_, 1].set_title(f"annotation: {dec_true[k]}", fontsize=8)
    im = trav_overlay(x, p)
    cnt, _ = cv2.findContours(corr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, cnt, -1, (255, 255, 255), 2)
    axes[r_, 2].imshow(im); axes[r_, 2].set_title(f"network: {dec_pred[k]} (corridor in white)", fontsize=8)
for a in axes.ravel():
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB08_522_segmentation.png"), dpi=110)
plt.show()
fig, ax = plt.subplots(figsize=(5, 2.6))
ax.plot(np.arange(1, len(hist) + 1), hist, color="#1565c0")
ax.set_xlabel("epoch"); ax.set_ylabel("loss, training part"); ax.set_title("Training of the head", fontsize=9)
plt.tight_layout()
plt.show()
torch.save(head.state_dict(), os.path.join(MODEL_DIR, "lraspp_traversability_head.pt"))
RESULT_52 = {"segmentation": {"train_frames": len(SEG_TR), "test_frames": len(SEG_TE), "test_sessions": test_seqs,
                              "iou_test": dict(zip(TRAV_NAMES, map(float, np.round(iou_te, 4)))),
                              "miou_test": round(float(iou_te.mean()), 4), "decision_agreement": round(float(agree), 3),
                              "ms_per_frame": round(float(ms_frame), 1), "device": DEVICE}}
""", form=True)

md(r"""
## Block 5.2.3. Tracking of objects by detection: Kalman filter, association, ByteTrack

**Tracking by detection.** The detector is run in every frame; the tracker joins the detections of consecutive frames into tracks with a constant identity (ID):
1. **Prediction.** For every track a Kalman filter with the state $(c_x, c_y, w, h, \dot c_x, \dot c_y, \dot w, \dot h)$ predicts its box in the new frame (constant velocity, Notebook 05 of the practical course).
2. **Association.** The matrix $\mathrm{IoU}$ of the predicted boxes and the detections is formed, and the Hungarian method finds the pairs with the largest total IoU; a pair is accepted if its IoU is not less than a threshold.
3. **Update.** The matched track corrects its filter by the detection; an unmatched detection with a high score opens a new track; a track without a detection for more than `MAX_LOST` frames is deleted.

**ByteTrack** (Zhang Y. et al., ECCV 2022) adds a second association: the tracks left without a detection are matched with the **detections of low score** ($\text{LOW} \leq s < \text{HIGH}$), which an ordinary tracker throws away. A vehicle partly hidden by a tree or blurred by motion gets a low score, but its track continues.

**Measures** (CLEAR-MOT and identity measures, the annotation of M3OT):
$$\mathrm{MOTA} = 1 - \frac{FN + FP + \mathrm{IDSW}}{GT}, \qquad \mathrm{IDF1} = \frac{2\,\mathrm{IDTP}}{GT + N_{\text{hyp}}},$$
where IDSW is the number of changes of the ID of a true object, IDTP the number of detections assigned to the correct identity by the best one-to-one assignment of the tracks to the true objects.

**Parameters.** `SEQUENCE` — sequence of M3OT (day crossroads); `N_FRAMES` — frames (10 frames per second); `DETECTOR`, `IMG_SIZE` — model and size of the input (the vehicles are 13–20 px in the frames 640 × 512, therefore the frame is enlarged); `HIGH`, `LOW` — thresholds of the score; `MAX_LOST` — frames without a detection before a track is deleted. The detections are computed once for a sequence, detector and input; the trackers are then compared quickly.

**Expected result.** The table of MOTA, IDF1, IDSW, FP and FN of the two trackers; the trajectories of the tracks on the last frame; the number of IDs against the number of true vehicles.

**What to observe.** YOLO11s at the input 1920 finds the vehicles of 13–20 px with high precision (0.94) but with low scores: at the threshold 0.3 the recall is only 0.36 (AP50 = 0.68). With the same detections the IoU tracker reaches MOTA = 0.32 and IDF1 = 0.49; ByteTrack, which also uses the detections between 0.1 and 0.3, reaches MOTA = 0.44 and IDF1 = 0.62: the false negatives fall from 598 to 483 of 906 true boxes, the false positives grow only from 17 to 24, and there is one identity switch. Both trackers are far below 1 because the detector misses most vehicles; the tracker cannot create a track for a vehicle that is never detected. The time of the tracker is below 1 ms per frame.

**Control question 8.** Why does ByteTrack reduce the false negatives without raising the false positives, although it uses detections of low score?
""")

code(r"""
#@title 5.2.3 Tracking of vehicles in a UAV video: IoU tracker and ByteTrack { display-mode: "form" }
SEQUENCE = "1-05" #@param ["1-05", "1-07", "1-08"]
N_FRAMES = 60 #@param {type:"slider", min:20, max:200, step:10}
DETECTOR = "yolo11s.pt" #@param ["yolo11n.pt", "yolo11s.pt"]
IMG_SIZE = 1920 #@param [640, 1280, 1920] {type:"raw"}
HIGH = 0.3 #@param {type:"slider", min:0.2, max:0.8, step:0.05}
LOW = 0.1 #@param {type:"slider", min:0.05, max:0.4, step:0.05}
MAX_LOST = 30 #@param {type:"slider", min:1, max:60, step:1}

VEHICLES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
det_model = load_yolo(DETECTOR)
FR_T = m3.frames(SEQUENCE, start=0, count=N_FRAMES)
gt_tab = m3.tracks(SEQUENCE)
key = (SEQUENCE, N_FRAMES, DETECTOR, IMG_SIZE)
if globals().get("DET_KEY") != key:
    DETS, t0 = [], time.time()
    for img in FR_T:
        b, s, c, sp = aitbc.yolo_predict(det_model, img, imgsz=IMG_SIZE, conf=0.05, classes=VEHICLES)
        DETS.append((b, s))
    DET_MS = 1000 * (time.time() - t0) / N_FRAMES
    DET_KEY = key
GT_T = {f + 1: [(int(r.id), [r.x, r.y, r.x + r.w, r.y + r.h]) for r in gt_tab[gt_tab.frame == f + 1].itertuples()]
        for f in range(N_FRAMES)}
TRACKS = {}
rows = []
aitbc.ByteTracker().update(*DETS[0])   # warm-up: the first call loads the Hungarian method of SciPy
for name, use_low in (("IoU tracker (score >= HIGH only)", False), ("ByteTrack (+ second association LOW...HIGH)", True)):
    trk = aitbc.ByteTracker(high=HIGH, low=LOW, new_track=min(HIGH + 0.1, 0.9), max_lost=MAX_LOST, use_low=use_low)
    t0 = time.time()
    hyp = {f + 1: [(i, b) for i, b, s in trk.update(*DETS[f])] for f in range(N_FRAMES)}
    ms = 1000 * (time.time() - t0) / N_FRAMES
    m = aitbc.mot_metrics(GT_T, hyp, 0.5)
    TRACKS[name] = hyp
    rows.append({"tracker": name, "MOTA": round(m["MOTA"], 3), "IDF1": round(m["IDF1"], 3), "ID switches": m["IDSW"],
                 "FP": m["FP"], "FN": m["FN"], "true boxes": m["GT"], "true vehicles": m["gt_ids"],
                 "track IDs": m["hyp_ids"], "time of the tracker, ms/frame": round(ms, 2)})
TAB_523 = pd.DataFrame(rows).set_index("tracker")
ev = aitbc.evaluate_detections(DETS, [[b for _, b in GT_T[f + 1]] for f in range(N_FRAMES)], conf=HIGH)
print(f"Detector {DETECTOR}, input {IMG_SIZE}: {DET_MS:.0f} ms per frame ({DEVICE}); AP50 = {ev['AP50']:.3f}, "
      f"recall at score >= {HIGH}: {ev['recall']:.3f}, precision {ev['precision']:.3f}")
display(TAB_523)
fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.3, 1]})
last = FR_T[-1].copy()
hyp = TRACKS[list(TRACKS)[1]]
paths_ = collections.defaultdict(list)
for f in range(1, N_FRAMES + 1):
    for i, b in hyp[f]:
        paths_[i].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
ax[0].imshow(last)
cmap = plt.cm.tab20
for i, pts in paths_.items():
    pts = np.array(pts)
    ax[0].plot(pts[:, 0], pts[:, 1], "-", color=cmap(i % 20), lw=1.5)
for i, b in hyp[N_FRAMES]:
    ax[0].add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=cmap(i % 20), lw=1.5))
    ax[0].text(b[0], b[1] - 2, str(i), color=cmap(i % 20), fontsize=7)
ax[0].set_title(f"M3OT {SEQUENCE}: ByteTrack, trajectories of {len(paths_)} tracks over {N_FRAMES} frames", fontsize=9)
ax[0].axis("off")
for name, hyp_ in TRACKS.items():
    ax[1].plot(np.arange(1, N_FRAMES + 1), [len(hyp_[f]) for f in range(1, N_FRAMES + 1)], label=name)
ax[1].plot(np.arange(1, N_FRAMES + 1), [len(GT_T[f]) for f in range(1, N_FRAMES + 1)], "k--", label="true vehicles")
ax[1].set_xlabel("frame"); ax[1].set_ylabel("tracked objects in the frame"); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB08_523_tracking.png"), dpi=110)
plt.show()
RESULT_52["tracking"] = {"sequence": SEQUENCE, "frames": N_FRAMES, "detector": DETECTOR, "img_size": IMG_SIZE,
                         "high": HIGH, "low": LOW, "table": TAB_523.reset_index().to_dict("records")}
""", form=True)

md(r"""
## Block 5.2.4. Video-analysis pipeline and the requirements of real time

**Pipeline.** For every frame: decoding of the video → preparation of the input (letterbox, normalisation) → inference of the network → post-processing (decoding of the boxes, NMS) → tracking → drawing → recording or transmission. The pipeline works in **real time** if the time per frame does not exceed the period of the frames:
$$t_{\text{frame}} = \sum_k t_k \leq \frac{1}{f_{\text{video}}}, \qquad f_{\max} = \frac{1}{t_{\text{frame}}}.$$
The longest stage is the inference; it grows with the size of the input and of the model.

**Detection on every $k$-th frame.** When the budget is short, the detector is run only on every $k$-th frame; between the detections the boxes are predicted by the Kalman filters of the tracks. The time of detection per frame falls $k$ times; the error grows with the motion of the objects between the detections.

**Parameters.** `TARGET_FPS` — required frame rate; `DETECT_EVERY` — $k$ for the saved video; `SAVE_VIDEO` — write the video with the tracks into `outputs/video`. The sequence, the detector and the detections are those of block 5.2.3.

**Expected result.** The table of the time of the stages and of the maximum frame rate for the models YOLO11n and YOLO11s at three sizes of the input; the table MOTA and IDF1 against $k$; the saved video.

**What to observe.** On the processor of the test computer the inference is the dominant stage: YOLO11n at 640 needs about 50 ms per frame (about 20 frames per second), YOLO11s at 1920 about 0.87 s (about 1 frame per second); decoding, preparation and NMS take a few milliseconds. When the detector runs only on every 2nd, 3rd or 5th frame, MOTA (0.44) and IDF1 (0.62) do not fall in this sequence: the vehicles at the crossroads move by a few pixels per frame and the Kalman filter predicts them well. With fast targets (an FPV drone, a vehicle close to the camera) the same $k$ would lose the tracks.

**Control question 9.** The onboard computer gives 12 frames per second for the detector at the input 1280. Which settings of the pipeline give 25 frames per second, and what is lost?
""")

code(r"""
#@title 5.2.4 Time of the stages, detection on every k-th frame and the saved video { display-mode: "form" }
TARGET_FPS = 10 #@param [10, 25, 30] {type:"raw"}
DETECT_EVERY = 2 #@param [1, 2, 3, 5] {type:"raw"}
SAVE_VIDEO = True #@param {type:"boolean"}

img = FR_T[0]
jpg = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
t_dec = aitbc.time_call(lambda: cv2.imdecode(jpg, cv2.IMREAD_COLOR), repeats=10)
rows = []
for name in ("yolo11n.pt", "yolo11s.pt"):
    mdl = load_yolo(name)
    for imgsz in (640, 1280, 1920):
        sp = []
        for k in range(4):
            sp.append(aitbc.yolo_predict(mdl, FR_T[k], imgsz=imgsz, conf=0.05, classes=VEHICLES)[3])
        sp = pd.DataFrame(sp[1:]).median()
        tot = t_dec + sp["preprocess"] + sp["inference"] + sp["postprocess"]
        rows.append({"model": name.replace(".pt", ""), "input": imgsz, "decoding, ms": round(t_dec, 1),
                     "preparation, ms": round(sp["preprocess"], 1), "inference, ms": round(sp["inference"], 1),
                     "post-processing (NMS), ms": round(sp["postprocess"], 1), "total, ms": round(tot, 1),
                     "max frame rate": round(1000 / tot, 1), f"real time at {TARGET_FPS} fps": "yes" if tot <= 1000 / TARGET_FPS else "no"})
TAB_524 = pd.DataFrame(rows).set_index(["model", "input"])
display(TAB_524)

rows = []
for k in (1, 2, 3, 5):
    trk = aitbc.ByteTracker(high=HIGH, low=LOW, new_track=min(HIGH + 0.1, 0.9), max_lost=MAX_LOST)
    hyp, t_trk = {}, 0.0
    for f in range(N_FRAMES):
        t0 = time.time()
        out = trk.update(*DETS[f]) if f % k == 0 else trk.coast()
        t_trk += time.time() - t0
        hyp[f + 1] = [(i, b) for i, b, s in out]
    m = aitbc.mot_metrics(GT_T, hyp, 0.5)
    rows.append({"detection on every k-th frame": k, "MOTA": round(m["MOTA"], 3), "IDF1": round(m["IDF1"], 3),
                 "ID switches": m["IDSW"], "detector, ms per frame (average)": round(DET_MS / k, 1),
                 "tracker, ms per frame": round(1000 * t_trk / N_FRAMES, 2)})
    if k == DETECT_EVERY:
        HYP_K = hyp
TAB_524K = pd.DataFrame(rows).set_index("detection on every k-th frame")
display(TAB_524K)

if SAVE_VIDEO:
    t0 = time.time()
    out_frames = []
    for f in range(N_FRAMES):
        objs = [{"box": b, "cls": "vehicle"} for i, b in HYP_K[f + 1]]
        ids = [i for i, b in HYP_K[f + 1]]
        im = aitbc.draw_boxes(FR_T[f], objs, labels=False, ids=ids, thickness=1, color=(255, 230, 0))
        cv2.putText(im, f"frame {f + 1}  detection every {DETECT_EVERY}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out_frames.append(im)
    t_draw = 1000 * (time.time() - t0) / N_FRAMES
    vpath = aitbc.write_video(os.path.join(VIDEO_DIR, f"NB08_tracking_{SEQUENCE}_k{DETECT_EVERY}.mp4"), out_frames, fps=10)
    info = aitbc.video_info(vpath)
    print(f"Video: {vpath} ({os.path.getsize(vpath) / 1e6:.1f} MB, {info}); drawing and writing {t_draw:.1f} ms per frame")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, f in zip(ax, (0, N_FRAMES - 1)):
        a.imshow(out_frames[f]); a.set_title(f"frame {f + 1} of the saved video", fontsize=9); a.axis("off")
    plt.tight_layout()
    plt.show()
RESULT_52["real_time"] = {"stages": TAB_524.reset_index().to_dict("records"),
                          "detect_every": TAB_524K.reset_index().to_dict("records")}
update_project_json("5.2", {"lesson_5_2": RESULT_52})
""", form=True)

md(r"""
### Self-check of Lesson 5.2

1. How are the 64 classes of the terrain joined into the groups of traversability, and why is the sky not evaluated?
2. What is trained in the transfer learning of LR-ASPP, and why are the test frames taken from other sessions?
3. Describe the steps of a tracker by detection and the second association of ByteTrack.
4. What do MOTA and IDF1 measure, and which of them reacts to the identity switches more strongly?
5. How is the maximum frame rate of a pipeline found, and what does the detection on every $k$-th frame give?

## Data sources and licences

The course library reads the files directly from the repositories of the data sets; the data are not redistributed with the notebook. When images or results are published, the data sets are cited.

1. **DUT Anti-UAV.** Zhao J., Zhang J., Li D., Wang D. Vision-based Anti-UAV Detection and Tracking. *IEEE Transactions on Intelligent Transportation Systems* (2022); arXiv:2205.10851 · Data: https://github.com/wangdongdut/DUT-Anti-UAV, copy https://huggingface.co/datasets/aegean-ai/dut-anti-uav · Licence Apache-2.0.
2. **M3OT.** Nie Z. et al. *Scientific Data* 12, 1927 (2025) · Licence CC BY 4.0.
3. **GOOSE.** Mortimer P. et al. The GOOSE Dataset for Perception in Unstructured Environments. *IEEE ICRA 2024*, 14838–14844 · https://goose-dataset.de · Licence CC BY-SA 4.0.
4. **YOLO11.** Jocher G., Qiu J. Ultralytics YOLO11 (2024), https://github.com/ultralytics/ultralytics · Licence AGPL-3.0 (installed in the session, not redistributed). **COCO.** Lin T.-Y. et al. Microsoft COCO: Common Objects in Context. *ECCV 2014*.
5. **LR-ASPP, MobileNetV3.** Howard A. et al. Searching for MobileNetV3. *ICCV 2019*, 1314–1324; weights of torchvision (BSD-3-Clause).
6. **ByteTrack.** Zhang Y. et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV 2022*. **CLEAR-MOT.** Bernardin K., Stiefelhagen R. *EURASIP J. Image and Video Processing* (2008). **IDF1.** Ristani E. et al. *ECCV Workshops 2016*. **Slicing (SAHI).** Akyon F. C. et al. *IEEE ICIP 2022*.

**Next notebook.** Notebook 09 — Topic 5, Lesson 5.3 (practical class): YOLO11 on a single frame, on a series of UAV frames, on FPV frames and on a thermal frame, and the saved video; Lesson 5.4 (practical class, mini-project): training of a counter-UAV detector on DUT Anti-UAV, its evaluation, tracking of a drone and export to ONNX.
""")


# =====================================================================================
def build(path_out, lib_path="aitbc.py"):
    lib = open(lib_path, encoding="utf-8").read()
    assert "\'\'\'" not in lib, "library source must not contain triple single quotes"
    nb = new_notebook()
    nb.metadata = {"colab": {"provenance": [], "toc_visible": True, "name": os.path.basename(path_out)},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"}
    for kind, text, form in CELLS:
        if kind == "md":
            nb.cells.append(new_markdown_cell(text))
        else:
            cell = new_code_cell(text.replace("%%LIB%%", lib))
            if form:
                cell.metadata["cellView"] = "form"
            nb.cells.append(cell)
    nbformat.validate(nb)
    nbformat.write(nb, path_out)
    return nb


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "AITBC_NB08_YOLO_detection_segmentation_and_tracking.ipynb"
    lib = sys.argv[2] if len(sys.argv) > 2 else "aitbc.py"
    nb = build(out, lib)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    print(f"written {out}: {len(nb.cells)} cells ({n_md} text, {len(nb.cells) - n_md} code)")
