# Builder of Notebook 09 of the course "AI Technologies (Basic Course)" (AIT-BC).
# Topic 5, Lessons 5.3 (practical class) and 5.4 (practical class, mini-project). Real open data sets only.
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
# AI Technologies (Basic Course) · Notebook 09
## Topic 5. Neural-network detection, segmentation and tracking of objects in the video of UAVs and UGVs

| Lesson | Type, duration | Title |
|---|---|---|
| **5.3** | Practical class, 2 h | Detection of objects by YOLO11 in a single frame, in a series of UAV frames, in FPV frames and in a thermal frame; the saved video |
| **5.4** | Practical class, 4 h | Mini-project: a detector and tracker of drones for a counter-UAV post (training of YOLO11, evaluation, tracking, export to ONNX) |

Course: *AI Technologies (Basic Course)*, AIT-BC · Content module 2 · Kruty Heroes Military Institute of Telecommunications and Information Technology, Kyiv, 2026.

**How to work with this notebook**

1. Run the cells from top to bottom: *Runtime → Run all*, or `Shift+Enter` cell by cell. Section 0 is always run first; after a restart of the runtime it is run again.
2. Every task begins with a text cell: the method, the parameters, the expected result and the control question. The answers are written in the text cell **Report** at the end of each lesson.
3. The cells marked **(code shown)** are the practical skills of the lessons: read the code, change the parameters and run it again. The tasks of a lesson use the results of the previous tasks of the same lesson.
4. Lesson 5.3 runs on the processor in 1–3 minutes per task. Lesson 5.4 trains a neural network: **switch on a GPU** (*Runtime → Change runtime type → T4 GPU*). On the processor of the test computer (16 logical cores) one epoch with the default settings took about 110 s, i.e. about 37 min for 20 epochs; the processor of Colab is slower, a GPU is many times faster.
5. The results are saved in the course project on Google Drive (`outputs`, `models`, `project.json`), created in Notebook 02.

**Data.** Only real images and videos with the annotation of the authors of the data sets: **M3OT** (video of DJI Mavic 3T drones over roads with tracks of vehicles), **TII-RATM** (camera of an FPV racing quadrotor), **HIT-UAV** (thermal frames of a reconnaissance UAV with boxes of persons and vehicles) and **DUT Anti-UAV** (images and videos of drones taken by ground cameras).

**Contents**

- **0. Preparation** — 0.1 runtime, course library, YOLO and data sets · 0.2 course project
- **Lesson 5.3 (practical class)**
  - 5.3.1 Model YOLO11 and the parameters of inference
  - 5.3.2 Detection on a single frame of a UAV
  - 5.3.3 Detection on a series of UAV frames
  - 5.3.4 Detection on FPV frames
  - 5.3.5 Detection on a thermal frame
  - 5.3.6 Saving of the annotated video and of the table of detections
- **Lesson 5.4 (practical class, mini-project)**
  - 5.4.1 Training data set in the YOLO format
  - 5.4.2 Baseline: the model trained on COCO
  - 5.4.3 Training of YOLO11 for the class UAV
  - 5.4.4 Evaluation on the test part and the working threshold
  - 5.4.5 Small drones: slicing of the frame
  - 5.4.6 Tracking of a drone in a video
  - 5.4.7 Export to ONNX and the check of real time
  - 5.4.8 Report of the project
- **Summary**: self-check of Topic 5, data sources and licences
""")

# =====================================================================================
md(r"""
---
# 0. Preparation

### Block 0.1. Runtime, course library, YOLO and data sets

**Purpose.** To prepare the runtime: the libraries, the library **Ultralytics** (it is installed in the session if it is absent), the course library `aitbc` (version 2.7.0, as in Notebook 08) and the data sets of the lessons.

**Parameters.** `CACHE_ON_DRIVE` — keep the downloaded files on Google Drive; otherwise they are kept on the disk of the session.

**Output.** A table of versions, the computing device and the size of the cache.

*Licence note.* Ultralytics and the weights of YOLO11 are distributed under AGPL-3.0; they are installed from PyPI in the session and are not part of the course materials. The weights trained in Lesson 5.4 are derived from YOLO11 and fall under the same licence.
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
import logging
from ultralytics.utils import LOGGER as YOLO_LOGGER
YOLO_LOGGER.setLevel(logging.WARNING)   # Ultralytics prints only warnings; the cells print their own results

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
    return YOLO(name if os.path.isabs(name) else os.path.join(WEIGHTS_DIR, name))


t0 = time.time()
m3 = aitbc.M3OT()              # video of UAVs over roads with tracks of vehicles
hit = aitbc.HITUAV()           # thermal frames of a reconnaissance UAV
tii = aitbc.TIIRacing()        # camera of an FPV racing quadrotor
dut = aitbc.DUTAntiUAV()       # drones seen from the ground (counter-UAV task)
VERSIONS = {"Python": platform.python_version(), "NumPy": np.__version__, "OpenCV": cv2.__version__,
            "pandas": pd.__version__, "PyTorch": torch.__version__, "torchvision": torchvision.__version__,
            "Ultralytics": ultralytics.__version__, "aitbc": aitbc.__version__}
n_files, size_mb = aitbc.cache_size()
dev = torch.cuda.get_device_name(0) if DEVICE == "cuda" else f"CPU, {os.cpu_count()} logical cores"
display(pd.DataFrame({"Value": list(VERSIONS.values()) + ["Google Colab" if IN_COLAB else "local Jupyter", dev,
                                                           f"{aitbc.DATA_ROOT}: {n_files} files, {size_mb:.0f} MB",
                                                           f"{time.time() - t0:.0f} s"]},
                     index=list(VERSIONS) + ["environment", "computing device", "cache of the data sets", "preparation time"]))
if DEVICE == "cpu":
    print("No GPU: Lesson 5.3 runs normally; for Lesson 5.4 switch on Runtime -> Change runtime type -> T4 GPU.")
""", form=True)

md(r"""
### Block 0.2. Course project

**Purpose.** To connect the course project created in Notebook 02. The results are saved in it: figures in `outputs/figures`, tables and the report of the mini-project in `outputs/reports`, videos in `outputs/video`, the trained detector (PyTorch and ONNX) in `models`, the parameters in `project.json`.

**Parameters.** `USE_DRIVE` — the project is on Google Drive; otherwise on the disk of the session. `PROJECT_PATH` — folder of the project, the same as in Notebooks 02–08.
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
# Lesson 5.3 (Practical class). Detection of objects by YOLO11 in frames and video of UAVs

**Learning questions**
1. Loading of a YOLO11 model and the parameters of inference.
2. Detection on a single frame of a UAV and its check by the annotation.
3. Detection on a series of frames and the stability of the result.
4. Detection on FPV frames and on thermal frames: the limits of a model trained on everyday photographs.
5. Saving of the annotated video and of the table of detections.

**Task of the lesson.** The operator of a reconnaissance UAV receives a video of a crossroads from 100–120 m. A ready detector YOLO11 (weights of COCO) must mark the vehicles in a frame and in a series of frames; the same detector is tried on the frames of an FPV drone and of a thermal camera. Every result is checked by the annotation of the data sets: how many objects are found, how many detections are false, how much time a frame takes.

**Result of the lesson.** The tables of the detection quality on visible, FPV and thermal frames, the annotated video `outputs/video/NB09_detections_<sequence>.mp4` and the table of detections `outputs/reports/NB09_detections_<sequence>.csv`, the parameters in `project.json`.
""")

md(r"""
## Task 5.3.1. Model YOLO11 and the parameters of inference

**Method.** A model is created by the name of its weights file, `YOLO("yolo11n.pt")`; the file (5.4 MB for YOLO11n, 18.4 MB for YOLO11s) is downloaded once. The call `model.predict(image, imgsz=..., conf=..., iou=..., classes=...)` returns for every image the boxes, the scores and the classes after NMS. Its main parameters:

| Parameter | Meaning |
|---|---|
| `imgsz` | size $S$ of the input (the frame is letterboxed to $S \times S$, multiple of 32) |
| `conf` | threshold of the confidence |
| `iou` | threshold of NMS |
| `classes` | list of the classes kept (indices of COCO) |
| `max_det` | maximum number of boxes in an image |

**Parameters.** `MODEL` — YOLO11n or YOLO11s.

**Expected result.** The table of the model (layers, parameters, operations GFLOPs at 640) and the classes of COCO used in the lesson with their indices.

**Control question 1.** Which classes of COCO are relevant for a reconnaissance UAV, and which objects of a battlefield have no class in COCO?
""")

code(r"""
#@title 5.3.1 Model YOLO11 and its classes (code shown)
MODEL = "yolo11s.pt" #@param ["yolo11n.pt", "yolo11s.pt"]

from ultralytics.utils.torch_utils import get_num_params, get_flops
yolo = load_yolo(MODEL)
layers = sum(1 for m in yolo.model.modules() if len(m._modules) == 0)   # layers with parameters or operations
params = get_num_params(yolo.model)
gflops = get_flops(yolo.model, 640)
CLASS_IDS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5, "truck": 7,
             "bird": 14, "kite": 33, "tv": 62}
VEHICLES = [2, 3, 5, 7]
display(pd.DataFrame({"value": [MODEL, layers, f"{params / 1e6:.2f} M", f"{gflops:.1f}", len(yolo.names),
                                os.path.getsize(yolo.ckpt_path) / 1e6 if getattr(yolo, "ckpt_path", None) else "-"]},
                     index=["weights", "layers", "parameters", "GFLOPs (input 640)", "classes of COCO", "file, MB"]))
display(pd.DataFrame({"index in COCO": list(CLASS_IDS.values())}, index=list(CLASS_IDS)).T)
""")

md(r"""
## Task 5.3.2. Detection on a single frame of a UAV

**Data.** A frame of the sequence of **M3OT** (DJI Mavic 3T, 100–120 m, 640 × 512 px, day crossroads); the annotation gives the boxes of all vehicles. The vehicles occupy 10–25 pixels.

**Method.** The detector is run with the classes *car, motorcycle, bus, truck* for three sizes of the input; a detection is correct if its IoU with a free true box is $\geq 0.5$ (block 5.1.4 of Notebook 08). Precision $P = TP/(TP+FP)$, recall $R = TP/N_{\text{true}}$.

**Parameters.** `SEQUENCE`, `FRAME` — sequence and number of the frame; `IMG_SIZE` — input of the shown result; `CONF`, `IOU` — thresholds of the confidence and of NMS.

**Expected result.** The frame with the true boxes (green) and the detections (red), the table of the number of detections, the precision, the recall and the time for the inputs 640, 1280 and 1920.

**What to observe.** In frame 100 of the crossroads 1-05 there are 15 annotated vehicles of 13–20 px. YOLO11s with the weights of COCO reports only 2 of them at the input 640, none at 1280 and 3 at 1920; every reported vehicle is correct (precision 1.0), but the recall does not exceed 0.2. The model was trained on photographs taken from the ground, where a car occupies hundreds of pixels; a car seen from above at 100–120 m is outside its training data. One frame is a small sample, so the recall changes irregularly with the input; the series of task 5.3.3 gives a more stable estimate. The inference at the input 1920 takes about 0.8 s per frame on the processor of the test computer.

**Control question 2.** Why does the enlargement of the input above the size of the frame (640 → 1920) raise the recall, although no new information is added to the frame?
""")

code(r"""
#@title 5.3.2 Single frame of a UAV (code shown)
SEQUENCE = "1-05" #@param ["1-05", "1-07", "1-08", "1-09"]
FRAME = 100 #@param {type:"integer"}
IMG_SIZE = 1920 #@param [640, 1280, 1920] {type:"raw"}
CONF = 0.25 #@param {type:"slider", min:0.05, max:0.9, step:0.05}
IOU = 0.7 #@param {type:"slider", min:0.3, max:0.95, step:0.05}

tracks = m3.tracks(SEQUENCE)
FRAME = int(np.clip(FRAME, 1, m3.n_frames(SEQUENCE)))
img = m3.frames(SEQUENCE, start=FRAME - 1, count=1)[0]
gt = [[r.x, r.y, r.x + r.w, r.y + r.h] for r in tracks[tracks.frame == FRAME].itertuples()]
rows, shown = [], None
for s in (640, 1280, 1920):
    b, sc, c, sp = aitbc.yolo_predict(yolo, img, imgsz=s, conf=CONF, iou=IOU, classes=VEHICLES)
    tp, _ = aitbc.match_greedy(b, sc, gt, 0.5)
    rows.append({"input": s, "detections": len(b), "correct (IoU >= 0.5)": int(tp.sum()), "true vehicles": len(gt),
                 "precision": round(tp.sum() / max(len(b), 1), 3), "recall": round(tp.sum() / max(len(gt), 1), 3),
                 f"inference, ms ({DEVICE})": round(sp["inference"], 1)})
    if s == IMG_SIZE:
        shown = (b, sc, tp)
display(pd.DataFrame(rows).set_index("input"))
b, sc, tp = shown
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img)
for g in gt:
    ax.add_patch(plt.Rectangle(g[:2], g[2] - g[0], g[3] - g[1], fill=False, ec="lime", lw=1.2))
for bb, s, t in zip(b, sc, tp):
    ax.add_patch(plt.Rectangle(bb[:2], bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="red" if t else "magenta", lw=1.2, ls="-" if t else "--"))
    ax.text(bb[0], bb[1] - 1, f"{s:.2f}", color="yellow", fontsize=6)
ax.set_title(f"M3OT {SEQUENCE}, frame {FRAME}: true boxes green, correct detections red, false magenta "
             f"(input {IMG_SIZE}, conf {CONF})", fontsize=9)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB09_532_single_frame.png"), dpi=110)
plt.show()
""")

md(r"""
## Task 5.3.3. Detection on a series of UAV frames

**Method.** The detector is run on `N_FRAMES` consecutive frames (every `STEP`-th frame); for every frame the number of detections, the number of correct detections and the number of true vehicles are counted. The **stability** of the detector is seen in the curve of the recall over the frames: a vehicle that is found in one frame and missed in the next gives a flickering result that the tracker of Notebook 08 must bridge.

**Parameters.** `N_FRAMES`, `STEP` — length and step of the series; the sequence, the input and the thresholds of task 5.3.2.

**Expected result.** The curves of the true vehicles, the detections and the correct detections over the frames; the precision, the recall and the frame rate of the series. The annotated frames are kept for the video of task 5.3.6.

**What to observe.** In 60 frames (frames 100–218, every second frame) the detector reaches a precision of 0.88 but a recall of only 0.08: in every frame less than half of the vehicles are found, and after frame 165 almost none. The curve of the correct detections jumps from frame to frame while the number of true vehicles (15–21) changes slowly: this flicker is what a tracker must bridge (Notebook 08, block 5.2.3). On the processor the series runs at about 1 frame per second.

**Control question 3.** Which share of the frames of the series has a recall below 0.5, and which conditions of the scene (size, contrast, occlusion) cause it?
""")

code(r"""
#@title 5.3.3 Series of UAV frames (code shown)
N_FRAMES = 60 #@param {type:"slider", min:10, max:200, step:10}
STEP = 2 #@param {type:"slider", min:1, max:5, step:1}

n_all = m3.n_frames(SEQUENCE)
start = max(0, min(FRAME - 1, n_all - N_FRAMES * STEP))
SERIES = m3.frames(SEQUENCE, start=start, count=N_FRAMES, step=STEP)
SERIES_IDX = [start + 1 + k * STEP for k in range(len(SERIES))]
DET_SERIES, per_frame = [], []
t0 = time.time()
for f, im in zip(SERIES_IDX, SERIES):
    b, sc, c, sp = aitbc.yolo_predict(yolo, im, imgsz=IMG_SIZE, conf=CONF, iou=IOU, classes=VEHICLES)
    gt = [[r.x, r.y, r.x + r.w, r.y + r.h] for r in tracks[tracks.frame == f].itertuples()]
    tp, _ = aitbc.match_greedy(b, sc, gt, 0.5)
    DET_SERIES.append((f, b, sc, c, tp))
    per_frame.append({"frame": f, "true": len(gt), "detections": len(b), "correct": int(tp.sum())})
t_series = time.time() - t0
PF = pd.DataFrame(per_frame)
P = PF.correct.sum() / max(PF.detections.sum(), 1)
R = PF.correct.sum() / max(PF.true.sum(), 1)
low = np.mean(PF.correct / PF.true.clip(lower=1) < 0.5)
display(pd.DataFrame({"value": [len(SERIES), f"{P:.3f}", f"{R:.3f}", f"{low:.0%}", f"{len(SERIES) / t_series:.1f} ({DEVICE})"]},
                     index=["frames", "precision", "recall", "frames with recall < 0.5", "frames per second"]))
fig, ax = plt.subplots(figsize=(12, 3.4))
ax.plot(PF.frame, PF.true, "k--", label="true vehicles")
ax.plot(PF.frame, PF.detections, color="#ef6c00", label="detections")
ax.plot(PF.frame, PF.correct, color="#1565c0", label="correct detections")
ax.set_xlabel("frame"); ax.set_ylabel("objects"); ax.legend(fontsize=8)
ax.set_title(f"M3OT {SEQUENCE}: {MODEL}, input {IMG_SIZE}, conf {CONF}", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB09_533_series.png"), dpi=110)
plt.show()
RESULT_53 = {"model": MODEL, "sequence": SEQUENCE, "img_size": IMG_SIZE, "conf": CONF,
             "series": {"frames": len(SERIES), "precision": round(float(P), 4), "recall": round(float(R), 4),
                        "fps": round(len(SERIES) / t_series, 2), "device": DEVICE}}
""")

md(r"""
## Task 5.3.4. Detection on FPV frames

**Data.** Frames of the camera of an FPV racing quadrotor of **TII-RATM** (640 × 480, wide-angle lens 155 × 115°, speed above 20 m/s) during a slalom between racing gates in a hall. The scene has no annotation of objects; the task is to see what a detector of everyday classes reports in such frames.

**Method.** All 80 classes are kept; for every frame the detections and the **sharpness** (variance of the Laplacian, Notebook 03) are computed. A detection of a class that is absent in the scene is a false alarm caused by the difference of the domain: the shapes and the textures of the frames differ from the photographs of COCO.

**Parameters.** `T_START`, `DURATION` — interval of the flight, s; `FRAME_STEP` — every $k$-th frame (125 frames per second); `CONF_FPV` — threshold of the confidence.

**Expected result.** The table of the classes reported with their number and mean confidence; four frames with the detections; the number of detections against the sharpness.

**What to observe.** In 48 frames of the slalom the model reports 21 objects, all of them false: 16 times *tv* (in 12 frames), and once *clock*, *laptop*, *chair*, *toilet* and *airplane*. The square racing gates with their dark openings look like the screens of COCO photographs. Most detections appear in the sharper frames; the strongly blurred frames of fast turns give few detections. A detector of everyday classes gives no useful information in such a scene and would raise false alarms in an autonomous function.

**Control question 4.** Why are false alarms on FPV frames more dangerous for an autonomous function (for example, guidance) than for an operator, and what must be done before such a detector is used on an FPV drone?
""")

code(r"""
#@title 5.3.4 FPV frames of a racing quadrotor (code shown)
T_START = 15.0 #@param {type:"number"}
DURATION = 4.0 #@param {type:"number"}
FRAME_STEP = 10 #@param {type:"slider", min:2, max:25, step:1}
CONF_FPV = 0.25 #@param {type:"slider", min:0.1, max:0.9, step:0.05}

FPV, fpv_tab = tii.clip(T_START, T_START + DURATION, step=FRAME_STEP)
rows, cnt, confs = [], collections.Counter(), collections.defaultdict(list)
FPV_DET = []
for k, im in enumerate(FPV):
    b, sc, c, sp = aitbc.yolo_predict(yolo, im, imgsz=640, conf=CONF_FPV)
    FPV_DET.append((b, sc, c))
    for s, cl in zip(sc, c):
        cnt[yolo.names[int(cl)]] += 1
        confs[yolo.names[int(cl)]].append(float(s))
    rows.append({"t": float(fpv_tab.t[k]), "sharpness": aitbc.sharpness(im), "detections": len(b)})
FT = pd.DataFrame(rows)
display(pd.DataFrame({"detections": [cnt[n] for n in cnt], "mean confidence": [round(np.mean(confs[n]), 3) for n in cnt],
                      "frames with the class": [sum(any(yolo.names[int(x)] == n for x in c) for _, _, c in FPV_DET) for n in cnt]},
                     index=list(cnt)).sort_values("detections", ascending=False) if cnt else pd.DataFrame({"result": ["no detections"]}))
order = np.argsort(-FT.detections.values)[:2].tolist() + np.argsort(FT.sharpness.values)[:2].tolist()
fig, axes = plt.subplots(1, 5, figsize=(17, 3.2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.1]})
for a, k in zip(axes[:4], order):
    b, sc, c = FPV_DET[k]
    a.imshow(FPV[k])
    for bb, s, cl in zip(b, sc, c):
        a.add_patch(plt.Rectangle(bb[:2], bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="red", lw=1.2))
        a.text(bb[0], bb[1] - 2, f"{yolo.names[int(cl)]} {s:.2f}", color="yellow", fontsize=6)
    a.set_title(f"t = {FT.t[k]:.2f} s, sharpness {FT.sharpness[k]:.0f}", fontsize=8)
    a.axis("off")
axes[4].scatter(FT.sharpness, FT.detections, s=14, color="#1565c0")
axes[4].set_xscale("log"); axes[4].set_xlabel("sharpness (log)"); axes[4].set_ylabel("detections in the frame")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB09_534_fpv.png"), dpi=110)
plt.show()
RESULT_53["fpv"] = {"frames": len(FPV), "detections": int(sum(cnt.values())), "classes": dict(cnt)}
""")

md(r"""
## Task 5.3.5. Detection on a thermal frame

**Data.** Night frames of **HIT-UAV** (thermal camera Zenmuse XT2, 640 × 512, 60–130 m) with boxes of persons. The frames are *white-hot*: warm objects are bright.

**Method.** A thermal frame is a grey image; for YOLO it is repeated in three channels. Two variants are compared: the frame as recorded (white-hot) and the inverted frame (*black-hot*, $255 - I$), in which a warm person is a dark silhouette on a light background, as in the visible photographs of COCO. A person is found if a detection of the class *person* has IoU $\geq 0.5$ with its box.

**Parameters.** `N_THERMAL` — night frames, taken from different flight conditions; `CONF_T` — threshold of the confidence; `SHOW_T` — number of the frame shown.

**Expected result.** The table of the recall of persons and of the false detections per frame for the two polarities and the inputs 640 and 1280; one frame in both polarities with the detections.

**What to observe.** In 30 night frames with 258 annotated persons the frames as recorded (white-hot) give almost nothing: 0 persons at the input 640 and 1 at 1280. After the inversion (black-hot) the model finds 32 persons at the input 1280 (recall 0.12) with 0.27 false detections per frame: a dark silhouette on a light background is closer to the photographs of COCO. The result is still far from useful; for thermal channels a detector is built on thermal data: the classical warm-object detector of Notebook 05 finds about two thirds of the night persons of HIT-UAV, and the verifier of Notebook 07 removes most of its false alarms.

**Control question 5.** Why does the polarity of a thermal frame change the result of a detector trained on visible photographs, and why is the inversion not a solution for an onboard system?
""")

code(r"""
#@title 5.3.5 Thermal frame: polarity and size of the input (code shown)
N_THERMAL = 30 #@param {type:"slider", min:10, max:100, step:10}
CONF_T = 0.25 #@param {type:"slider", min:0.05, max:0.9, step:0.05}
SHOW_T = 0 #@param {type:"integer"}

night = hit.images[hit.images.night & (hit.images.get("Person", 0) >= 3)].copy()
night["condition"] = night.altitude.astype(str) + "_" + night.angle.astype(str)
pick = night.sample(frac=1.0, random_state=0).groupby("condition").head(3)   # at most 3 frames per flight condition
TH_FILES = pick.sample(min(N_THERMAL, len(pick)), random_state=0).file.tolist()
hit.prefetch(TH_FILES)
rows, examples = [], {}
for pol in ("white-hot (as recorded)", "black-hot (inverted)"):
    for s in (640, 1280):
        found = n_p = fp = 0
        for k, f in enumerate(TH_FILES):
            g, objs = hit.load(f)
            x = 255 - g if pol.startswith("black") else g
            b, sc, c, sp = aitbc.yolo_predict(yolo, np.dstack([x] * 3), imgsz=s, conf=CONF_T, classes=[0])
            persons = [o["box"] for o in objs if o["cls"] == "Person"]
            tp, gi = aitbc.match_greedy(b, sc, persons, 0.5)
            found += int(tp.sum()); n_p += len(persons); fp += int((~tp).sum())
            if k == SHOW_T % len(TH_FILES) and s == 1280:
                examples[pol] = (x, b, persons, tp)
        rows.append({"polarity": pol, "input": s, "persons": n_p, "found": found, "recall": round(found / max(n_p, 1), 3),
                     "false detections per frame": round(fp / len(TH_FILES), 2)})
TAB_535 = pd.DataFrame(rows).set_index(["polarity", "input"])
display(TAB_535)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for a, (pol, (x, b, persons, tp)) in zip(axes, examples.items()):
    a.imshow(x, cmap="gray", vmin=0, vmax=255)
    for p in persons:
        a.add_patch(plt.Rectangle(p[:2], p[2] - p[0], p[3] - p[1], fill=False, ec="lime", lw=1))
    for bb, t in zip(b, tp):
        a.add_patch(plt.Rectangle(bb[:2], bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="red" if t else "magenta", lw=1.3))
    a.set_title(f"{TH_FILES[SHOW_T % len(TH_FILES)]}, {pol}, input 1280: {int(tp.sum())} of {len(persons)} persons found", fontsize=9)
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB09_535_thermal.png"), dpi=110)
plt.show()
RESULT_53["thermal"] = TAB_535.reset_index().to_dict("records")
""")

md(r"""
## Task 5.3.6. Saving of the annotated video and of the table of detections

**Method.** The frames of the series of task 5.3.3 are annotated (correct detections red, false magenta, true boxes green, the number of the frame and the counts) and written by OpenCV into an MP4 file (codec `mp4v`, 5 frames per second for a step of 2). The detections are written into a CSV table: frame, class, confidence, box, correct or not. The video is read back to check the number of frames and the size.

**Parameters.** `FPS_VIDEO` — frame rate of the saved video; `SCALE_VIDEO` — scale of the frames (2 — the frames are enlarged for viewing).

**Expected result.** The paths and sizes of the video and of the table, the properties of the video read back, the first and the last frame.

**Control question 6.** Which data must be saved together with the video so that another operator can check and repeat the result?
""")

code(r"""
#@title 5.3.6 Annotated video and the table of detections (code shown)
FPS_VIDEO = 5 #@param {type:"slider", min:1, max:25, step:1}
SCALE_VIDEO = 2 #@param [1, 2] {type:"raw"}

frames_out, rows = [], []
for (f, b, sc, c, tp), im in zip(DET_SERIES, SERIES):
    gt = [[r.x, r.y, r.x + r.w, r.y + r.h] for r in tracks[tracks.frame == f].itertuples()]
    out = cv2.resize(im, None, fx=SCALE_VIDEO, fy=SCALE_VIDEO, interpolation=cv2.INTER_NEAREST) if SCALE_VIDEO != 1 else im.copy()
    for g in gt:
        g = (np.array(g) * SCALE_VIDEO).astype(int)
        cv2.rectangle(out, tuple(g[:2]), tuple(g[2:]), (0, 255, 0), 1)
    for bb, s, cl, t in zip(b, sc, c, tp):
        q = (np.array(bb) * SCALE_VIDEO).astype(int)
        cv2.rectangle(out, tuple(q[:2]), tuple(q[2:]), (255, 0, 0) if t else (255, 0, 255), 2)
        rows.append({"frame": f, "class": yolo.names[int(cl)], "confidence": round(float(s), 4),
                     "x1": round(float(bb[0]), 1), "y1": round(float(bb[1]), 1), "x2": round(float(bb[2]), 1),
                     "y2": round(float(bb[3]), 1), "correct": bool(t)})
    cv2.putText(out, f"M3OT {SEQUENCE} frame {f}: {int(tp.sum())}/{len(gt)} found, {len(b) - int(tp.sum())} false",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    frames_out.append(out)
vpath = aitbc.write_video(os.path.join(VIDEO_DIR, f"NB09_detections_{SEQUENCE}.mp4"), frames_out, fps=FPS_VIDEO)
cpath = os.path.join(REPORT_DIR, f"NB09_detections_{SEQUENCE}.csv")
pd.DataFrame(rows).to_csv(cpath, index=False)
back, fps_back = aitbc.read_video(vpath)
display(pd.DataFrame({"value": [vpath, f"{os.path.getsize(vpath) / 1e6:.2f} MB", len(back), fps_back,
                                f"{back[0].shape[1]} x {back[0].shape[0]}", cpath, len(rows)]},
                     index=["video", "size of the video", "frames read back", "frame rate", "frame size",
                            "table of detections", "rows of the table"]))
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
for a, k in zip(ax, (0, len(back) - 1)):
    a.imshow(back[k]); a.set_title(f"frame {k + 1} of {len(back)} read back from the file", fontsize=9); a.axis("off")
plt.tight_layout()
plt.show()
RESULT_53["video"] = os.path.relpath(vpath, PROJECT_ROOT)
RESULT_53["detections_csv"] = os.path.relpath(cpath, PROJECT_ROOT)
update_project_json("5.3", {"lesson_5_3": RESULT_53})
""")

md(r"""
### Report of Lesson 5.3

Double-click this cell and write the answers.

1. Model, parameters, operations, classes of interest; answer to control question 1:
2. Single frame: precision and recall for the inputs 640, 1280, 1920; answer to control question 2:
3. Series of frames: precision, recall, frame rate, share of weak frames; answer to control question 3:
4. FPV frames: classes reported, relation with the sharpness; answer to control question 4:
5. Thermal frames: recall of persons for the two polarities; answer to control question 5:
6. Saved video and table; answer to control question 6:
""")

# =====================================================================================
md(r"""
---
# Lesson 5.4 (Practical class, mini-project). A detector and tracker of drones for a counter-UAV post

**Learning questions**
1. Formation of a training data set in the YOLO format from an annotated data set.
2. Baseline: a detector trained on COCO.
3. Training (fine-tuning) of YOLO11 for a new class.
4. Evaluation on the test part, the choice of the working threshold on the validation part, small targets.
5. Tracking of a drone in a video and the measures of single-object tracking.
6. Export to ONNX and the check of real time.
7. Report of the project.

**Task of the project.** A post of air defence has a daylight camera directed to the sky. Lesson 5.1 showed that a YOLO11 model trained on COCO has no class *drone* and finds drones only as *airplanes, birds or kites*. In the project a detector of the class **UAV** is trained on the training part of **DUT Anti-UAV**, its threshold is chosen on the validation part, its quality is measured on the test part and compared with the baseline, the detector is joined with ByteTrack to follow a drone in a video, and the model is exported to ONNX for an onboard or a post computer.

**Result of the project.** The trained detector `models/uav_yolo11n_best.pt` and `models/uav_yolo11n_best.onnx`, the tables of quality, the video `outputs/video/NB09_tracking_<video>.mp4`, the report `outputs/reports/NB09_project_report.md` and the parameters in `project.json`.

**Time.** About 4 hours: the tasks 5.4.1–5.4.2 (30 min), the training 5.4.3 (with a GPU a few minutes for the default settings, on the processor of the test computer about 37 min; then experiments with the number of images, epochs and the input), the evaluation and the analysis 5.4.4–5.4.5 (60 min), the tracking 5.4.6 (30 min), the export 5.4.7 (20 min), the report 5.4.8 and its defence (60 min).
""")

md(r"""
## Task 5.4.1. Training data set in the YOLO format

**Format.** For an image `images/<part>/<name>.jpg` the file `labels/<part>/<name>.txt` has one line per object: `class cx cy w h`, where the centre and the size of the box are given as shares of the width and the height of the image (Notebook 06):
$$c_x = \frac{x_1 + x_2}{2W},\quad c_y = \frac{y_1 + y_2}{2H},\quad w = \frac{x_2 - x_1}{W},\quad h = \frac{y_2 - y_1}{H}.$$
The file `data.yaml` gives the folders of the parts and the names of the classes.

**Parts.** The official parts of DUT Anti-UAV are used: training images from *train*, validation images (choice of the epoch and of the threshold) from *val*, test images from *test*. Images larger than `MAX_SIDE` are reduced; the relative labels do not change. Because the images of a data set are often frames of the same videos, the pairs *test image — nearest training image* by the difference hash (Notebook 06) are shown for a **visual check** of near duplicates. For images of a plain sky the hash carries little information (a small drone changes almost no bit), so a small distance alone does not prove a duplicate.

**Parameters.** `N_TRAIN`, `N_VAL`, `N_TEST` — images taken from the three parts; `MAX_SIDE` — maximum side of a stored image; `SEED_P` — choice of the images.

**Expected result.** The table of the parts (images, drones, sizes of the drones in the stored image and in the input 640), a gallery of training images with the boxes read back from the label files and the four closest pairs of test and training images.

**What to observe.** The parts have 400, 120 and 300 images with one drone in almost every image. Stored with the maximum side 1280, the median drone is 19–28 px; in the input 640 it is 10–14 px, and 75 % of the training drones are smaller than 16 px: the task is a small-target task. The four closest pairs by the difference hash all have the distance 0, and they are not only a similar sky: every pair is the same scene from the same camera position with the drone at another place. The official parts of DUT Anti-UAV share scenes between the training and the test part, so the test measures the detector partly on backgrounds it has seen.

**Control question 7.** Why are the boxes in the gallery drawn from the label files and not from the annotation of the data set?
""")

code(r"""
#@title 5.4.1 Data set of DUT Anti-UAV in the YOLO format (code shown)
N_TRAIN = 400 #@param {type:"slider", min:100, max:5200, step:100}
N_VAL = 120 #@param {type:"slider", min:50, max:1000, step:10}
N_TEST = 300 #@param {type:"slider", min:100, max:2200, step:100}
MAX_SIDE = 1280 #@param [640, 960, 1280, 1920] {type:"raw"}
SEED_P = 0 #@param {type:"integer"}

DS_ROOT = os.path.join(WORK_DIR, "datasets", "dut_uav")
ann = {s: dut.annotations(s) for s in ("train", "val", "test")}
rng = np.random.default_rng(SEED_P)
SEL = {}
for s, n in (("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)):
    ok = np.where(ann[s].n > 0)[0]
    SEL[s] = ann[s].image_id.values[np.sort(rng.choice(ok, min(n, len(ok)), replace=False))].tolist()
key = json.dumps({s: [len(v), v[:5]] for s, v in SEL.items()}) + f"|{MAX_SIDE}"
key_file = os.path.join(DS_ROOT, "selection.txt")
t0 = time.time()
if not (os.path.exists(key_file) and open(key_file).read() == key):
    shutil.rmtree(DS_ROOT, ignore_errors=True)
    for s in SEL:
        dut.to_yolo(DS_ROOT, s, s, SEL[s], max_side=MAX_SIDE)
    os.makedirs(DS_ROOT, exist_ok=True)
    open(key_file, "w").write(key)
DATA_YAML = os.path.join(DS_ROOT, "data.yaml")
with open(DATA_YAML, "w") as fh:
    fh.write(f"path: {DS_ROOT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: UAV\n")
print(f"Data set {DS_ROOT} ready in {time.time() - t0:.0f} s")


def read_labels(part, name):
    # boxes in pixels of the stored image, read back from the YOLO label file
    img_path = os.path.join(DS_ROOT, "images", part, f"{name}.jpg")
    h, w = cv2.imread(img_path).shape[:2]
    boxes = []
    for line in open(os.path.join(DS_ROOT, "labels", part, f"{name}.txt")):
        c, cx, cy, bw, bh = map(float, line.split())
        boxes.append(aitbc.yolo_to_xyxy(cx, cy, bw, bh, w, h))
    return img_path, np.array(boxes, float).reshape(-1, 4), (w, h)


DS = {s: [read_labels(s, n) for n in SEL[s]] for s in SEL}
rows = []
for s, items in DS.items():
    side = np.array([math.sqrt((b[2] - b[0]) * (b[3] - b[1])) for _, bs, _ in items for b in bs])
    side640 = np.array([math.sqrt((b[2] - b[0]) * (b[3] - b[1])) * 640 / max(wh) for _, bs, wh in items for b in bs])
    rows.append({"part": s, "images": len(items), "drones": len(side), "median side, px (stored image)": round(float(np.median(side)), 1),
                 "median side in the input 640, px": round(float(np.median(side640)), 1),
                 "side < 16 px in the input 640": f"{np.mean(side640 < 16):.0%}"})
display(pd.DataFrame(rows).set_index("part"))


def small_hash(path):
    return aitbc.dhash(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (64, 36), interpolation=cv2.INTER_AREA))


H_TR = np.array([small_hash(p) for p, _, _ in DS["train"]])
H_TE = np.array([small_hash(p) for p, _, _ in DS["test"]])
d_near, j_near = aitbc.hamming_nearest(H_TE, H_TR)
fig, axes = plt.subplots(1, 6, figsize=(17, 2.6))
for a, (p, bs, wh) in zip(axes, DS["train"][:6]):
    im = cv2.imread(p)[..., ::-1].copy()
    for b in bs.astype(int):
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (255, 40, 40), max(2, wh[0] // 300))
    a.imshow(im); a.set_title(f"{os.path.basename(p)}: {len(bs)} box(es)", fontsize=8); a.axis("off")
plt.tight_layout()
plt.show()
closest = np.argsort(d_near)[:4]
fig, axes = plt.subplots(2, 4, figsize=(15, 5.4))
for k, t in enumerate(closest):
    for r_, (p, title) in enumerate(((DS["test"][t][0], f"test {os.path.basename(DS['test'][t][0])}"),
                                     (DS["train"][j_near[t]][0], f"train {os.path.basename(DS['train'][j_near[t]][0])}, distance {d_near[t]}"))):
        axes[r_, k].imshow(cv2.imread(p)[..., ::-1]); axes[r_, k].set_title(title, fontsize=8); axes[r_, k].axis("off")
plt.suptitle("Closest pairs test - training by the difference hash (64 bits): duplicates or only a similar sky?", fontsize=10)
plt.tight_layout()
plt.show()
RESULT_54 = {"data": {"n_train": len(DS["train"]), "n_val": len(DS["val"]), "n_test": len(DS["test"]),
                      "max_side": MAX_SIDE, "closest_hash_distances": [int(d_near[t]) for t in closest]}}
""")

md(r"""
## Task 5.4.2. Baseline: the model trained on COCO

**Method.** The YOLO11 model with the weights of COCO is evaluated on the test images: the detections of the classes *airplane, bird, kite* are taken as detections of a drone (Lesson 5.1). The measures are those of Notebook 08: AP50 and AP50-95 (COCO, 101 points), and at the threshold 0.25 the precision, the recall, the false detections per image and the recall by the size of the drone in the stored image.

**Parameters.** `BASE_MODEL` — weights from which the training of task 5.4.3 starts; `IMG_SIZE` — input of the training and of all evaluations of the project.

**Expected result.** The table of the baseline and its time per image.

**What to observe.** The model of COCO reaches AP50 = 0.439 and AP50-95 = 0.258 on the 300 test images. At the threshold 0.25 it finds about a third of the drones (recall 0.33) with a precision of 0.75; the recall depends strongly on the size: 0.06 for drones below 16 px, 0.25 for 16–32 px, 0.40 for 32–64 px and 0.55 for larger drones. One image takes about 40 ms on the processor.

**Control question 8.** Why must the baseline be measured on the same test images, with the same input and the same measures as the trained model?
""")

code(r"""
#@title 5.4.2 Baseline: YOLO11 with the weights of COCO (code shown)
BASE_MODEL = "yolo11n.pt" #@param ["yolo11n.pt", "yolo11s.pt"]
IMG_SIZE = 640 #@param [640, 960, 1280] {type:"raw"}

COCO_UAV = [4, 14, 33]  # airplane, bird, kite
THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def predict_set(model, items, imgsz, classes=None, conf=0.01):
    preds, t = [], 0.0
    for p, _, _ in items:
        img = cv2.imread(p)[..., ::-1]
        t1 = time.time()
        b, s, c, _ = aitbc.yolo_predict(model, img, imgsz=imgsz, conf=conf, classes=classes)
        t += time.time() - t1
        preds.append((b, s))
    return preds, 1000 * t / max(len(items), 1)


def summary(ev, ms, conf):
    out = {"AP50": round(ev["AP50"], 3), "AP50-95": round(ev["AP"], 3), "threshold": round(conf, 3),
           "precision": round(ev["precision"], 3), "recall": round(ev["recall"], 3),
           "false detections per image": round(ev["false_per_image"], 2)}
    out.update({f"recall {k}": round(v[0], 3) for k, v in ev["recall_by_size"].items()})
    out[f"ms per image ({DEVICE})"] = round(ms, 1)
    return out


SIZE_EDGES = (0, 16, 32, 64, 1e9)
GT_TEST = [bs for _, bs, _ in DS["test"]]
GT_VAL = [bs for _, bs, _ in DS["val"]]
base = load_yolo(BASE_MODEL)
PRED_BASE, MS_BASE = predict_set(base, DS["test"], IMG_SIZE, COCO_UAV)
EV_BASE = aitbc.evaluate_detections(PRED_BASE, GT_TEST, iou_thrs=THRS, size_edges=SIZE_EDGES, conf=0.25)
TAB_BASE = pd.DataFrame([summary(EV_BASE, MS_BASE, 0.25)], index=[f"{BASE_MODEL} (COCO), classes airplane+bird+kite"])
display(TAB_BASE.T)
""")

md(r"""
## Task 5.4.3. Training of YOLO11 for the class UAV

**Method.** The training starts from the weights of COCO (transfer learning, Lesson 4.4): the backbone and the neck keep their features, the output layer of the classes is replaced for one class. Ultralytics trains with the augmentation of the images (mosaic of four images, scaling, shifts, colour changes, horizontal flips) and the loss
$$L = \lambda_{\text{box}} L_{\text{CIoU}} + \lambda_{\text{cls}} L_{\text{BCE}} + \lambda_{\text{dfl}} L_{\text{DFL}},$$
where $L_{\text{CIoU}}$ measures the overlap, the distance of the centres and the shape of the predicted and the true box, $L_{\text{BCE}}$ is the binary cross-entropy of the class and $L_{\text{DFL}}$ the distribution focal loss of the positions of the sides. After every epoch mAP is computed on the validation part; the weights of the best epoch are saved (`best.pt`).

**Parameters.** `EPOCHS` — epochs; `BATCH` — images per step; `FREEZE` — first layers of the network that are not trained (0 — all layers are trained, 10 — the backbone is frozen); `PATIENCE` — epochs without improvement before the training stops. The input is `IMG_SIZE` of task 5.4.2.

**Expected result.** One line per epoch (losses, mAP50 and mAP50-95 of the validation part, time), the curves of the training, the time per epoch and the best epoch; the file `models/uav_yolo11n_best.pt`.

**What to observe.** The 20 epochs took 37 min on the processor of the test computer (110 s per epoch, 400 images). The mAP of the validation part first falls from 0.42 after epoch 1 to almost 0 at epoch 4 (the new output layer and the large steps of the first epochs disturb the pretrained features; the validation loss of the class has a peak there — note the logarithmic scale) and then rises; the best epoch is 18 with mAP50 = 0.654 and mAP50-95 = 0.356. The curve has not clearly levelled off by epoch 20 (0.34–0.36 in the last three epochs): more epochs and more training images (with a GPU) are the first experiments of the project.

**Control question 9.** What does the curve of mAP of the validation part show if it rises and then falls, and how does `PATIENCE` use it?
""")

code(r"""
#@title 5.4.3 Training of YOLO11 for the class UAV (code shown)
EPOCHS = 20 #@param {type:"slider", min:1, max:100, step:1}
BATCH = 16 #@param [8, 16, 32] {type:"raw"}
FREEZE = 0 #@param {type:"slider", min:0, max:22, step:1}
PATIENCE = 10 #@param {type:"slider", min:3, max:50, step:1}

RUN_DIR = os.path.join(WORK_DIR, "runs_uav")
model = load_yolo(BASE_MODEL)
T_EPOCH = []


def epoch_line(trainer):
    if trainer.epoch >= trainer.epochs:     # the final validation after the last epoch
        return
    m = trainer.metrics
    T_EPOCH.append(time.time())
    dt = T_EPOCH[-1] - (T_EPOCH[-2] if len(T_EPOCH) > 1 else T0_TRAIN)
    ll = trainer.label_loss_items(trainer.tloss, prefix="train")    # dict of the mean losses of the epoch
    loss = [ll.get(f"train/{k}_loss", float("nan")) for k in ("box", "cls", "dfl")]
    print(f"epoch {trainer.epoch + 1:3d}/{trainer.epochs}: loss box {loss[0]:.3f} cls {loss[1]:.3f} dfl {loss[2]:.3f} | "
          f"val mAP50 {m.get('metrics/mAP50(B)', 0):.3f} mAP50-95 {m.get('metrics/mAP50-95(B)', 0):.3f} | {dt:.0f} s", flush=True)


model.add_callback("on_fit_epoch_end", epoch_line)
T0_TRAIN = time.time()
model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, device=0 if DEVICE == "cuda" else "cpu",
            workers=2 if IN_COLAB else 0, freeze=FREEZE, patience=PATIENCE, plots=False, amp=DEVICE == "cuda",
            project=RUN_DIR, name="uav", exist_ok=True, seed=0, deterministic=True, verbose=False)
T_TRAIN = time.time() - T0_TRAIN
best_pt = os.path.join(RUN_DIR, "uav", "weights", "best.pt")
UAV_PT = os.path.join(MODEL_DIR, "uav_yolo11n_best.pt")
shutil.copy2(best_pt, UAV_PT)
hist = pd.read_csv(os.path.join(RUN_DIR, "uav", "results.csv"))
hist.columns = [c.strip() for c in hist.columns]
ib = int(hist["metrics/mAP50-95(B)"].idxmax())
display(pd.DataFrame({"value": [len(hist), f"{T_TRAIN / 60:.1f} min ({DEVICE})", f"{T_TRAIN / len(hist):.0f} s",
                                int(hist.epoch[ib]), f"{hist['metrics/mAP50(B)'][ib]:.3f}", f"{hist['metrics/mAP50-95(B)'][ib]:.3f}",
                                f"{os.path.getsize(UAV_PT) / 1e6:.1f} MB"]},
                     index=["epochs done", "time of the training", "time per epoch", "best epoch (mAP50-95 of val)",
                            "mAP50 of val at the best epoch", "mAP50-95 of val at the best epoch", "file uav_yolo11n_best.pt"]))
fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.6))
for c_, n in (("train/box_loss", "box"), ("train/cls_loss", "class"), ("train/dfl_loss", "DFL")):
    ax[0].plot(hist.epoch, hist[c_], label=f"loss {n}, training part")
for c_, n in (("val/box_loss", "box"), ("val/cls_loss", "class")):
    ax[0].plot(hist.epoch, hist[c_], "--", label=f"loss {n}, validation part")
ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (log scale)"); ax[0].legend(fontsize=7)
ax[1].plot(hist.epoch, hist["metrics/mAP50(B)"], label="mAP50")
ax[1].plot(hist.epoch, hist["metrics/mAP50-95(B)"], label="mAP50-95")
ax[1].axvline(hist.epoch[ib], color="grey", ls="--", lw=1)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation part"); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB09_543_training.png"), dpi=110)
plt.show()
RESULT_54["training"] = {"base": BASE_MODEL, "img_size": IMG_SIZE, "epochs": int(len(hist)), "batch": BATCH,
                         "freeze": FREEZE, "minutes": round(T_TRAIN / 60, 2), "device": DEVICE, "best_epoch": int(hist.epoch[ib]),
                         "val_map50": round(float(hist["metrics/mAP50(B)"][ib]), 4),
                         "val_map50_95": round(float(hist["metrics/mAP50-95(B)"][ib]), 4)}
""")

md(r"""
## Task 5.4.4. Evaluation on the test part and the working threshold

**Method.**
1. The working threshold is chosen on the **validation** part as the confidence of the maximum $F_1 = \frac{2PR}{P+R}$ (for the baseline in the same way); the test part is not used for any choice.
2. On the **test** part the trained detector and the baseline are compared: AP50, AP50-95, the precision and the recall at the working threshold, the false detections per image, the recall by size.
3. For a check the function `val` of Ultralytics computes mAP on the same test images.
4. The errors are shown: missed drones (false negatives) and false detections (false positives) at the working threshold.

**Expected result.** The table *baseline — trained detector*, the precision-recall curves of both, the galleries of the errors.

**What to observe.** The working thresholds chosen on the validation part are 0.43 for the trained detector and 0.075 for the baseline. On the test part the training raises AP50 from 0.439 to 0.727 and AP50-95 from 0.258 to 0.370 (Ultralytics `val`: 0.731 and 0.370). At the working thresholds the precision grows from 0.58 to 0.91 and the recall from 0.46 to 0.62, while the false detections per image fall from 0.34 to 0.06. The gain is in the small drones (recall below 16 px 0.26 → 0.56, 16–32 px 0.42 → 0.63); for drones of 64 px and more there is no gain (0.62 → 0.61): the training images contain mostly small drones, and several of the missed drones are large drones close to the camera. The strongest false detections lie on a lamp post against the clouds, on a hillside with a power line and on a logo; one of them shows a drone: either the annotation misses it or its box differs from the annotated box (IoU < 0.5) — an error of the annotation is counted as an error of the detector.

**Control question 10.** Which errors remain after the training (size, background, illumination), and which data must be added to the training part to reduce them?
""")

code(r"""
#@title 5.4.4 Evaluation of the trained detector on the test part (code shown)
uav = load_yolo(UAV_PT)
PRED_VAL, _ = predict_set(uav, DS["val"], IMG_SIZE)
CONF_OP, F1_VAL = aitbc.f1_threshold(aitbc.evaluate_detections(PRED_VAL, GT_VAL)["curve"])
PRED_VAL_B, _ = predict_set(base, DS["val"], IMG_SIZE, COCO_UAV)
CONF_B, F1_B = aitbc.f1_threshold(aitbc.evaluate_detections(PRED_VAL_B, GT_VAL)["curve"])
PRED_FT, MS_FT = predict_set(uav, DS["test"], IMG_SIZE)
EV_FT = aitbc.evaluate_detections(PRED_FT, GT_TEST, iou_thrs=THRS, size_edges=SIZE_EDGES, conf=CONF_OP)
EV_B2 = aitbc.evaluate_detections(PRED_BASE, GT_TEST, iou_thrs=THRS, size_edges=SIZE_EDGES, conf=CONF_B)
TAB_544 = pd.DataFrame([summary(EV_B2, MS_BASE, CONF_B), summary(EV_FT, MS_FT, CONF_OP)],
                       index=[f"baseline {BASE_MODEL} (COCO)", "trained detector UAV"])
display(TAB_544.T)
mv = uav.val(data=DATA_YAML, split="test", imgsz=IMG_SIZE, batch=16, plots=False, verbose=False,
             device=0 if DEVICE == "cuda" else "cpu", workers=0, project=RUN_DIR, name="val_test", exist_ok=True)
print(f"Check by Ultralytics val on the test part: mAP50 = {mv.box.map50:.3f}, mAP50-95 = {mv.box.map:.3f} "
      f"(the course function: {EV_FT['AP50']:.3f} and {EV_FT['AP']:.3f}; the methods of integration differ slightly)")
fig, ax = plt.subplots(figsize=(6, 4.2))
for ev, name, col in ((EV_B2, "baseline (COCO)", "#ef6c00"), (EV_FT, "trained detector", "#1565c0")):
    cv_ = ev["curve"]
    ax.plot(cv_["recall"], np.maximum.accumulate(cv_["precision"][::-1])[::-1], color=col, lw=2,
            label=f"{name}: AP50 = {ev['AP50']:.3f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
ax.set_title("Test part of DUT Anti-UAV", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB09_544_precision_recall.png"), dpi=110)
plt.show()
fn, fp = [], []
for (p, gb, _), (b, s) in zip(DS["test"], PRED_FT):
    k = s >= CONF_OP
    tp, gi = aitbc.match_greedy(b[k], s[k], gb, 0.5)
    hit = np.zeros(len(gb), bool); hit[gi[gi >= 0]] = True
    fn += [(p, g) for g, h in zip(gb, hit) if not h]
    fp += [(p, bb, ss) for bb, ss, t in zip(b[k], s[k], tp) if not t]
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for a, (p, g) in zip(axes[0], fn[:5]):
    im = cv2.imread(p)[..., ::-1]; side = max(g[2] - g[0], g[3] - g[1])
    a.imshow(aitbc.crop_view(im, g, scale=5, min_side=48))
    a.set_title(f"missed: {g[2] - g[0]:.0f} x {g[3] - g[1]:.0f} px", fontsize=8)
for a, (p, bb, ss) in zip(axes[1], sorted(fp, key=lambda x: -x[2])[:5]):
    im = cv2.imread(p)[..., ::-1]; side = max(bb[2] - bb[0], bb[3] - bb[1])
    a.imshow(aitbc.crop_view(im, bb, scale=5, min_side=48))
    a.set_title(f"false: score {ss:.2f}", fontsize=8)
for a in axes.ravel():
    a.axis("off")
plt.suptitle(f"Errors at the working threshold {CONF_OP:.2f}: {len(fn)} missed drones, {len(fp)} false detections "
             f"in {len(DS['test'])} test images", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB09_544_errors.png"), dpi=110)
plt.show()
RESULT_54["test"] = {"threshold_val_f1": round(CONF_OP, 3), "table": TAB_544.reset_index().to_dict("records"),
                     "ultralytics_map50": round(float(mv.box.map50), 4), "ultralytics_map50_95": round(float(mv.box.map), 4)}
""")

md(r"""
## Task 5.4.5. Small drones: slicing of the frame

**Method.** As in block 5.1.5 of Notebook 08, every stored test image is cut into windows `TILE` × `TILE` with the overlap `OVERLAP`, the trained detector is run on every window and on the whole image, and the boxes are merged by NMS. The variant is compared with the ordinary detection at `IMG_SIZE` by AP50, the recall by size at the working threshold and the time.

**Parameters.** `N_SLICE` — test images used (the slicing is several times slower); `TILE`, `OVERLAP`.

**Expected result.** The table of the two variants.

**What to observe.** On the first 100 test images the slicing (7 runs of the network per image) raises the recall from 0.65 to 0.73 and AP50 from 0.758 to 0.767; the recall of the drones below 16 px grows from 0.60 to 0.80. The cost is a lower precision (0.89 → 0.71, false detections 0.08 → 0.31 per image) and about 9 times more time (43 → 385 ms per image on the processor).

**Control question 11.** For which distances to a drone is the slicing needed, and how would you combine both variants in a post that must keep 25 frames per second?
""")

code(r"""
#@title 5.4.5 Slicing of the frame with the trained detector (code shown)
N_SLICE = 100 #@param {type:"slider", min:20, max:300, step:20}
TILE = 640 #@param [480, 640, 800] {type:"raw"}
OVERLAP = 0.2 #@param {type:"slider", min:0.0, max:0.4, step:0.05}

items = DS["test"][:N_SLICE]
gts = GT_TEST[:N_SLICE]
ev_full = aitbc.evaluate_detections(PRED_FT[:N_SLICE], gts, size_edges=SIZE_EDGES, conf=CONF_OP)
preds, t, runs = [], 0.0, 0
for p, _, _ in items:
    img = cv2.imread(p)[..., ::-1]
    t1 = time.time()
    b, s, c, n = aitbc.predict_tiled(uav, img, tile=TILE, overlap=OVERLAP, imgsz=TILE, conf=0.01, full_frame=True)
    t += time.time() - t1
    runs += n
    preds.append((b, s))
ev_tile = aitbc.evaluate_detections(preds, gts, size_edges=SIZE_EDGES, conf=CONF_OP)
TAB_545 = pd.DataFrame([summary(ev_full, MS_FT, CONF_OP) | {"runs per image": 1},
                        summary(ev_tile, 1000 * t / len(items), CONF_OP) | {"runs per image": round(runs / len(items), 1)}],
                       index=[f"whole image, input {IMG_SIZE}", f"slicing {TILE} px, overlap {OVERLAP} + whole image"])
TAB_545 = TAB_545.drop(columns=["AP50-95"])
display(TAB_545.T)
RESULT_54["slicing"] = TAB_545.reset_index().to_dict("records")
""")

md(r"""
## Task 5.4.6. Tracking of a drone in a video

**Data.** A tracking video of DUT Anti-UAV: one drone per frame, a true box in every frame where the drone is visible.

**Method.** The trained detector is run on every frame with a low threshold (0.05), the detections go to ByteTrack (Notebook 08), and the **target** is the track followed by the post: at the start and after its loss, the track with the highest score becomes the target. Two variants are compared: the detector alone (the detection with the highest score in the frame, if it is above the working threshold) and the detector with ByteTrack. Measures of single-object tracking over the frames with a visible drone: the **success rate** (share of frames with IoU $\geq 0.5$), the area under the success curve (mean over the thresholds of IoU from 0 to 1), the **precision** (share of frames with the error of the centre $\leq 20$ px) and the number of frames without a box.

**Parameters.** `VIDEO` — tracking video (`video02` 83 frames, `video03` 100 frames, 1280 × 720; `video04` 341 frames, `video06` 200 frames, 1920 × 1080); `MAX_FRAMES` — frames used; `IMG_SIZE_T` — input of the detector for the video.

**Expected result.** The table of the two variants, the curve of IoU over the frames, the saved video with the true box (green) and the box of the target (red).

**What to observe.** In the 100 frames of `video03` (1280 × 720) the drone is visible in every frame. The detector alone, taking the strongest detection above the working threshold, gives no box in 8 frames; its success rate is 0.92 and the area under the success curve 0.737. With ByteTrack the drone is lost only in the first 2 frames (before its track is opened); the success rate is 0.98 and the area 0.783. The tracker keeps the target through frames where the score of the detection is below the working threshold.

**Control question 12.** Why does the tracker keep the drone in frames where the detector alone gives nothing, and when does it create a false target?
""")

code(r"""
#@title 5.4.6 Tracking of a drone in a video (code shown)
VIDEO = "video03" #@param ["video02", "video03", "video04", "video06"]
MAX_FRAMES = 100 #@param {type:"slider", min:20, max:600, step:20}
IMG_SIZE_T = 1280 #@param [640, 960, 1280] {type:"raw"}

FR_V, GT_V = dut.video(VIDEO, max_frames=MAX_FRAMES)
DET_V, t0 = [], time.time()
for im in FR_V:
    b, s, c, _ = aitbc.yolo_predict(uav, im, imgsz=IMG_SIZE_T, conf=0.05)
    DET_V.append((b, s))
ms_v = 1000 * (time.time() - t0) / len(FR_V)
pred_det = [b[np.argmax(s)].tolist() if len(s) and s.max() >= CONF_OP else None for b, s in DET_V]
trk = aitbc.ByteTracker(high=max(CONF_OP, 0.3), low=0.05, new_track=max(CONF_OP, 0.3), max_lost=30)
target, pred_trk = None, []
for b, s in DET_V:
    out = trk.update(b, s)
    ids = {i: (bb, sc) for i, bb, sc in out}
    if target not in ids:
        target = max(ids, key=lambda i: ids[i][1]) if ids else None
    pred_trk.append(ids[target][0] if target in ids else None)
rows = []
for name, pr in (("detector alone (highest score >= working threshold)", pred_det), ("detector + ByteTrack", pred_trk)):
    m = aitbc.sot_metrics(pr, GT_V)
    rows.append({"variant": name, "frames with a visible drone": m["frames"], "success rate (IoU >= 0.5)": round(m["success50"], 3),
                 "area under the success curve": round(m["success_auc"], 3), "precision (centre <= 20 px)": round(m["precision"], 3),
                 "frames without a box": m["missed"]})
TAB_546 = pd.DataFrame(rows).set_index("variant")
print(f"{VIDEO}: {len(FR_V)} frames {FR_V[0].shape[1]}x{FR_V[0].shape[0]}, drone visible in {sum(g is not None for g in GT_V)}; "
      f"detector {ms_v:.0f} ms per frame ({DEVICE}, input {IMG_SIZE_T})")
display(TAB_546)
iou_t = [aitbc.box_iou_matrix([p], [g])[0, 0] if (p is not None and g is not None) else (np.nan if g is None else 0.0)
         for p, g in zip(pred_trk, GT_V)]
iou_d = [aitbc.box_iou_matrix([p], [g])[0, 0] if (p is not None and g is not None) else (np.nan if g is None else 0.0)
         for p, g in zip(pred_det, GT_V)]
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(iou_d, color="#ef6c00", lw=1, label="detector alone")
ax.plot(iou_t, color="#1565c0", lw=1.6, label="detector + ByteTrack")
ax.axhline(0.5, color="grey", ls="--", lw=1)
ax.set_xlabel("frame"); ax.set_ylabel("IoU with the true box"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "NB09_546_tracking_iou.png"), dpi=110)
plt.show()
frames_out = []
for k, (im, g, p) in enumerate(zip(FR_V, GT_V, pred_trk)):
    out = im.copy()
    if g is not None:
        cv2.rectangle(out, tuple(map(int, g[:2])), tuple(map(int, g[2:])), (0, 255, 0), 2)
    if p is not None:
        cv2.rectangle(out, tuple(map(int, p[:2])), tuple(map(int, p[2:])), (255, 0, 0), 2)
    cv2.putText(out, f"{VIDEO} frame {k + 1}: green - true, red - target of the tracker", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    frames_out.append(out)
vpath = aitbc.write_video(os.path.join(VIDEO_DIR, f"NB09_tracking_{VIDEO}.mp4"), frames_out, fps=25)
print(f"Video: {vpath} ({os.path.getsize(vpath) / 1e6:.1f} MB)")
k_show = int(np.nanargmin(np.where(np.isnan(iou_t), np.inf, iou_t))) if np.any(~np.isnan(iou_t)) else 0
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for a, k in zip(ax, (len(frames_out) // 2, k_show)):
    a.imshow(frames_out[k]); a.set_title(f"frame {k + 1}", fontsize=9); a.axis("off")
plt.tight_layout()
plt.show()
RESULT_54["tracking"] = {"video": VIDEO, "frames": len(FR_V), "img_size": IMG_SIZE_T,
                         "table": TAB_546.reset_index().to_dict("records"), "video_file": os.path.relpath(vpath, PROJECT_ROOT)}
""")

md(r"""
## Task 5.4.7. Export to ONNX and the check of real time

**Method.** The trained detector is exported by Ultralytics to ONNX (opset 17, fixed input `IMG_SIZE` × `IMG_SIZE`); the ONNX file is run by ONNX Runtime through the same interface `YOLO("....onnx")`. Two checks are made:
1. **Equivalence.** Both models receive the same images, letterboxed to `IMG_SIZE` × `IMG_SIZE` in advance; the boxes of the two models are matched (IoU > 0.9) and the largest differences of the coordinates and of the scores are found. They must be of the order of the rounding errors.
2. **Quality and time** on the original test images: AP50, the recall at the working threshold, the time per image and the maximum frame rate. On an original frame the PyTorch model is letterboxed to a rectangle (multiple of 32) and the ONNX model with a fixed input to a full square, so their AP50 may differ slightly.

**Parameters.** `N_CHECK` — test images of the check; `REQUIRED_FPS` — frame rate of the camera of the post.

**Expected result.** The size of the ONNX file, the differences of the boxes and scores on the same inputs, the table *PyTorch — ONNX* with AP50, time per image and maximum frame rate.

**What to observe.** On the same letterboxed inputs the two models give exactly the same boxes: 62 of 62 matched, the largest difference of the coordinates is 0.0003 px and of the scores about $10^{-6}$. On the original images AP50 is 0.801 for PyTorch and 0.805 for ONNX on 50 test images (the inputs are letterboxed differently); they take 40 and 38 ms per image on the processor of the test computer (25 and 26 frames per second): just at the limit of real time at 25 frames per second. The ONNX file (10.6 MB) is twice the size of the `.pt` file (5.5 MB) because Ultralytics stores the weights of `best.pt` in 16-bit precision and the ONNX file in 32-bit precision.

**Control question 13.** Which checks of the ONNX file are needed before it is installed on the computer of the post, and which of them were made here?
""")

code(r"""
#@title 5.4.7 Export of the detector to ONNX (code shown)
N_CHECK = 50 #@param {type:"slider", min:10, max:300, step:10}
REQUIRED_FPS = 25 #@param [10, 25, 30] {type:"raw"}

onnx_src = uav.export(format="onnx", imgsz=IMG_SIZE, opset=17, simplify=False, dynamic=False, verbose=False)
UAV_ONNX = os.path.join(MODEL_DIR, "uav_yolo11n_best.onnx")
if os.path.abspath(onnx_src) != os.path.abspath(UAV_ONNX):   # the export is written next to the .pt file
    shutil.copy2(onnx_src, UAV_ONNX)
uav_onnx = YOLO(UAV_ONNX, task="detect")
items = DS["test"][:N_CHECK]
# 1. equivalence: the same letterboxed input for both models
d_box, d_score, n_pt, n_ox, n_match = [], [], 0, 0, 0
for p, _, _ in items:
    lb, r, pad = aitbc.letterbox(cv2.imread(p)[..., ::-1], IMG_SIZE)
    b1, s1, _, _ = aitbc.yolo_predict(uav, lb, imgsz=IMG_SIZE, conf=0.25)
    b2, s2, _, _ = aitbc.yolo_predict(uav_onnx, lb, imgsz=IMG_SIZE, conf=0.25)
    n_pt += len(b1); n_ox += len(b2)
    if len(b1) and len(b2):
        iou = aitbc.box_iou_matrix(b1, b2)
        j = iou.argmax(1)
        ok = iou[np.arange(len(b1)), j] > 0.9
        n_match += int(ok.sum())
        if ok.any():
            d_box.append(np.abs(b1[ok] - b2[j[ok]]).max())
            d_score.append(np.abs(s1[ok] - s2[j[ok]]).max())
print(f"Same letterboxed input {IMG_SIZE} x {IMG_SIZE}, {len(items)} test images: {n_pt} boxes of PyTorch, {n_ox} of ONNX, "
      f"{n_match} matched; largest difference of the coordinates {max(d_box) if d_box else float('nan'):.4f} px, "
      f"of the scores {max(d_score) if d_score else float('nan'):.5f}")
# 2. quality and time on the original test images
p_pt, ms_pt = predict_set(uav, items, IMG_SIZE)
p_ox, ms_ox = predict_set(uav_onnx, items, IMG_SIZE)
rows = []
for name, pr, ms in (("PyTorch (.pt)", p_pt, ms_pt), ("ONNX Runtime (.onnx)", p_ox, ms_ox)):
    ev = aitbc.evaluate_detections(pr, GT_TEST[:N_CHECK], conf=CONF_OP)
    rows.append({"model": name, "AP50": round(ev["AP50"], 3), "recall at the working threshold": round(ev["recall"], 3),
                 f"ms per image ({DEVICE})": round(ms, 1), "max frame rate": round(1000 / ms, 1),
                 f"real time at {REQUIRED_FPS} fps": "yes" if 1000 / ms >= REQUIRED_FPS else "no"})
TAB_547 = pd.DataFrame(rows).set_index("model")
display(TAB_547)
print(f"ONNX file {UAV_ONNX}: {os.path.getsize(UAV_ONNX) / 1e6:.1f} MB")
RESULT_54["onnx"] = {"file": os.path.relpath(UAV_ONNX, PROJECT_ROOT), "mb": round(os.path.getsize(UAV_ONNX) / 1e6, 2),
                     "table": TAB_547.reset_index().to_dict("records"), "boxes_pt_onnx_matched": [n_pt, n_ox, n_match],
                     "max_box_difference_px": round(float(max(d_box)), 5) if d_box else None,
                     "max_score_difference": round(float(max(d_score)), 6) if d_score else None}
""")

md(r"""
## Task 5.4.8. Report of the project

**Method.** The results of the tasks 5.4.1–5.4.7 are collected into a report in Markdown (`outputs/reports/NB09_project_report.md`) and into `project.json`. The conclusions are written by the student in the text cell below the report: what was achieved, which errors remain, what must be done for the use at a real post (data of the own cameras and backgrounds, night and thermal channel, tests in the field).

**Expected result.** The text of the report with the tables.
""")

code(r"""
#@title 5.4.8 Report of the project (code shown)
from IPython.display import Markdown


def md_table(df):
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values) + "\n"


report = f'''# Mini-project: detector and tracker of drones for a counter-UAV post

Date: {time.strftime("%Y-%m-%d %H:%M")} · device: {dev} · Ultralytics {ultralytics.__version__} · aitbc {aitbc.__version__}

## 1. Data
DUT Anti-UAV (Zhao et al., 2022, Apache-2.0): training {len(DS["train"])}, validation {len(DS["val"])}, test {len(DS["test"])} images
(official parts), stored with the maximum side {MAX_SIDE} px; distances of the four closest pairs test - training by the difference hash: {RESULT_54["data"]["closest_hash_distances"]}.

## 2. Training
Start from {BASE_MODEL} (COCO), input {IMG_SIZE}, {RESULT_54["training"]["epochs"]} epochs, batch {BATCH}, frozen layers {FREEZE};
time {RESULT_54["training"]["minutes"]} min ({DEVICE}); best epoch {RESULT_54["training"]["best_epoch"]}, mAP50-95 of val {RESULT_54["training"]["val_map50_95"]}.

## 3. Test part: baseline and trained detector (working threshold chosen on val by F1)
{md_table(TAB_544.T)}
## 4. Slicing of the frame
{md_table(TAB_545.T)}
## 5. Tracking of a drone ({VIDEO}, {len(FR_V)} frames)
{md_table(TAB_546)}
## 6. ONNX
{md_table(TAB_547)}
'''
rpath = os.path.join(REPORT_DIR, "NB09_project_report.md")
open(rpath, "w", encoding="utf-8").write(report)
display(Markdown(report))
print("Report:", rpath)
update_project_json("5.4", {"lesson_5_4": RESULT_54})
""")

md(r"""
### Report of Lesson 5.4 (conclusions of the project)

Double-click this cell and write the answers.

1. Data set and its check; answer to control question 7:
2. Baseline; answer to control question 8:
3. Training: epochs, time, best epoch, curves; answer to control question 9:
4. Test part: comparison with the baseline, remaining errors; answer to control question 10:
5. Slicing; answer to control question 11:
6. Tracking of a drone; answer to control question 12:
7. ONNX and real time; answer to control question 13:
8. What must be done before the detector is used at a real post:

---
## Self-check of Topic 5

1. How does a one-stage detector YOLO give boxes, classes and confidences, and what is removed by NMS?
2. How are IoU, precision, recall, AP50 and AP50-95 computed, and which of them depends on the working threshold?
3. Why are small targets the main difficulty of the detection from a UAV and of a counter-UAV post, and which remedies were tried?
4. How is the traversable area of a ground platform obtained from a segmentation, and how is the network tested without leakage?
5. What does ByteTrack add to an IoU tracker, and how are MOTA and IDF1 computed?
6. Which stages of a video pipeline determine its frame rate, and how is real time checked?
7. Which steps turn a detector of COCO into a detector of a new military-relevant class, and how is the gain measured?

## Data sources and licences

The course library reads the files directly from the repositories of the data sets; the data are not redistributed with the notebook. When images or results are published, the data sets are cited.

1. **M3OT.** Nie Z. et al. *Scientific Data* 12, 1927 (2025) · Licence CC BY 4.0.
2. **TII-RATM.** Bosello M. et al. *IEEE Robotics and Automation Letters* 9(4), 3799–3806 (2024) · Licence CC BY 4.0.
3. **HIT-UAV.** Suo J. et al. HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection. *Scientific Data* 10, 227 (2023), https://doi.org/10.1038/s41597-023-02066-6 · Licence CC BY 4.0.
4. **DUT Anti-UAV.** Zhao J., Zhang J., Li D., Wang D. Vision-based Anti-UAV Detection and Tracking. *IEEE Transactions on Intelligent Transportation Systems* (2022); arXiv:2205.10851 · https://github.com/wangdongdut/DUT-Anti-UAV, copy https://huggingface.co/datasets/aegean-ai/dut-anti-uav · Licence Apache-2.0.
5. **YOLO11.** Jocher G., Qiu J. Ultralytics YOLO11 (2024), https://github.com/ultralytics/ultralytics · Licence AGPL-3.0; the weights trained in Lesson 5.4 are derived works under the same licence. **COCO.** Lin T.-Y. et al. *ECCV 2014*.
6. **ByteTrack.** Zhang Y. et al. *ECCV 2022*. **ONNX Runtime** (MIT).

This notebook completes the course *AI Technologies (Basic Course)*.
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
    out = sys.argv[1] if len(sys.argv) > 1 else "AITBC_NB09_YOLO_practice_and_counter_UAV_mini_project.ipynb"
    lib = sys.argv[2] if len(sys.argv) > 2 else "aitbc.py"
    nb = build(out, lib)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    print(f"written {out}: {len(nb.cells)} cells ({n_md} text, {len(nb.cells) - n_md} code)")
