# Builder of Notebook 07 of the course "AI Technologies (Basic Course)" (AIT-BC).
# Topic 4, Lessons 4.3 and 4.4 (practical classes). All exercises use real open data sets.
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
# AI Technologies (Basic Course) · Notebook 07
## Topic 4. Machine learning and neural networks in the recognition of military objects

| Lesson | Type, duration | Title |
|---|---|---|
| **4.3** | Practical class, 2 h | Classification of thermal signatures by HOG + SVM: target — background |
| **4.4** | Practical class, 2 h | Transfer learning of MobileNet for the recognition of military vehicles and export of the model to ONNX |

Course: *AI Technologies (Basic Course)*, AIT-BC · Content module 2 · Kruty Heroes Military Institute of Telecommunications and Information Technology, Kyiv, 2026.

**How to work with this notebook**

1. Run the cells from top to bottom: *Runtime → Run all*, or `Shift+Enter` cell by cell. Section 0 is always run first; after a restart of the runtime it is run again.
2. Every block begins with a text cell: the purpose, the formulas, the parameters of the form, the expected result and the control question. The answers are written in the text cell **Report** at the end of each lesson.
3. The cells marked **(code shown)** are the practical skills of the lesson: read the code, change it and run it again. The blocks of a lesson use the results of the previous blocks of the same lesson.
4. Lesson 4.3 runs on the processor in 1–3 minutes per block. Lesson 4.4 trains a neural network: with a GPU (*Runtime → Change runtime type → T4 GPU*) one epoch takes seconds, on the processor of Colab minutes; block 4.4.3 prints the estimated time after the first epoch.
5. The results are saved in the course project on Google Drive (`outputs`, `models`, `project.json`), created in Notebook 02.

**Data.** Only real images are used, with the annotation of the authors of the data sets:
- **HIT-UAV** — thermal frames of a reconnaissance UAV (DJI Matrice M210 V2, camera Zenmuse XT2, 60–130 m, camera angle 30–90°, day and night) with boxes of persons and vehicles; the flight conditions are written in the file names;
- **Military vehicles** — 9444 photographs of 24 types of military vehicles (Notebook 06), joined into 7 groups by purpose.

**Contents**

- **0. Preparation** — 0.1 runtime, course library and data sets · 0.2 course project
- **Lesson 4.3 (practical class)**
  - 4.3.1 Thermal frames and candidates of the classical detector
  - 4.3.2 HOG features of thermal signatures
  - 4.3.3 Division into the training and the test part
  - 4.3.4 Training of the SVM and the threshold of the decision
  - 4.3.5 Analysis of the errors
  - 4.3.6 Saving and checking the verifier
- **Lesson 4.4 (practical class)**
  - 4.4.1 Preparation of the data and augmentation
  - 4.4.2 Stage 1: frozen network and a new output layer
  - 4.4.3 Stage 2: fine-tuning of the last blocks
  - 4.4.4 Evaluation of the model and Grad-CAM
  - 4.4.5 Export to ONNX and INT8 quantisation
  - 4.4.6 Recognition of a new image by the ONNX model
- **Summary**: self-check of Topic 4, data sources and licences
""")

# =====================================================================================
md(r"""
---
# 0. Preparation

### Block 0.1. Runtime, course library and data sets

**Purpose.** To prepare the runtime: the libraries, the course library `aitbc` (version 2.6.0 adds the candidates of the thermal detector, square fragments and Grad-CAM) and the data sets of the lessons.

**Parameters.**
- `CACHE_ON_DRIVE` — keep the downloaded files on Google Drive (they remain after the end of the session); otherwise they are kept on the disk of the session.
- `PREFETCH_ALL` — download now the two parquet files of military vehicles (166 MB) for Lesson 4.4; the thermal frames of Lesson 4.3 (about 30 MB) are downloaded in block 4.3.1.

**Output.** A table of versions, the computing device and the size of the cache.
""")

code(r"""
#@title 0.1 Runtime, course library and data sets (run once) { display-mode: "form" }
CACHE_ON_DRIVE = False #@param {type:"boolean"}
PREFETCH_ALL = True #@param {type:"boolean"}

import sys, os, platform, time, json, math, importlib, shutil, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
import sklearn
import skimage
import torch
import torchvision
pd.set_option("display.max_colwidth", 200)

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
WORK_DIR = "/content" if IN_COLAB else os.getcwd()

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

t0 = time.time()
hit = aitbc.HITUAV()                                   # thermal frames of a reconnaissance UAV
mv = aitbc.MilitaryVehicles() if PREFETCH_ALL else None
VERSIONS = {"Python": platform.python_version(), "NumPy": np.__version__, "OpenCV": cv2.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scikit-image": skimage.__version__,
            "PyTorch": torch.__version__, "torchvision": torchvision.__version__, "aitbc": aitbc.__version__}
n_files, size_mb = aitbc.cache_size()
dev = torch.cuda.get_device_name(0) if DEVICE == "cuda" else f"CPU, {os.cpu_count()} logical cores"
display(pd.DataFrame({"Value": list(VERSIONS.values()) + ["Google Colab" if IN_COLAB else "local Jupyter", dev,
                                                           f"{aitbc.DATA_ROOT}: {n_files} files, {size_mb:.0f} MB",
                                                           f"{time.time() - t0:.0f} s"]},
                     index=list(VERSIONS) + ["environment", "computing device", "cache of the data sets", "preparation time"]))
""", form=True)

md(r"""
### Block 0.2. Course project

**Purpose.** To connect the course project created in Notebook 02. The results of this notebook are saved in it: figures and tables in `outputs`, the verifier of Lesson 4.3 and the models of Lesson 4.4 (PyTorch, ONNX) in `models` and the parameters of the lessons in `project.json`.

**Parameters.** `USE_DRIVE` — the project is on Google Drive (Colab asks for access to the Google account); otherwise it is on the disk of the session. `PROJECT_PATH` — folder of the project; the same as in Notebooks 02–06.

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
# Lesson 4.3 (Practical class). Classification of thermal signatures by HOG + SVM: target — background

**Learning questions**
1. Formation of a data set of thermal signatures: the targets and the background.
2. Computation of the HOG features of fragments of thermal frames.
3. Division of the data into the training and the test part.
4. Training of the SVM classifier and the choice of the threshold of the decision.
5. Analysis of the errors of classification.

**Task of the lesson.** At night the thermal camera of a reconnaissance UAV shows a person as a small warm spot of 10–30 pixels. The classical detector of Topic 3 (white top-hat and threshold, Notebook 05) finds most such spots, but also many warm objects that are not persons: stones and roofs heated by the day, engines, lamps, animals. In this lesson a **verifier** is built: every candidate of the detector is cut out as a fragment, described by its HOG vector and classified by an SVM into two classes, **target** (person) and **background** (false alarm). The verifier keeps the targets and rejects most false alarms.

**Result of the lesson.** The trained verifier `models/thermal_person_verifier_hog_svm.joblib`, its average precision on flight conditions not used in the training, the precision and the recall of the detector before and after the verification, the galleries of the errors in `outputs`, the parameters in `project.json`.
""")

md(r"""
## Task 4.3.1. Thermal frames and candidates of the classical detector

**Data.** The file name of a HIT-UAV frame `T_HHH_AA_W_NNNNN.jpg` gives the time of day (`T`: 0 — day, 1 — night), the flight altitude `HHH` in metres and the camera angle `AA` in degrees (90° — vertical view); `NNNNN` is the number of the frame. The frames of one **flight condition** (time of day, altitude, angle) are consecutive frames of the same flight. For every condition `FRAMES_PER_CONDITION` frames are taken at random.

**Candidates.** In every frame the warm-object detector of Topic 3 is run with the size of a person in pixels computed from the ground sampling distance of the centre row (formula of Notebook 05):
$$\text{GSD} = \frac{h}{\sin\delta}\cdot\frac{2\tan(\alpha_H/2)}{N},\qquad \delta = \text{camera angle},\ \alpha_H = 25°,\ N = 640 .$$
Every candidate (box of a warm spot) is labelled by the annotation of the authors: **target** — its centre lies in a box of a person; **background** — it lies outside all annotated boxes (false alarm); **other object** — it lies in a box of a vehicle or a bicycle (it is not used for the two classes). The fragment of a candidate is a square of 1.5 × the longer side of its box around its centre, stored as 48 × 48 px.

**Parameters.**
- `FRAMES_PER_CONDITION` — frames per flight condition (105 conditions; 6 frames — about 630 frames, 30 MB);
- `CONDITIONS` — day and night, night only or day only;
- `SHOW_FRAME` — number of the night frame shown.

**Expected result.** The table of the candidates by the time of day, the recall and the precision of the detector, a night frame with the candidates coloured by their label.

**What to observe.** In 630 frames of 105 conditions the detector gives 10 258 candidates. At night it finds 64 % of the persons (848 of 1319) with a precision of 0.18, i.e. about 4.7 false alarms for every found person; by day it finds only 24 % with a precision of 0.08: the ground heated by the sun gives many warm spots, and persons are less contrasting (Notebook 05). About one candidate in five lies on a vehicle or a bicycle (other object). The fragments of persons are small vertical warm spots; the false alarms are parts of roofs, edges of roads, lamps and spots of heated ground.

**Control question 1.** Why is the label of a candidate taken from the annotation and not given by hand, and which error of the annotation becomes an error of the data set of candidates?
""")

code(r"""
#@title 4.3.1 Frames of flight conditions and candidates of the detector (code shown)
FRAMES_PER_CONDITION = 6 #@param {type:"slider", min:2, max:15, step:1}
CONDITIONS = "day and night" #@param ["day and night", "night only", "day only"]
SHOW_FRAME = 0 #@param {type:"integer"}

frames = hit.images.copy()
frames["condition"] = np.where(frames.night, "night", "day") + " " + frames.altitude.astype(str) + " m " + frames.angle.astype(str) + "°"
if CONDITIONS != "day and night":
    frames = frames[frames.night == (CONDITIONS == "night only")]
FR = frames.sample(frac=1.0, random_state=0).groupby("condition").head(FRAMES_PER_CONDITION).sort_values("file").reset_index(drop=True)
t0 = time.time()
hit.prefetch(FR.file.tolist())
t_load = time.time() - t0
rows = []
t0 = time.time()
for r in FR.itertuples():
    g, objs = hit.load(r.file)
    gsd = float(aitbc.oblique_gsd(r.altitude, r.angle, [g.shape[0] / 2])[0][0])     # GSD of the centre row, m/px
    for c in aitbc.thermal_candidates(g, objs, gsd, size=48):                         # detector of Topic 3 + labels
        rows.append(dict(file=r.file, condition=r.condition, night=r.night, altitude=r.altitude, angle=r.angle,
                         split_authors=r.split, **c))
CAND = pd.DataFrame(rows)
t_cand = time.time() - t0
tab = CAND.pivot_table(index="night", columns="label", values="side", aggfunc="count", fill_value=0)
tab.index = ["day" if not v else "night" for v in tab.index]
tab["persons in the frames"] = [int(FR[FR.night == (k == "night")].Person.sum()) for k in tab.index]
tab["recall of the detector"] = (tab["target"] / tab["persons in the frames"]).round(3)
tab["precision of the detector"] = (tab["target"] / (tab["target"] + tab["background"])).round(3)
display(tab.style.set_caption(f"{len(FR)} frames of {FR.condition.nunique()} flight conditions: {len(CAND)} candidates "
                              f"(download {t_load:.0f} s, detection {t_cand:.0f} s)"))
night_files = FR[FR.night].file.tolist() or FR.file.tolist()
f_show = night_files[SHOW_FRAME % len(night_files)]
g, objs = hit.load(f_show)
vis = aitbc.to_rgb(g).copy()
for o in objs:
    x1, y1, x2, y2 = [int(v) for v in o["box"]]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
COL = {"target": (0, 230, 0), "background": (255, 40, 40), "other object": (255, 200, 0)}
for c in CAND[CAND.file == f_show].itertuples():
    x1, y1, x2, y2 = [int(v) for v in c.box]
    cv2.rectangle(vis, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), COL[c.label], 2)
fig, ax = plt.subplots(1, 2, figsize=(17, 6.4), gridspec_kw={"width_ratios": [1.4, 1]})
ax[0].imshow(vis)
ax[0].set_title(f"{f_show}: annotation white; candidates: target green, false alarm red, other object yellow", fontsize=9)
ax[0].axis("off")
show = pd.concat([CAND[CAND.label == "target"].sample(min(8, (CAND.label == "target").sum()), random_state=1),
                  CAND[CAND.label == "background"].sample(min(8, (CAND.label == "background").sum()), random_state=1)])
tiles = np.vstack([np.hstack([np.pad(c, 2, constant_values=255) for c in show.chip.values[k * 8:(k + 1) * 8]]) for k in range(2)])
ax[1].imshow(tiles, cmap="gray")
ax[1].set_title("fragments 48 × 48: upper row targets (persons), lower row false alarms", fontsize=9)
ax[1].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson43_candidates.png"), dpi=80)
plt.show()
""")

md(r"""
## Task 4.3.2. HOG features of thermal signatures

**Method.** The fragment is reduced to $S \times S$ pixels and its HOG vector is computed (Lesson 4.1): gradient, histograms of orientations in cells of $c \times c$ pixels, normalisation in blocks of 2 × 2 cells. The length of the vector is
$$N = \left(\frac{S}{c} - 1\right)^2 \cdot 4 \cdot n_o;\qquad S = 32,\ c = 4,\ n_o = 9:\ N = 7^2 \cdot 4 \cdot 9 = 1764 .$$
A small cell is needed: a person of 10–30 px occupies only a few cells of 4 × 4 px of a fragment of 32 × 32 px.

**Parameters.** `HOG_SIZE` — $S$; `CELL` — $c$; `ORIENTATIONS` — $n_o$.

**Expected result.** The fragments of targets and false alarms with their HOG pictures, the mean HOG picture of both classes and the length of the vector.

**What to observe.** A person at night gives a compact bright spot with gradients of all orientations around it (a "ring" of the HOG); a false alarm is more often a part of a straight edge (a roof, a road, a wall) with one dominating orientation, or a spot without a clear shape. The mean pictures of the classes differ in the centre of the fragment.

**Control question 2.** What is the length of the HOG vector for $S = 24$, $c = 4$, $n_o = 9$, and why does a cell of 8 × 8 px describe a person of 12 px poorly?
""")

code(r"""
#@title 4.3.2 HOG of thermal fragments (code shown)
HOG_SIZE = 32 #@param [24, 32, 48] {type:"raw"}
CELL = 4 #@param [4, 8] {type:"raw"}
ORIENTATIONS = 9 #@param [6, 9, 12] {type:"raw"}

from skimage.feature import hog
t0 = time.time()
small = [cv2.resize(c, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA) for c in CAND.chip]
XH = aitbc.hog_features(small, size=HOG_SIZE, cell=CELL, orientations=ORIENTATIONS)     # N x D
t_hog = time.time() - t0
print(f"HOG of {len(XH)} fragments: {XH.shape[1]} values each (formula: {aitbc.hog_length(HOG_SIZE, CELL, 2, ORIENTATIONS)}), {t_hog:.1f} s")
fig, ax = plt.subplots(3, 8, figsize=(17, 6.6))
for j, lab in enumerate(["target", "background"]):
    idx = CAND.index[CAND.label == lab][:4]
    for k, i in enumerate(idx):
        _, hi = hog(small[i], orientations=ORIENTATIONS, pixels_per_cell=(CELL, CELL), cells_per_block=(2, 2), visualize=True)
        ax[0, 4 * j + k].imshow(small[i], cmap="gray")
        ax[0, 4 * j + k].set_title(lab if lab == "target" else "false alarm", fontsize=8.5)
        ax[1, 4 * j + k].imshow(hi, cmap="gray")
for j, lab in enumerate(["target", "background"]):
    m_img = np.mean([hog(small[i], orientations=ORIENTATIONS, pixels_per_cell=(CELL, CELL), cells_per_block=(2, 2), visualize=True)[1]
                     for i in CAND.index[CAND.label == lab][:300]], axis=0)
    ax[2, 4 * j].imshow(np.mean([small[i] for i in CAND.index[CAND.label == lab][:300]], axis=0), cmap="gray")
    ax[2, 4 * j].set_title("mean fragment", fontsize=8.5)
    ax[2, 4 * j + 1].imshow(m_img, cmap="gray")
    ax[2, 4 * j + 1].set_title("mean HOG picture", fontsize=8.5)
for a in ax.ravel():
    a.axis("off")
fig.suptitle("left: targets (persons); right: false alarms of the detector", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson43_hog.png"), dpi=80)
plt.show()
""")

md(r"""
## Task 4.3.3. Division into the training and the test part

**Method.** The candidates of the classes *target* and *background* form the data set of the verifier: $\mathbf x_i$ — HOG vector, $y_i = 1$ for a target and $y_i = 0$ for the background. The quality must be estimated on frames of flights that the classifier has not seen (Lesson 4.1, block 4.1.6). Two divisions are compared:
- **by flight condition** — a random share `TEST_SHARE` of the conditions (time of day, altitude, angle) goes completely into the test part (`GroupShuffleSplit`, the condition is the group);
- **official split of the authors** — the parts `train`/`val` and `test` of HIT-UAV, formed by random frames: the neighbouring frames of the same flight fall into different parts.

**Parameters.** `SPLIT`; `TEST_SHARE` — share of the conditions in the test part.

**Expected result.** The table of the two parts (frames, conditions, targets, false alarms), the number of conditions present in both parts, and the share of the test frames of the official split whose neighbouring frame (number ±1) is in the training part.

**What to observe.** With the division by flight condition the 26 test conditions (149 frames) do not occur in the training part (78 conditions, 431 frames). In the official split of the authors 33.7 % of the 579 test frames have the neighbouring frame of the same flight in the training part. About one candidate in seven is a target (0.13–0.14). In the check of the course the average precision of the verifier was not higher with the official split than with the division by condition (the test parts differ), because the verifier classifies small fragments of separate frames; the division by flight condition is still the one that estimates the work on a new flight.

**Control question 3.** Why is the share of targets among the candidates (about one in seven) important for the training and for the estimation of the classifier?
""")

code(r"""
#@title 4.3.3 Training and test part (code shown)
SPLIT = "by flight condition" #@param ["by flight condition", "official split of the authors (random frames)"]
TEST_SHARE = 0.25 #@param {type:"slider", min:0.15, max:0.4, step:0.05}

from sklearn.model_selection import GroupShuffleSplit
use = CAND.label.isin(["target", "background"]).values
D = CAND[use].reset_index(drop=True)
X = XH[use]
y = (D.label == "target").astype(int).values
if SPLIT.startswith("by flight"):
    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=TEST_SHARE, random_state=0).split(X, y, groups=D.condition))
else:
    tr_idx, te_idx = np.where(D.split_authors != "test")[0], np.where(D.split_authors == "test")[0]
parts = {}
for name, ii in (("training", tr_idx), ("test", te_idx)):
    parts[name] = {"frames": D.file.values[ii].tolist(), "conditions": set(D.condition.values[ii]),
                   "targets": int(y[ii].sum()), "false alarms": int((1 - y[ii]).sum()),
                   "night candidates": int(D.night.values[ii].sum())}
display(pd.DataFrame({k: {"frames": len(set(v["frames"])), "flight conditions": len(v["conditions"]), "targets": v["targets"],
                          "false alarms": v["false alarms"], "share of targets": round(v["targets"] / (v["targets"] + v["false alarms"]), 3),
                          "candidates at night": v["night candidates"]} for k, v in parts.items()}).T)
both = parts["training"]["conditions"] & parts["test"]["conditions"]
print(f"Flight conditions present in both parts: {len(both)} of {D.condition.nunique()}")
# neighbouring frames in the official split of the authors (all 2898 frames of HIT-UAV)
allf = hit.images.copy()
allf["num"] = allf.file.str.extract(r"_(\d+)\.jpg$")[0].astype(int)
allf["cond"] = allf.file.str.rsplit("_", n=1).str[0]
train_keys = set(zip(allf.cond[allf.split != "test"], allf.num[allf.split != "test"]))
te_all = allf[allf.split == "test"]
nb = np.mean([((c, n - 1) in train_keys) or ((c, n + 1) in train_keys) for c, n in zip(te_all.cond, te_all.num)])
print(f"Official split: {100 * nb:.1f} % of the {len(te_all)} test frames have a neighbouring frame (number ± 1) in the training part")
""")
md(r"""
## Task 4.3.4. Training of the SVM and the threshold of the decision

**Method.** The SVM computes for a candidate the value of the decision function $f(\mathbf x)$ (Lesson 4.1): $f > 0$ — target, $f < 0$ — background; $|f|$ grows with the distance from the boundary. With unequal classes (about one target for seven false alarms) the weights of the errors are equalised by `class_weight="balanced"`: an error on a class of $n_k$ examples costs $n / (2 n_k)$. The decision is taken with a **threshold** $\tau$: the candidate is accepted when $f(\mathbf x) > \tau$. Moving $\tau$ exchanges recall for precision; the curve *precision — recall* over all thresholds and its area, the **average precision** (AP), describe the verifier independently of the threshold:
$$\text{AP} = \sum_k (R_k - R_{k-1})\, P_k ,$$
where $P_k, R_k$ are the precision and the recall at the $k$-th threshold. A random verifier has AP equal to the share of targets.

**Parameters.**
- `KERNEL` — RBF or linear SVM; `C`; `BALANCED` — equalise the classes;
- `SELECT_C` — choose $C$ by cross-validation with 3 folds of flight conditions inside the training part (4 values of $C$, 4–5 minutes on a processor);
- `THRESHOLD` — $\tau$.

**Expected result.** The time of the training, the number of support vectors, the AP at night and by day, the curves precision — recall with the point of the chosen threshold, the precision, recall and F1 at the threshold and the confusion matrix.

**What to observe.** The SVM with the RBF kernel is trained in about 20 s on about 6000 candidates and keeps about half of them (3074) as support vectors. The AP at night is 0.82 against 0.14 for a random verifier; by day only 0.43. At τ = 0 the precision is 0.85 and the recall 0.50: half of the persons are rejected. To keep 90 % of the targets the threshold must be lowered to about τ = −0.8, where the precision is 0.35: about two false alarms for every accepted person instead of about six among the candidates of the detector. The choice of C by cross-validation (`SELECT_C`, about 4–5 minutes) gives AP 0.61 for C = 0.1 and 0.70 for C = 1–100.

**Control question 4.** The verifier must not lose more than 10 % of the persons found by the detector. How is the threshold chosen, and on which part of the data?
""")

code(r"""
#@title 4.3.4 SVM verifier: training, precision-recall curve, threshold (code shown)
KERNEL = "RBF" #@param ["RBF", "linear"]
C = 10 #@param [0.1, 1, 10, 100] {type:"raw"}
BALANCED = True #@param {type:"boolean"}
SELECT_C = False #@param {type:"boolean"}
THRESHOLD = 0.0 #@param {type:"slider", min:-2, max:2, step:0.25}

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, precision_recall_fscore_support


def make_svm(c, n):
    cw = "balanced" if BALANCED else None
    if KERNEL == "RBF":
        return make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", class_weight=cw))
    return make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", alpha=1.0 / (c * n), class_weight=cw, max_iter=200, tol=1e-4, random_state=0))


Xtr, ytr, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
if SELECT_C:
    cv_rows = []
    for c in (0.1, 1, 10, 100):
        aps = []
        for a, b in GroupKFold(3).split(Xtr, ytr, D.condition.values[tr_idx]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_ = make_svm(c, len(a)).fit(Xtr[a], ytr[a])
            aps.append(average_precision_score(ytr[b], m_.decision_function(Xtr[b])))
        cv_rows.append({"C": c, "AP (cross-validation)": np.mean(aps), "±": np.std(aps)})
    cv_tab = pd.DataFrame(cv_rows).set_index("C")
    display(cv_tab.style.format(precision=3).set_caption("Choice of C on the training part (3 folds of flight conditions)"))
    C = float(cv_tab["AP (cross-validation)"].idxmax())
    print("Chosen C =", C)
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    VERIFIER = make_svm(C, len(Xtr)).fit(Xtr, ytr)
t_fit = time.time() - t0
s_te = VERIFIER.decision_function(Xte)
night_te = D.night.values[te_idx]
pred = (s_te > THRESHOLD).astype(int)
P_, R_, F_, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
rows = {"training candidates (targets / false alarms)": f"{len(ytr)} ({int(ytr.sum())} / {int(len(ytr) - ytr.sum())})",
        "time of the training": f"{t_fit:.1f} s",
        "support vectors": int(VERIFIER[-1].n_support_.sum()) if KERNEL == "RBF" else "— (linear model)",
        "AP at night / by day / all": " / ".join(f"{average_precision_score(yte[m], s_te[m]):.3f}" if yte[m].sum() else "—"
                                              for m in (night_te, ~night_te, np.ones(len(yte), bool))),
        "AP of a random verifier (share of targets)": f"{yte.mean():.3f}",
        f"precision / recall / F1 at τ = {THRESHOLD:+.2f}": f"{P_:.3f} / {R_:.3f} / {F_:.3f}"}
pc, rc, thc = precision_recall_curve(yte, s_te)
k90 = int(np.where(rc[:-1] >= 0.9)[0].max())            # largest threshold that keeps 90 % of the targets
rows["τ keeping 90 % of the targets (illustration on the test part): τ, precision"] = f"{thc[k90]:+.2f}, {pc[k90]:.3f}"
display(pd.DataFrame({"value": list(rows.values())}, index=list(rows)))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
for m, lab, col in ((night_te, "night", "#1565c0"), (~night_te, "day", "#ef6c00")):
    if yte[m].sum():
        p, r, th = precision_recall_curve(yte[m], s_te[m])
        ax[0].plot(r, p, color=col, label=f"{lab}: AP {average_precision_score(yte[m], s_te[m]):.3f}")
        k = np.searchsorted(th, THRESHOLD)
        if k < len(th):
            ax[0].plot(r[k], p[k], "o", color=col)
ax[0].set_xlabel("recall")
ax[0].set_ylabel("precision")
ax[0].set_title(f"verifier HOG + {KERNEL} SVM (C = {C:g}); points: τ = {THRESHOLD:+.2f}", fontsize=9)
ax[0].grid(alpha=0.3)
ax[0].legend()
cm = confusion_matrix(yte, pred)
ax[1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=12, color="w" if cm[i, j] > cm.max() / 2 else "k")
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["background", "target"])
ax[1].set_yticks([0, 1])
ax[1].set_yticklabels(["false alarm", "person"])
ax[1].set_xlabel("decision of the verifier")
ax[1].set_ylabel("label of the candidate")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson43_precision_recall.png"), dpi=80)
plt.show()
""")

md(r"""
## Task 4.3.5. Analysis of the errors

**Method.** At the threshold $\tau$ two kinds of errors are studied on the test part: **missed targets** (persons rejected by the verifier, false negatives) and **accepted false alarms** (false positives). The errors are grouped by the time of day, the flight altitude, the camera angle and the size of the candidate. Then the verifier is put after the detector: in every test frame the detector gives its candidates and the verifier keeps those with $f > \tau$; the recall, the precision and the number of false alarms per frame of the whole pipeline are compared with the detector alone.

**Parameters.** `SHOW_TEST_FRAME` — number of the night test frame shown with the decisions.

**Expected result.** The galleries of the missed targets and of the accepted false alarms, the table of the errors by groups, the table *detector — detector + verifier* for night and day, a test frame with the kept (green) and rejected (red) candidates and the persons missed by the pipeline (yellow).

**What to observe.** At τ = 0 the verifier rejects almost all false alarms: at night their number falls from 12.9 to 0.17 per frame and the precision of the pipeline rises from 0.20 to 0.91, but the recall falls from 0.66 to 0.37. With τ = −0.5 (official split of the authors, in the check of the course) the night recall falls only from 0.72 to 0.61 at a precision of 0.55 and 1.8 false alarms per frame instead of 13.2. Small candidates (up to 10 px) are rejected most often (recall 0.32 against 0.62 for 10–20 px): a fragment of a few pixels has almost no shape. By day the recall of the pipeline is below 0.1. Some accepted false alarms look like persons: they may be persons missing in the annotation.

**Control question 5.** Which errors of the verifier are caused by the data (small size, day, annotation) and which by the classifier, and how can each kind be reduced?
""")

code(r"""
#@title 4.3.5 Errors of the verifier and the effect on the whole detector (code shown)
SHOW_TEST_FRAME = 0 #@param {type:"integer"}

Dte = D.iloc[te_idx].reset_index(drop=True)
Dte["score"], Dte["accepted"] = s_te, s_te > THRESHOLD
fn = Dte[(Dte.label == "target") & ~Dte.accepted].sort_values("score")
fp = Dte[(Dte.label == "background") & Dte.accepted].sort_values("score", ascending=False)
fig, ax = plt.subplots(2, 8, figsize=(17, 5.2))
for row, (tab_, name) in enumerate(((fn, "missed person"), (fp, "accepted false alarm"))):
    for k in range(8):
        a = ax[row, k]
        a.axis("off")
        if k < len(tab_):
            r = tab_.iloc[k]
            a.imshow(r.chip, cmap="gray")
            a.set_title(f"{name}\n{r.condition}, f = {r.score:+.2f}", fontsize=7.5)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson43_errors.png"), dpi=80)
plt.show()
Dte["size, px"] = pd.cut(Dte.side, [0, 10, 20, 40, 1000], labels=["≤ 10", "10–20", "20–40", "> 40"])
Dte["time of day"] = np.where(Dte.night, "night", "day")
Dte["altitude, m"] = pd.cut(Dte.altitude, [0, 80, 110, 200], labels=["60–80", "90–110", "120–130"])


def err_table(by):
    g_ = Dte.groupby(by, observed=True)
    return pd.DataFrame({"targets": g_.apply(lambda t: int((t.label == "target").sum())),
                         "recall of the verifier": g_.apply(lambda t: ((t.label == "target") & t.accepted).sum() / max(1, (t.label == "target").sum())),
                         "false alarms": g_.apply(lambda t: int((t.label == "background").sum())),
                         "accepted false alarms, %": g_.apply(lambda t: 100 * ((t.label == "background") & t.accepted).sum() / max(1, (t.label == "background").sum()))})


for by in ("time of day", "altitude, m", "size, px"):
    display(err_table(by).style.format({"recall of the verifier": "{:.3f}", "accepted false alarms, %": "{:.1f}"}).set_caption(f"Errors by {by}"))
# the whole pipeline on the test frames: detector alone and detector + verifier
te_frames = sorted(set(Dte.file))
ff = FR.set_index("file").loc[te_frames]
pipe = []
for night in (True, False):
    fs = ff.index[ff.night == night]
    if len(fs) == 0:
        continue
    t_ = Dte[Dte.file.isin(fs)]
    n_p = int(ff.loc[fs].Person.sum())
    tp0, fp0 = int((t_.label == "target").sum()), int((t_.label == "background").sum())
    tp1, fp1 = int(((t_.label == "target") & t_.accepted).sum()), int(((t_.label == "background") & t_.accepted).sum())
    pipe.append({"test frames": f"{'night' if night else 'day'}: {len(fs)}", "persons": n_p,
                 "detector: recall": tp0 / max(n_p, 1), "detector: precision": tp0 / max(tp0 + fp0, 1), "detector: false alarms per frame": fp0 / len(fs),
                 "+ verifier: recall": tp1 / max(n_p, 1), "+ verifier: precision": tp1 / max(tp1 + fp1, 1), "+ verifier: false alarms per frame": fp1 / len(fs)})
PIPE = pd.DataFrame(pipe).set_index("test frames")
display(PIPE.style.format(precision=3).set_caption(f"Detector of Topic 3 and detector + verifier, τ = {THRESHOLD:+.2f}"))
nf = [f for f in te_frames if ff.loc[f, "night"]] or te_frames
f_show = nf[SHOW_TEST_FRAME % len(nf)]
g, objs = hit.load(f_show)
vis = aitbc.to_rgb(g).copy()
kept_boxes = []
for c in Dte[Dte.file == f_show].itertuples():
    x1, y1, x2, y2 = [int(v) for v in c.box]
    cv2.rectangle(vis, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 230, 0) if c.accepted else (255, 40, 40), 2)
    if c.accepted:
        kept_boxes.append(c.box)
m_ = aitbc.match_detections(np.array(kept_boxes).reshape(-1, 4), objs, {"Person"})
for o in m_["missed"]:
    x1, y1, x2, y2 = [int(v) for v in o["box"]]
    cv2.rectangle(vis, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 230, 0), 1)
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(vis)
ax.set_title(f"{f_show}: kept green, rejected red, persons missed by the pipeline yellow", fontsize=9)
ax.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson43_pipeline_frame.png"), dpi=80)
plt.show()
PIPE.to_csv(os.path.join(REPORT_DIR, "lesson43_detector_and_verifier.csv"))
""")

md(r"""
## Task 4.3.6. Saving and checking the verifier

**Method.** The trained verifier is a chain *standardisation → SVM*. To be used on another computer it is saved with everything that defines its input: the size of the fragment and its scale relative to the box of the candidate, the parameters of the HOG, the threshold $\tau$ and the classes. The file is loaded again and its decisions on the test part are compared with those of the model in memory.

**Expected result.** The file `models/thermal_person_verifier_hog_svm.joblib` with its size, the agreement of the decisions of the loaded model (must be 1.000) and the record in `project.json`. The file of the RBF SVM is large (about 40 MB): it keeps all support vectors (about 3000 vectors of 1764 values), which are needed to compute $f(\mathbf x)$; the linear SVM keeps one vector of weights (a few kB).

**Control question 6.** The onboard computer computes the HOG with a cell of 8 px instead of 4 px. What happens with the decisions of the saved verifier, and how is such an error prevented?
""")

code(r"""
#@title 4.3.6 Saving the verifier (code shown)
import joblib
path = os.path.join(MODEL_DIR, "thermal_person_verifier_hog_svm.joblib")
bundle = {"model": VERIFIER, "threshold": float(THRESHOLD), "classes": ["background", "target (person)"],
          "fragment": {"scale_of_box": 1.5, "stored_size_px": 48, "resized_to_px": HOG_SIZE, "palette": "white-hot"},
          "hog": {"orientations": ORIENTATIONS, "pixels_per_cell": CELL, "cells_per_block": 2, "block_norm": "L2-Hys"},
          "detector": "aitbc.detect_warm_objects (Topic 3), GSD of the centre row",
          "data": "HIT-UAV (Suo et al., Scientific Data 10, 227, 2023), " + SPLIT,
          "kernel": KERNEL, "C": float(C), "balanced": BALANCED,
          "AP_test": float(average_precision_score(yte, s_te))}
joblib.dump(bundle, path)
loaded = joblib.load(path)
agree = float(np.mean((loaded["model"].decision_function(Xte) > loaded["threshold"]) == (s_te > THRESHOLD)))
print(f"Saved: {path} ({os.path.getsize(path) / 2 ** 20:.2f} MB); agreement of the decisions of the loaded verifier: {agree:.3f}")
update_project_json("4.3", {"lesson_4_3": {k: v for k, v in bundle.items() if k != "model"} | {
    "model_file": os.path.relpath(path, PROJECT_ROOT), "frames_per_condition": FRAMES_PER_CONDITION, "conditions": CONDITIONS,
    "pipeline": PIPE.round(3).reset_index().to_dict("records")}})
""")

md(r"""
### Report of Lesson 4.3

Double-click this cell and write the answers.

1. Frames, flight conditions and candidates; recall and precision of the detector at night and by day; answer to control question 1:
2. Parameters of the HOG and the length of the vector; differences of the HOG of targets and false alarms; answer to control question 2:
3. Division into the parts; conditions in both parts; neighbouring frames of the official split; answer to control question 3:
4. Kernel, C, AP at night and by day, the chosen threshold and its precision and recall; answer to control question 4:
5. Errors by the time of day, altitude and size; the table detector — detector + verifier; answer to control question 5:
6. File of the verifier and the check after loading; answer to control question 6:
""")
# =====================================================================================
md(r"""
---
# Lesson 4.4 (Practical class). Transfer learning of MobileNet for the recognition of military vehicles and export to ONNX

**Learning questions**
1. Preparation of the data for the training of a neural network: training, validation and test parts, augmentation.
2. Transfer learning: a frozen network trained on ImageNet and a new output layer.
3. Fine-tuning of the last blocks of the network.
4. Evaluation of the model and visualisation of the regions that define its decision (Grad-CAM).
5. Export of the model to ONNX and the check of the inference for the onboard computer.

**Task of the lesson.** MobileNetV3-small, trained on 1.2 million images of ImageNet, is adapted to the recognition of the 7 groups of military vehicles of the course (tank, infantry fighting vehicle, armoured personnel carrier and scout car, artillery, multiple rocket launcher, air defence and radar, missile system) in two stages: first a new output layer is trained on the frozen network, then the last blocks of the network are trained together with it with a small learning rate. In Lesson 4.2 the same frozen network with a linear layer reached an accuracy of 0.63 on the 7 groups (block 4.2.5); the lesson measures what the augmentation and the fine-tuning add.

**Result of the lesson.** The model `models/military_vehicles_mobilenetv3_finetuned.pt` and its ONNX file with the file of labels, the curves of the training, the metrics on the test part and the Grad-CAM pictures in `outputs`, the parameters in `project.json`.
""")

md(r"""
## Task 4.4.1. Preparation of the data and augmentation

**Parts of the data.** The cleaned training part of block 4.1.2 (without repeated copies) gives at most `N_PER_CLASS_44` images of every type; a share `VAL_SHARE` of them (stratified by class) forms the **validation part**, used to choose the number of epochs and to compare the stages. The **test part** is the cleaned test part of Lesson 4.1 (without copies and near duplicates of training images, 1502 images) and is used once, in block 4.4.4.

**Augmentation.** In every epoch every training image is changed at random: a crop of 60–100 % of its area resized to $S \times S$ (`RandomResizedCrop`), a horizontal flip, changes of brightness, contrast and saturation by up to ±30 % (`ColorJitter`). The network sees a new variant of the image in every epoch, which reduces the fitting to the training images. The validation and test images are only resized to $S \times S$. All images are normalised by the mean and the standard deviation of ImageNet:
$$x' = \frac{x/255 - \mu_c}{\sigma_c},\qquad \mu = (0.485,\ 0.456,\ 0.406),\ \sigma = (0.229,\ 0.224,\ 0.225).$$

**Parameters.** `LEVEL_44`; `N_PER_CLASS_44`; `VAL_SHARE`; `INPUT_SIZE` — $S$ (160 px is about twice faster than 224 px); `AUGMENT`.

**Expected result.** The table of the parts and eight augmented variants of one training image.

**Control question 7.** Why must the augmentation not be applied to the validation and test images, and which augmentation would be wrong for thermal frames or for frames of a vertical view?
""")

code(r"""
#@title 4.4.1 Parts of the data and augmentation (code shown)
LEVEL_44 = "7 groups of the course" #@param ["7 groups of the course", "24 types"]
N_PER_CLASS_44 = 150 #@param {type:"slider", min:50, max:400, step:25}
VAL_SHARE = 0.15 #@param {type:"slider", min:0.1, max:0.3, step:0.05}
INPUT_SIZE = 160 #@param [128, 160, 224] {type:"raw"}
AUGMENT = True #@param {type:"boolean"}

import torch.nn as nn
from torchvision.transforms import v2 as T
from sklearn.model_selection import train_test_split
if mv is None:
    mv = aitbc.MilitaryVehicles()
TRAIN_ALL, TEST_IDX = mv.clean(near=2)
col44 = "cls" if LEVEL_44 == "24 types" else "group"
LABELS = sorted(mv.table.cls.unique()) if col44 == "cls" else list(aitbc.MilitaryVehicles.GROUP_ORDER)
sub = mv.subset(TRAIN_ALL, N_PER_CLASS_44, seed=0)
TR_IDX, VA_IDX = train_test_split(sub, test_size=VAL_SHARE, random_state=0, stratify=mv.table[col44].values[sub])
label_of = lambda idx: torch.tensor([LABELS.index(v) for v in mv.table[col44].values[idx]])
IMG = {}                                           # decoded images, resized once to 1.15 S (for the random crops)


def image_for(i, s):
    if (i, s) not in IMG:
        IMG[(i, s)] = mv.image(i, size=s)
    return IMG[(i, s)]


AUG = T.Compose([T.ToImage(), T.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0), antialias=True), T.RandomHorizontalFlip(),
                 T.ColorJitter(0.3, 0.3, 0.3), T.ToDtype(torch.float32, scale=True),
                 T.Normalize(aitbc.IMAGENET_MEAN.tolist(), aitbc.IMAGENET_STD.tolist())])


def batches(idx, train, bs=64):
    # mini-batches (x, y): augmented training images or resized validation / test images
    for b in range(0, len(idx), bs):
        ii = idx[b:b + bs]
        if train and AUGMENT:
            x = torch.stack([AUG(image_for(i, int(INPUT_SIZE * 1.15))) for i in ii])
        else:
            x = torch.from_numpy(aitbc.imagenet_batch([image_for(i, INPUT_SIZE) for i in ii], INPUT_SIZE))
        yield x.to(DEVICE), label_of(ii).to(DEVICE)


parts44 = pd.DataFrame({"images": [len(TR_IDX), len(VA_IDX), len(TEST_IDX)],
                        "classes": [len(set(mv.table[col44].values[p])) for p in (TR_IDX, VA_IDX, TEST_IDX)]},
                       index=["training (augmented)" if AUGMENT else "training", "validation", "test (cleaned, block 4.1.2)"])
display(parts44.style.set_caption(f"{LEVEL_44}: {len(LABELS)} classes, input {INPUT_SIZE} × {INPUT_SIZE} px"))
i0 = TR_IDX[0]
fig, ax = plt.subplots(1, 8, figsize=(18, 2.6))
ax[0].imshow(image_for(i0, INPUT_SIZE))
ax[0].set_title(f"original: {mv.table[col44].values[i0]}", fontsize=8.5)
for k in range(1, 8):
    t_ = AUG(image_for(i0, int(INPUT_SIZE * 1.15))).numpy().transpose(1, 2, 0) * aitbc.IMAGENET_STD + aitbc.IMAGENET_MEAN
    ax[k].imshow(np.clip(t_, 0, 1))
    ax[k].set_title(f"augmented {k}", fontsize=8.5)
for a in ax:
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson44_augmentation.png"), dpi=80)
plt.show()
""")

md(r"""
## Task 4.4.2. Stage 1: frozen network and a new output layer

**Method.** The output layer of MobileNetV3-small for the 1000 classes of ImageNet is replaced by a new one: dropout (a random 20 % of the 576 features are set to zero during the training) and a linear layer of $576 \cdot K + K$ parameters ($K$ — number of classes) with softmax. The convolutional part is frozen: its features are computed once for every training and validation image (without augmentation), after which an epoch of the training of the new layer takes a fraction of a second. The loss is the cross-entropy with label smoothing 0.1 (the target probability of the true class is 0.9 + 0.1/K instead of 1), the optimiser is AdamW.

**Parameters.** `EPOCHS_1` — epochs of the training of the new layer; `LR_HEAD` — its learning rate.

**Expected result.** The curves of the loss on the training part and of the accuracy on the validation part, the number of trained parameters, the accuracy of stage 1 on the validation part.

**What to observe.** Stage 1 trains 4039 parameters (576 · 7 + 7) of 2.5 million. The features of 3569 training and validation images are computed in 10–20 s on a processor, 30 epochs of the new layer take about 2 s. The validation accuracy reaches about 0.60 at 160 px and 0.64 at 224 px.

**Control question 8.** How many parameters are trained at stage 1 for the 7 groups, and why can they be trained on the features computed once?
""")

code(r"""
#@title 4.4.2 Stage 1: new output layer on the frozen network (code shown)
EPOCHS_1 = 30 #@param {type:"slider", min:5, max:60, step:5}
LR_HEAD = 0.001 #@param [0.0003, 0.001, 0.003] {type:"raw"}

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import accuracy_score, f1_score
torch.manual_seed(0)
NET = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
NET.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(576, len(LABELS)))
NET = NET.to(DEVICE).eval()
for p in NET.features.parameters():
    p.requires_grad = False


def features_of(idx):
    with torch.no_grad():
        return torch.cat([NET.avgpool(NET.features(x)).flatten(1) for x, _ in batches(idx, train=False)])


t0 = time.time()
F_tr, F_va = features_of(TR_IDX), features_of(VA_IDX)
y_tr, y_va = label_of(TR_IDX).to(DEVICE), label_of(VA_IDX).to(DEVICE)
t_feat = time.time() - t0
opt = torch.optim.AdamW(NET.classifier.parameters(), lr=LR_HEAD, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
hist1 = []
t0 = time.time()
for ep in range(EPOCHS_1):
    NET.classifier.train()
    perm = torch.randperm(len(F_tr), device=DEVICE)
    tot = 0.0
    for b in range(0, len(F_tr), 64):
        ii = perm[b:b + 64]
        opt.zero_grad()
        loss = lossf(NET.classifier(F_tr[ii]), y_tr[ii])
        loss.backward()
        opt.step()
        tot += loss.item() * len(ii)
    NET.classifier.eval()
    with torch.no_grad():
        acc_va = (NET.classifier(F_va).argmax(1) == y_va).float().mean().item()
    hist1.append((tot / len(F_tr), acc_va))
t_head = time.time() - t0
HEAD_STAGE1 = {k: v.detach().clone() for k, v in NET.classifier.state_dict().items()}
h1 = np.array(hist1)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ax[0].plot(range(1, EPOCHS_1 + 1), h1[:, 0])
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("loss, training part")
ax[1].plot(range(1, EPOCHS_1 + 1), h1[:, 1], color="#2e7d32")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("accuracy, validation part")
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.show()
n_head = sum(p.numel() for p in NET.classifier.parameters())
print(f"Features of {len(TR_IDX) + len(VA_IDX)} images: {t_feat:.0f} s on {DEVICE}; {EPOCHS_1} epochs of the new layer: {t_head:.1f} s; "
      f"trained parameters: {n_head} (576 · {len(LABELS)} + {len(LABELS)}); accuracy on the validation part: {h1[-1, 1]:.3f}")
STAGE1_VAL = float(h1[-1, 1])
""")

md(r"""
## Task 4.4.3. Stage 2: fine-tuning of the last blocks

**Method.** The last `UNFREEZE_BLOCKS` blocks of the convolutional part are unfrozen and trained together with the output layer of stage 1 on augmented images. The learning rate of the blocks (`LR_BACKBONE`) is about ten times smaller than that of the output layer: the features of ImageNet must be adjusted, not destroyed. The learning rate decreases over the epochs by a cosine law; the batch normalisation layers of the frozen blocks keep the statistics of ImageNet (`eval` mode). After every epoch the accuracy on the validation part is measured, and the weights of the best epoch are kept.

**Parameters.** `UNFREEZE_BLOCKS` — 0 (only the output layer, now on augmented images) … 13 (the whole network); `EPOCHS_2`; `LR_BACKBONE`.

**Expected result.** The number of trained parameters, the time of an epoch and the estimated time of the stage (printed after the first epoch), the curve of the validation accuracy, the comparison with stage 1.

**What to observe.** With 4 unfrozen blocks 740 527 parameters are trained. On a processor of 16 cores an epoch takes 15 s at 160 px and 27 s at 224 px (in Colab without a GPU several times longer, with a T4 GPU a few seconds). After 3 epochs at 160 px the validation accuracy changes little (0.599 → 0.603); with 224 px and 6 epochs it rises from 0.644 to 0.688. Without augmentation and with all 13 blocks unfrozen on the 24 types the network fits the training images and the test accuracy does not improve (0.317 → 0.316).

**Control question 9.** Why are the last blocks of the network unfrozen and not the first ones, and what happens with a large learning rate of the unfrozen blocks?
""")

code(r"""
#@title 4.4.3 Stage 2: fine-tuning with augmentation (code shown)
UNFREEZE_BLOCKS = 4 #@param {type:"slider", min:0, max:13, step:1}
EPOCHS_2 = 3 #@param {type:"slider", min:1, max:10, step:1}
LR_BACKBONE = 0.0001 #@param [0.00003, 0.0001, 0.0003] {type:"raw"}

import copy
NET.classifier.load_state_dict(HEAD_STAGE1)
for p in NET.features.parameters():
    p.requires_grad = False
blocks = list(NET.features)[len(NET.features) - UNFREEZE_BLOCKS:] if UNFREEZE_BLOCKS else []
for blk in blocks:
    for p in blk.parameters():
        p.requires_grad = True
groups = [{"params": NET.classifier.parameters(), "lr": 3 * LR_BACKBONE if UNFREEZE_BLOCKS else LR_HEAD}]
if UNFREEZE_BLOCKS:
    groups.append({"params": [p for blk in blocks for p in blk.parameters()], "lr": LR_BACKBONE})
opt = torch.optim.AdamW(groups, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS_2)
n_train = sum(p.numel() for p in NET.parameters() if p.requires_grad)
rng44 = np.random.default_rng(0)
best = (STAGE1_VAL, copy.deepcopy(NET.state_dict()), 0)
hist2, t_ep = [], []
for ep in range(EPOCHS_2):
    t0 = time.time()
    NET.train()
    for m in NET.features.modules():                    # frozen batch normalisation keeps the statistics of ImageNet
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()
    for x, yb in batches(rng44.permutation(TR_IDX), train=True):
        opt.zero_grad()
        lossf(NET(x), yb).backward()
        opt.step()
    sched.step()
    NET.eval()
    with torch.no_grad():
        pv = torch.cat([NET(x).argmax(1) for x, _ in batches(VA_IDX, train=False)])
    acc_va = (pv == y_va).float().mean().item()
    t_ep.append(time.time() - t0)
    hist2.append(acc_va)
    if acc_va > best[0]:
        best = (acc_va, copy.deepcopy(NET.state_dict()), ep + 1)
    if ep == 0:
        print(f"Epoch 1: {t_ep[0]:.0f} s on {DEVICE}; estimated time of the stage: {t_ep[0] * EPOCHS_2 / 60:.1f} min")
NET.load_state_dict(best[1])
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(range(0, EPOCHS_2 + 1), [STAGE1_VAL] + hist2, "o-")
ax.set_xlabel("epoch of stage 2 (0 — after stage 1)")
ax.set_ylabel("accuracy, validation part")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
display(pd.DataFrame({"value": [UNFREEZE_BLOCKS, f"{n_train:,}".replace(",", " "), f"{np.mean(t_ep):.0f} s ({DEVICE})",
                                f"{STAGE1_VAL:.3f}", f"{max(hist2):.3f} (epoch {int(np.argmax(hist2)) + 1})",
                                f"epoch {best[2]}" if best[2] else "stage 1 (fine-tuning did not improve)"]},
                     index=["unfrozen blocks", "trained parameters", "time of an epoch", "validation accuracy after stage 1",
                            "best validation accuracy of stage 2", "weights kept"]))
torch.save({"state_dict": NET.state_dict(), "labels": LABELS, "input_size": INPUT_SIZE, "level": LEVEL_44},
           os.path.join(MODEL_DIR, "military_vehicles_mobilenetv3_finetuned.pt"))
""")
md(r"""
## Task 4.4.4. Evaluation of the model and Grad-CAM

**Evaluation.** The models of stage 1 and stage 2 are applied once to the test part; the accuracy, the macro F1 (Lesson 4.1), the metrics of every class and the confusion matrix are computed.

**Grad-CAM** (gradient-weighted class activation mapping) shows which regions of the image raised the score $z_c$ of the recognised class $c$. For the feature maps $A^k$ of the last block (5 × 5 or 7 × 7 cells):
$$\alpha_k = \frac{1}{Z}\sum_{i,j}\frac{\partial z_c}{\partial A^k_{ij}},\qquad L_c = \operatorname{ReLU}\Big(\sum_k \alpha_k A^k\Big),$$
the map $L_c$ is enlarged to the size of the image and shown over it. A correct decision based on the vehicle covers the hull, the turret, the launcher; a decision based on the background, a caption or a frame of a picture reveals a defect of the data.

**Parameters.** `N_CAM` — number of test images with Grad-CAM (half correct, half wrong).

**Expected result.** The table *stage 1 — stage 2* on the test part, the metrics of the classes, the confusion matrix and the Grad-CAM pictures.

**What to observe.** On the test part stage 1 gives 0.583 and stage 2 0.592 at 160 px with 3 epochs; at 224 px with 6 epochs 0.622 and 0.656 (macro F1 0.586 → 0.622), against 0.632 for the frozen network with logistic regression in block 4.2.5. The worst group is artillery: the self-propelled howitzer 2S19 is built on a tank chassis and is often recognised as a tank, the towed D-30 is small in the picture. The Grad-CAM of correct decisions covers the hull, the barrel or the launcher. Many errors fall on paper models, drawings and boxes of model kits; with the default parameters the gallery shows a drawing of the light tank T-80 of the Second World War under the label of the modern main battle tank T-80, recognised as an armoured personnel carrier: an error of the data, not of the network.

**Control question 10.** The Grad-CAM of a correctly recognised "missile system" covers a caption on the picture and not the launcher. Is this model good, and what must be done with the data?
""")

code(r"""
#@title 4.4.4 Test part, confusion matrix and Grad-CAM (code shown)
N_CAM = 8 #@param [4, 8, 12] {type:"raw"}

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
y_te = label_of(TEST_IDX).numpy()
NET.eval()
with torch.no_grad():
    p2 = torch.cat([NET(x).argmax(1) for x, _ in batches(TEST_IDX, train=False)]).cpu().numpy()
# stage 1 model on the test part: the network with the weights of ImageNet and the output layer of stage 1
net1 = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
net1.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(576, len(LABELS)))
net1.classifier.load_state_dict(HEAD_STAGE1)
net1 = net1.to(DEVICE).eval()
with torch.no_grad():
    p1 = torch.cat([net1(x).argmax(1) for x, _ in batches(TEST_IDX, train=False)]).cpu().numpy()
del net1
res = pd.DataFrame({"accuracy": [accuracy_score(y_te, p1), accuracy_score(y_te, p2)],
                    "F1 (macro)": [f1_score(y_te, p1, average="macro"), f1_score(y_te, p2, average="macro")]},
                   index=["stage 1: frozen network + new layer", f"stage 2: fine-tuning of {UNFREEZE_BLOCKS} blocks, {EPOCHS_2} epochs"])
display(res.style.format(precision=3).set_caption(f"Test part: {len(TEST_IDX)} images, {LEVEL_44}, input {INPUT_SIZE} px"))
P, R, F1, S = precision_recall_fscore_support(y_te, p2, labels=range(len(LABELS)), zero_division=0)
per = pd.DataFrame({"precision": P, "recall": R, "F1": F1, "test images": S}, index=LABELS).sort_values("F1")
display(per.style.format(precision=3).set_caption("Metrics of the classes (stage 2)") if len(per) <= 8
        else pd.concat([per.head(4), per.tail(4)]).style.format(precision=3).set_caption("4 worst and 4 best classes (stage 2)"))
cm = confusion_matrix(y_te, p2, labels=range(len(LABELS)))
cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(0.45 * len(LABELS) + 4, 0.4 * len(LABELS) + 3))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
if len(LABELS) <= 8:
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if cmn[i, j] > 0.5 else "k")
ax.set_xticks(range(len(LABELS)))
ax.set_xticklabels(LABELS, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(LABELS)))
ax.set_yticklabels(LABELS, fontsize=8)
ax.set_xlabel("recognised class")
ax.set_ylabel("true class")
ax.set_title(f"stage 2: accuracy {accuracy_score(y_te, p2):.3f}", fontsize=9.5)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson44_confusion.png"), dpi=80)
plt.show()
ok, bad = np.where(p2 == y_te)[0], np.where(p2 != y_te)[0]
rng_cam = np.random.default_rng(1)                       # random test images of all classes
pick = list(rng_cam.choice(ok, min(N_CAM // 2, len(ok)), replace=False)) + list(rng_cam.choice(bad, min(N_CAM - N_CAM // 2, len(bad)), replace=False))
fig, ax = plt.subplots(2, len(pick), figsize=(2.3 * len(pick), 4.9))
for j, k in enumerate(pick):
    i = TEST_IDX[k]
    x = torch.from_numpy(aitbc.imagenet_batch([image_for(i, INPUT_SIZE)], INPUT_SIZE)).to(DEVICE)
    cam, c, prob = aitbc.grad_cam(NET, x, NET.features[-1])
    img = image_for(i, INPUT_SIZE)
    ax[0, j].imshow(img)
    ax[0, j].set_title(f"true: {LABELS[y_te[k]]}", fontsize=7.5)
    ax[1, j].imshow(img)
    ax[1, j].imshow(cam, cmap="jet", alpha=0.45)
    ax[1, j].set_title(f"recognised: {LABELS[c]} ({prob[c]:.2f})", fontsize=7.5, color="k" if c == y_te[k] else "#c62828")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson44_grad_cam.png"), dpi=80)
plt.show()
RESULT_44 = {"level": LEVEL_44, "input_size": INPUT_SIZE, "unfreeze_blocks": UNFREEZE_BLOCKS, "epochs_1": EPOCHS_1, "epochs_2": EPOCHS_2,
             "test_accuracy_stage1": round(float(accuracy_score(y_te, p1)), 3), "test_accuracy_stage2": round(float(accuracy_score(y_te, p2)), 3),
             "test_f1_stage2": round(float(f1_score(y_te, p2, average="macro")), 3)}
""")

md(r"""
## Task 4.4.5. Export to ONNX and check of the inference

**Method.** The fine-tuned network with a softmax at the output is exported to ONNX by the classical exporter of PyTorch (opset 17; in Lesson 4.2 the file of the new exporter was not read by OpenCV 4.12 and gave wrong classes in OpenCV 5.0). The file is executed by ONNX Runtime and by OpenCV DNN on `N_CHECK` test images; the largest difference of the probabilities with PyTorch and the agreement of the classes are computed, and the time per image is measured for a batch of 1 image (a camera frame on the onboard computer) and of 16 images. The preprocessing is written in the file of labels.

**Parameters.** `N_CHECK` — test images of the check; `DYNAMIC_BATCH` — free number of images in the input.

**Expected result.** The ONNX file and its file of labels in `models`, the table of the check, the time per image.

**What to observe.** The ONNX file of 3.6 MB gives the same probabilities as PyTorch up to about 5 · 10⁻⁶ in ONNX Runtime and in OpenCV DNN, and the classes agree for all checked images. One image takes about 1 ms in ONNX Runtime on the processor (about 0.5 ms per image in a batch of 16).

**Control question 11.** The onboard computer gives the frames in BGR order (OpenCV) and without normalisation. What will the model recognise, and where must the correction be made?
""")

code(r"""
#@title 4.4.5 ONNX file of the fine-tuned model, ONNX Runtime and OpenCV DNN (code shown)
N_CHECK = 128 #@param {type:"slider", min:32, max:512, step:32}
DYNAMIC_BATCH = True #@param {type:"boolean"}

import importlib.util, subprocess
need = [p for p, mod in (("onnx", "onnx"), ("onnxruntime", "onnxruntime")) if importlib.util.find_spec(mod) is None]
if need:
    print("Installing:", ", ".join(need))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + need, check=True)
import onnx
import onnxruntime as ort


class WithSoftmax(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image):
        return torch.softmax(self.net(image), dim=1)


model = WithSoftmax(copy.deepcopy(NET).cpu()).eval()
onnx_path = os.path.join(MODEL_DIR, "military_vehicles_mobilenetv3_finetuned.onnx")
kw = dict(input_names=["image"], output_names=["probabilities"], opset_version=17)
if DYNAMIC_BATCH:
    kw["dynamic_axes"] = {"image": {0: "batch"}, "probabilities": {0: "batch"}}
dummy = torch.from_numpy(aitbc.imagenet_batch([image_for(TEST_IDX[0], INPUT_SIZE)], INPUT_SIZE))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.onnx.export(model, (dummy,), onnx_path, dynamo=False, **kw)
onnx.checker.check_model(onnx.load(onnx_path))
LABELS_JSON = {"labels": LABELS, "level": LEVEL_44, "input": "image: N x 3 x S x S float32, RGB order", "size": INPUT_SIZE,
               "resize": "cv2.resize(..., (S, S), interpolation=cv2.INTER_AREA)", "scale": "x / 255",
               "mean": aitbc.IMAGENET_MEAN.tolist(), "std": aitbc.IMAGENET_STD.tolist(), "output": "probabilities N x K (softmax)",
               "model": f"MobileNetV3-small, ImageNet weights, new output layer + fine-tuning of {UNFREEZE_BLOCKS} blocks",
               "test_accuracy": RESULT_44["test_accuracy_stage2"], "opset": 17}
json.dump(LABELS_JSON, open(onnx_path.replace(".onnx", "_labels.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
pos_chk = np.sort(np.random.default_rng(0).choice(len(TEST_IDX), min(N_CHECK, len(TEST_IDX)), replace=False))   # random test images
idx = TEST_IDX[pos_chk]
xb = aitbc.imagenet_batch([image_for(i, INPUT_SIZE) for i in idx], INPUT_SIZE)
with torch.no_grad():
    p_t = model(torch.from_numpy(xb)).numpy()
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
p_o = sess.run(None, {"image": xb})[0] if DYNAMIC_BATCH else np.concatenate([sess.run(None, {"image": xb[k:k + 1]})[0] for k in range(len(xb))])


def time_ms(fn, n):
    t0 = time.time()
    fn()
    return (time.time() - t0) / n * 1000


t1 = time_ms(lambda: [sess.run(None, {"image": xb[k:k + 1]}) for k in range(32)], 32)
t16 = time_ms(lambda: [sess.run(None, {"image": xb[k:k + 16]}) for k in range(0, 64, 16)], 64) if DYNAMIC_BATCH else float("nan")
rows = {"file": f"{onnx_path} ({os.path.getsize(onnx_path) / 2 ** 20:.2f} MB)",
        "ONNX Runtime: largest difference of the probabilities": f"{np.abs(p_o - p_t).max():.2e}",
        "ONNX Runtime: agreement of the classes": f"{np.mean(p_o.argmax(1) == p_t.argmax(1)):.3f}",
        "ONNX Runtime (CPU): time per image, batch 1 / batch 16, ms": f"{t1:.1f} / {t16:.1f}"}
try:
    net_cv = cv2.dnn.readNetFromONNX(onnx_path)
    p_c = []
    for k in range(len(xb)):
        net_cv.setInput(xb[k:k + 1])
        p_c.append(net_cv.forward())
    p_c = np.concatenate(p_c)
    rows["OpenCV DNN: largest difference; agreement of the classes"] = f"{np.abs(p_c - p_t).max():.2e}; {np.mean(p_c.argmax(1) == p_t.argmax(1)):.3f}"
except Exception as e:
    rows["OpenCV DNN"] = f"not read by OpenCV {cv2.__version__}: {str(e).splitlines()[0][:120]}"
rows[f"accuracy of the ONNX model on {len(idx)} random test images"] = f"{np.mean(p_o.argmax(1) == y_te[pos_chk]):.3f}"
display(pd.DataFrame({"value": list(rows.values())}, index=list(rows)))
update_project_json("4.4", {"lesson_4_4": RESULT_44 | {"onnx": os.path.relpath(onnx_path, PROJECT_ROOT),
                                                       "max_difference_onnxruntime": float(np.abs(p_o - p_t).max())}})
""")

md(r"""
## Task 4.4.6. Recognition of a new image by the ONNX model

**Method.** The ONNX file and its file of labels are loaded as on another computer, without the network in memory: the image is prepared exactly as written in the file of labels (RGB order, resize to $S \times S$, scale 1/255, normalisation by the mean and the standard deviation), then ONNX Runtime gives the probabilities of the classes.

**Parameters.** `SOURCE` — a test image or an own image (in Colab a file is uploaded from the computer); `IMAGE_NUMBER` — number of the test image.

**Expected result.** The image with the three most probable classes and their probabilities.

**Control question 12.** Which conditions of the use (distance, angle of view, thermal camera, camouflage) differ from the training data of the model, and how will this affect its decisions?
""")

code(r"""
#@title 4.4.6 Recognition by the ONNX file (code shown)
SOURCE = "test image" #@param ["test image", "own image (upload)"]
IMAGE_NUMBER = 5 #@param {type:"integer"}

meta44 = json.load(open(onnx_path.replace(".onnx", "_labels.json"), encoding="utf-8"))
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def prepare(rgb, meta):
    s = meta["size"]
    x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array(meta["mean"], np.float32)) / np.array(meta["std"], np.float32)
    return x.transpose(2, 0, 1)[None].copy()


img, title = None, ""
if SOURCE.startswith("own") and IN_COLAB:
    from google.colab import files
    up = files.upload()
    if up:
        name, data = next(iter(up.items()))
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        title = name
if img is None:
    if SOURCE.startswith("own"):
        print("Uploading a file is possible in Google Colab; a test image is used.")
    i = TEST_IDX[IMAGE_NUMBER % len(TEST_IDX)]
    img = mv.image(i)
    title = f"test image {i}: true class {mv.table[col44].values[i]}"
prob = sess.run(None, {"image": prepare(img, meta44)})[0][0]
top = np.argsort(-prob)[:3]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
ax[0].imshow(img)
ax[0].set_title(title, fontsize=9)
ax[0].axis("off")
ax[1].barh([meta44["labels"][j] for j in top[::-1]], prob[top[::-1]], color="#1565c0")
ax[1].set_xlim(0, 1)
ax[1].set_xlabel("probability (ONNX Runtime)")
plt.tight_layout()
plt.show()
print("Three most probable classes:", "; ".join(f"{meta44['labels'][j]} {prob[j]:.2f}" for j in top))
""")

md(r"""
### Report of Lesson 4.4

Double-click this cell and write the answers.

1. Parts of the data, input size, augmentation; answer to control question 7:
2. Stage 1: trained parameters, epochs, validation accuracy; answer to control question 8:
3. Stage 2: unfrozen blocks, trained parameters, time of an epoch, validation accuracy; answer to control question 9:
4. Test part: stage 1 and stage 2, worst classes, Grad-CAM of correct and wrong decisions; answer to control question 10:
5. ONNX file, differences of the probabilities, time per image; answer to control question 11:
6. Recognition of a new image; answer to control question 12:

---
## Self-check of Topic 4

1. Which data set is needed to train a verifier of thermal candidates, and why must the candidates of the training be produced by the same detector as in use?
2. How are the training, validation and test parts formed for frames of reconnaissance flights, and what is estimated on each of them?
3. Why is the average precision a better description of a verifier than the accuracy, and how is the working threshold chosen?
4. What do the two stages of transfer learning change in the network, and when is the fine-tuning worth its cost?
5. Which checks are made before an ONNX model is put on the onboard computer?

## Data sources and licences

The course library reads the files directly from the repositories of the data sets; the data are not redistributed with the notebook. When images or results are published, the data sets are cited.

1. **HIT-UAV.** Suo J., Wang T., Zhang X., Chen H., Zhou W., Shi W. HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection. *Scientific Data*, 10, 227 (2023). https://doi.org/10.1038/s41597-023-02066-6 · Data: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset · Licence CC BY 4.0.
2. **Military vehicles.** Kricheli J. S., Vo K., Datta A., Ozgur S., Shakarian P. Error Detection and Constraint Recovery in Hierarchical Multi-Label Classification without Prior Knowledge. arXiv:2407.15192 (2024) · Data: https://huggingface.co/datasets/leibnitz-lab/military_vehicles · Licence declared by the authors: MIT; the photographs were collected from web sites, the rights to the individual photographs are not stated: use them only for training and research.
3. **ImageNet weights of MobileNetV3-small** — torchvision (BSD-3-Clause); Howard A. et al. Searching for MobileNetV3. *ICCV 2019*, 1314–1324.
4. **HOG.** Dalal N., Triggs B. Histograms of Oriented Gradients for Human Detection. *CVPR 2005*, 886–893. **Grad-CAM.** Selvaraju R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*, 618–626.

**Next notebook.** Notebook 08 — Topic 5, Lessons 5.1 (lecture) and 5.2 (group class): object detection by YOLO, the metrics IoU, confidence and mAP, small targets; segmentation of the traversable area for a ground robot, tracking of objects (ByteTrack), the video-analysis pipeline and the requirements of real time.
""")


# =====================================================================================
def build(path_out, lib_path="aitbc.py"):
    lib = open(lib_path, encoding="utf-8").read()
    assert "\'\'\'" not in lib, "library source must not contain triple single quotes"
    nb = new_notebook()
    nb.metadata = {"colab": {"provenance": [], "toc_visible": True, "name": os.path.basename(path_out)},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
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
    out = sys.argv[1] if len(sys.argv) > 1 else "AITBC_NB07_Thermal_HOG_SVM_and_MobileNet_transfer_learning.ipynb"
    nb = build(out)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    print(f"written {out}: {len(nb.cells)} cells ({n_md} text, {len(nb.cells) - n_md} code)")
