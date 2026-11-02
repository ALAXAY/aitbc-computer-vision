# Builder of Notebook 06 of the course "AI Technologies (Basic Course)" (AIT-BC).
# Topic 4, Lessons 4.1 (lecture) and 4.2 (group class). All exercises use real open data sets.
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
# AI Technologies (Basic Course) · Notebook 06
## Topic 4. Machine learning and neural networks in the recognition of military objects

| Lesson | Type, duration | Title |
|---|---|---|
| **4.1** | Lecture, 2 h | Machine learning in the recognition of military objects: formation of a data set, HOG features, SVM and Random Forest classifiers, metrics of classification |
| **4.2** | Group class, 2 h | Neural networks: the neuron, the fully connected network, the convolutional network, MobileNet, transfer learning and export of the model to ONNX |

Course: *AI Technologies (Basic Course)*, AIT-BC · Content module 2 · Kruty Heroes Military Institute of Telecommunications and Information Technology, Kyiv, 2026.

**How to work with this notebook**

1. Run the cells from top to bottom: *Runtime → Run all*, or `Shift+Enter` cell by cell. Section 0 is always run first; after a restart of the runtime it is run again.
2. Every block begins with a text cell: the purpose, the formulas, the parameters of the form, the expected result and the control question.
3. The blocks are demonstrations with measurements on real data: change the parameters of the forms and run the cell again. Blocks of Lesson 4.2 use the results of Lesson 4.1 (features, crops); if a block of Lesson 4.1 was not run, the block of Lesson 4.2 computes what it needs itself.
4. A GPU is not required: every block runs on the processor of the Colab session in 1–4 minutes with the default parameters. With a GPU (*Runtime → Change runtime type → T4 GPU*) the blocks of Lesson 4.2 run faster.
5. The results are saved in the course project on Google Drive (`outputs`, `models`, `project.json`), created in Notebook 02.

**Data.** Only real images are used, with the annotation of the authors of the data sets:
- **Military vehicles** — 9444 photographs of 24 types of military vehicles (tanks, infantry fighting vehicles, armoured personnel carriers, artillery, multiple rocket launchers, air defence and missile systems) with labels of the type and of the class;
- **SULAND** — 33 771 frames of 55 video sequences of a camera moved over grass, fallen leaves and gravel with surrogates of the anti-personnel surface landmines PFM-1 ("butterfly") and PMA-2 ("starfish"), recorded in Italy and in the USA, with boxes of the mines in the YOLO format in two versions of the annotation (the original one and the refined one).

**Contents**

- **0. Preparation** — 0.1 runtime, course library and data sets · 0.2 course project
- **Lesson 4.1 (lecture)**
  - 4.1.1 The task of recognition and the data set of military vehicles
  - 4.1.2 Audit of the data set: duplicates and contradictory labels
  - 4.1.3 Annotation of frames: the YOLO format, statistics and quality of the annotation
  - 4.1.4 HOG features
  - 4.1.5 SVM and Random Forest classifiers; accuracy, precision, recall, F1
  - 4.1.6 Division of the data set: random frames or whole sequences
- **Lesson 4.2 (group class)**
  - 4.2.1 The artificial neuron
  - 4.2.2 The fully connected network
  - 4.2.3 The convolutional network: convolution and feature maps
  - 4.2.4 MobileNet: depthwise separable convolution and the cost of computation
  - 4.2.5 Transfer learning: features of a network trained on ImageNet
  - 4.2.6 Export of the model to ONNX and check of the inference
- **Summary**: self-check of Lessons 4.1–4.2, data sources and licences
""")

# =====================================================================================
md(r"""
---
# 0. Preparation

### Block 0.1. Runtime, course library and data sets

**Purpose.** To prepare the runtime: the libraries, the course library `aitbc` (version 2.5.0 adds the readers of the data sets of military vehicles and of SULAND, the HOG and CNN features and the diagrams of the networks) and the open data sets of the lesson.

**Parameters.**
- `CACHE_ON_DRIVE` — keep the downloaded files on Google Drive (they remain after the end of the session); otherwise they are kept on the disk of the session.
- `PREFETCH_ALL` — download now the data of blocks 4.1.1–4.1.5: the two parquet files of military vehicles (166 MB), the two archives of the SULAND annotation (16 MB) and 12 frames of SULAND (6 MB). The 360 frames of block 4.1.6 (about 190 MB) are downloaded in that block.

**Output.** A table of versions, the processor or GPU of the session and the size of the cache.
""")

code(r"""
#@title 0.1 Runtime, course library and data sets (run once) { display-mode: "form" }
CACHE_ON_DRIVE = False #@param {type:"boolean"}
PREFETCH_ALL = True #@param {type:"boolean"}

import sys, os, platform, time, json, math, importlib, shutil
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
mv = aitbc.MilitaryVehicles()                      # 24 types of military vehicles
suland = aitbc.SULAND("v2")                        # refined annotation of SULAND
GALLERY_SEQ = ["ITA-v13", "ITA-v32", "ITA-v2", "ITA-v44", "US2", "US10"]
GALLERY = [(s, int(f)) for s in GALLERY_SEQ
           for f in suland.frames[(suland.frames.seq == s) & (suland.frames.n_boxes > 0)].frame.quantile([0.3, 0.7], interpolation="nearest")]
if PREFETCH_ALL:
    suland_v1 = aitbc.SULAND("v1")                 # original annotation of the authors
    suland.prefetch(GALLERY)
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

**Purpose.** To connect the course project created in Notebook 02. The results of this notebook are saved in it: figures and tables in `outputs`, the trained model in the ONNX format in `models` and the parameters of the lessons in `project.json`.

**Parameters.** `USE_DRIVE` — the project is on Google Drive (Colab asks for access to the Google account); otherwise it is on the disk of the session. `PROJECT_PATH` — folder of the project; the same as in Notebooks 02–05.

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
# Lesson 4.1 (Lecture). Machine learning in the recognition of military objects

**Learning questions**
1. The tasks of the recognition of military objects and the place of machine learning in them.
2. Formation and annotation of a data set on frames of a video camera.
3. Features of the histogram of oriented gradients (HOG).
4. Classifiers: the support vector machine (SVM) and the random forest (Random Forest).
5. Metrics of the quality of classification: accuracy, precision, recall and F1.

**Recognition** answers the question *what is the object*. Three tasks are distinguished:
- **classification** — one class for an image or for a fragment of a frame (a crop): "T-72", "BTR-80", "background";
- **detection** — the boxes of all objects of the given classes in a frame and the class of each box (Topic 5);
- **segmentation** — the class of every pixel (Topic 5).

**Machine learning** replaces the rules written by the developer with a function fitted to examples: a set of images with known classes (the training sample) defines the parameters of the classifier, and a separate set (the test sample) estimates its quality. The chain of the classical approach is *image → features (HOG) → classifier (SVM, Random Forest) → class*; in a neural network (Lesson 4.2) the features are also fitted to the examples. In both cases the result is not better than the data: the lecture begins with the data set, its audit and its annotation.
""")

md(r"""
### Block 4.1.1. The task of recognition and the data set of military vehicles

**Purpose.** To get acquainted with a real data set for the classification of military vehicles: the classes, the number of examples, the sizes and the sources of the images.

**Data set.** 9444 photographs collected by the authors from web sites (Kricheli et al., 2024): 7823 in the training part and 1621 in the test part, 24 fine types. The authors also give 7 coarse classes. The course uses its own grouping by the purpose of the vehicle, because two groups of the authors join different kinds of weapons:

| Group of the course | Types | Class of the authors |
|---|---|---|
| Tank | T-14, T-62, T-64, T-72, T-80, T-90 | Tank |
| IFV (infantry fighting vehicle) | BMP-1, BMP-2, BMP-T15, BMD | BMP, BMD |
| APC/recon (armoured personnel carrier and scout car) | BTR-60, BTR-70, BTR-80, BRDM, MT-LB | BTR, MT_LB |
| Artillery | 2S19 Msta (self-propelled howitzer), D-30 (**towed** howitzer) | Self Propelled Artillery |
| MLRS (multiple rocket launcher) | BM-30, Tornado, TOS-1 | Self Propelled Artillery |
| Air defence/radar | Pantsir-S1, 30N6E | Air Defense |
| Missile system | Iskander, RS-24 | Air Defense |

**Parameters.**
- `LEVEL` — the labels shown: the 24 types, the 7 groups of the course or the 7 classes of the authors.
- `EXAMPLES_PER_CLASS` — the number of examples of every class in the gallery; `SEED` — the choice of the examples.

**Expected result.** The table of the classes with the numbers of images, the gallery and the distribution of the sizes of the images.

**What to observe.** The number of training images differs between the types by a factor of 3.6 (from 119 for 30N6E to 424 for T-72), and the test part is distributed differently: from 6 % of the images of a type (BMP-1, T-64) to 35 % (BM-30). 90 % of the images are not larger than 300 × 300 pixels: they are web thumbnails of different origin. Among them there are photographs of real vehicles, renderings, boxes of scale models, infographics, explosions and photographs where the vehicle is not visible at all (with `SEED = 0`: the box of a model of the THAAD system under "30N6E", a person in an office under "RS-24"). The label of the type does not guarantee what the image contains.

**Control question 1.** Why does the class "Air Defense" of the authors make the task harder for a classifier, and what can a classifier learn from it?
""")

code(r"""
#@title 4.1.1 Classes and images of the data set of military vehicles { display-mode: "form" }
LEVEL = "24 types" #@param ["24 types", "7 groups of the course", "7 classes of the authors"]
EXAMPLES_PER_CLASS = 1 #@param {type:"slider", min:1, max:4, step:1}
SEED = 0 #@param {type:"integer"}

COL = {"24 types": "cls", "7 groups of the course": "group", "7 classes of the authors": "source_group"}
T = mv.table
col = COL[LEVEL]
counts = T.pivot_table(index=col, columns="split", values="file", aggfunc="count", fill_value=0)[["train", "test"]]
if col == "cls":
    counts.insert(0, "description", [aitbc.MilitaryVehicles.TYPES[c] for c in counts.index])
    counts.insert(1, "group of the course", [aitbc.MilitaryVehicles.GROUPS[c] for c in counts.index])
    counts.insert(2, "class of the authors", [T[T.cls == c].source_group.mode()[0] for c in counts.index])
else:
    counts.insert(0, "types (training images)", [", ".join(f"{k} {v}" for k, v in T[(T[col] == g) & (T.split == "train")].cls.value_counts().items())
                                                 for g in counts.index])
counts["share of test, %"] = (100 * counts.test / (counts.train + counts.test)).round(1)
display(counts.style.format(precision=1).set_caption(f"{LEVEL}: {len(counts)} classes, {int(counts.train.sum())} training and {int(counts.test.sum())} test images"))

rng = np.random.default_rng(SEED)
classes = list(counts.index)
ncol = 8 if EXAMPLES_PER_CLASS == 1 else EXAMPLES_PER_CLASS * 2
items = []
for c in classes:
    idx = T.index[(T[col] == c) & (T.split == "train")].values
    items += [(c, i) for i in rng.choice(idx, EXAMPLES_PER_CLASS, replace=False)]
nrow = int(math.ceil(len(items) / ncol))
fig, ax = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.0 * nrow))
for a in np.ravel(ax):
    a.axis("off")
for a, (c, i) in zip(np.ravel(ax), items):
    a.imshow(mv.image(i))
    a.set_title(f"{c}\n{T.width[i]}×{T.height[i]}", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson41_gallery.png"), dpi=80)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 3.6))
counts[["train", "test"]].plot.bar(ax=ax[0], width=0.8)
ax[0].set_ylabel("images")
ax[0].set_title(LEVEL)
ax[0].tick_params(axis="x", labelsize=7.5)
ax[1].scatter(T.width, T.height, s=4, alpha=0.3)
ax[1].set_xlabel("width, px")
ax[1].set_ylabel("height, px")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_title(f"sizes of the images: median {int(T.width.median())} × {int(T.height.median())} px")
plt.tight_layout()
plt.show()
share_small = float(((T.width <= 300) & (T.height <= 300)).mean())
print(f"Images not larger than 300 × 300 px: {100 * share_small:.0f} %; ratio of the largest to the smallest class "
      f"in the training part: {counts.train.max() / counts.train.min():.1f}")
""", form=True)

md(r"""
### Block 4.1.2. Audit of the data set: duplicates and contradictory labels

**Purpose.** To check the data set before training: to find repeated images inside a part, the images of the test part that also occur in the training part, and the contradictions of the labels of the same image.

**Method.**
- An **exact duplicate** is an image with the same file contents; it is found by the MD5 checksum of the file.
- A **near duplicate** is the same photograph after a change of the size, of the compression or of a small crop. It is found by the *difference hash* (dHash): the image is reduced to 9 × 8 pixels in grey, and every one of the 64 bits is the sign of the difference of two neighbouring pixels in a row. The distance between two hashes is the number of different bits (the Hamming distance $d_H$, from 0 to 64).

**Parameters.**
- `HAMMING_MAX` — the largest Hamming distance at which two images are regarded as near duplicates.
- `SHOW_PAIRS` — the number of pairs in the gallery.

**Expected result.** The table of the audit and the gallery of pairs *test image — the nearest training image* with their labels and the distance.

**What to observe.** In the test part 18 images are byte-for-byte copies of training images, and **all 18 have a different label in the training part** (BM-30 and Tornado, BTR-60 and BTR-80, T-90 and T-72 and others). Such an image cannot be classified correctly by any model, and a model that remembers the training image is counted wrong. After the removal of the exact copies 98 test images (6 %) still have a near duplicate at $d_H \le 2$ in the training part: the same photograph re-compressed or resized, 7 of them with a different label. At $d_H \ge 4$ also different vehicles on a uniform background (renderings) come together: the threshold is a compromise checked by eye. Blocks 4.1.5, 4.2.2 and 4.2.5 use the test part without exact copies and without near duplicates at $d_H \le 2$: 1502 images.

**Control question 2.** Why must the duplicates between the training and the test part be removed before the estimation of the quality, and what is done with the contradictory labels?
""")

code(r"""
#@title 4.1.2 Exact and near duplicates, contradictory labels { display-mode: "form" }
HAMMING_MAX = 2 #@param {type:"slider", min:0, max:12, step:1}
SHOW_PAIRS = 8 #@param {type:"slider", min:4, max:16, step:4}

T = mv.table
TRAIN_IDX, TEST_IDX = mv.clean()                     # parts without exact copies (MD5)
t0 = time.time()
HASHES = mv.hashes()                                 # dHash of all images (computed once)
print(f"dHash of {len(T)} images: {time.time() - t0:.1f} s")
dist, nn = aitbc.hamming_nearest(HASHES[TEST_IDX], HASHES[TRAIN_IDX])
near = pd.DataFrame({"test": TEST_IDX, "train": TRAIN_IDX[nn], "d": dist})
near["cls_test"] = T.cls.values[near.test]
near["cls_train"] = T.cls.values[near.train]
near["same"] = near.cls_test == near.cls_train
sel = near[near.d <= HAMMING_MAX]
exact = T[T.in_train]
audit = pd.DataFrame({"value": [
    int(T.dup_in_split[T.split == "train"].sum()), int(T.dup_in_split[T.split == "test"].sum()),
    len(exact), int((exact.train_label != exact.cls).sum()),
    f"{len(TRAIN_IDX)} / {len(TEST_IDX)}",
    len(sel), int((sel.cls_test != sel.cls_train).sum()), len(mv.clean(near=2)[1])]},
    index=["repeated copies inside the training part (MD5)", "repeated copies inside the test part (MD5)",
           "test images that are copies of training images (MD5)", "   of them with a different label",
           "cleaned parts: training / test images",
           f"near duplicates (dHash, d ≤ {HAMMING_MAX}) of cleaned test images in the training part",
           "   of them with a different label",
           "test images used in blocks 4.1.5, 4.2.2, 4.2.5 (without exact copies and near duplicates d ≤ 2)"])
display(audit)
display(exact.groupby(["cls", "train_label"]).size().rename("test images").reset_index()
        .rename(columns={"cls": "label in the test part", "train_label": "label of the same file in the training part"}))

pool = sel if len(sel) >= SHOW_PAIRS else near
show = pool.sort_values(["same", "d"]).head(SHOW_PAIRS // 2)                       # first the contradictory labels
show = pd.concat([show, pool[pool.same].sort_values("d", ascending=False).head(SHOW_PAIRS - len(show))])
fig, ax = plt.subplots(2, len(show), figsize=(2.3 * len(show), 4.6), squeeze=False)
for j, r in enumerate(show.itertuples()):
    ax[0, j].imshow(mv.image(r.test))
    ax[0, j].set_title(f"test: {r.cls_test}\nd = {r.d}", fontsize=8, color="k" if r.cls_test == r.cls_train else "#c62828")
    ax[1, j].imshow(mv.image(r.train))
    ax[1, j].set_title(f"training: {r.cls_train}", fontsize=8)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson41_duplicates.png"), dpi=80)
plt.show()
counts_d = near.d.value_counts().sort_index()
print("Number of cleaned test images by the distance to the nearest training image, d = 0...8:",
      ", ".join(f"{int(counts_d.get(k, 0))}" for k in range(9)))
""", form=True)

md(r"""
### Block 4.1.3. Annotation of frames: the YOLO format, statistics and quality of the annotation

**Purpose.** To study the annotation of objects on the frames of a video camera by the example of the SULAND data set: the format of the boxes, the transformation of coordinates, the statistics of a data set for detection and the errors of the annotation.

**SULAND.** A camera (iPhone 13, 1440 × 1920 pixels, vertical frame) is moved at a height of about one metre over grass, fallen leaves and gravel, as a camera of a ground robot for the survey of an area. On the ground lie surrogates of two anti-personnel surface landmines: **PFM-1** ("butterfly", a scattered mine) and **PMA-2** ("starfish"). The frames are cut from 55 video sequences: 45 recorded in Italy (training, validation and test parts) and 10 recorded in the USA (a separate part with other soil and lighting). The refined annotation **SULAND_v2** (2026) corrects the original annotation **v1** of the authors (2024): missing boxes, false boxes, inaccurate boundaries and the inverted numbers of the classes in the USA part.

**YOLO format.** For every frame `image.jpg` there is a text file `image.txt`, one line per object:

`class x_c y_c w h`, where $x_c, y_c$ are the coordinates of the centre of the box and $w, h$ its width and height, as fractions of the width $W$ and the height $H$ of the frame (0…1). A frame without objects has an empty file. Transformation into pixels:

$$x_1 = (x_c - w/2)\,W,\quad y_1 = (y_c - h/2)\,H,\quad x_2 = (x_c + w/2)\,W,\quad y_2 = (y_c + h/2)\,H;$$

in the COCO format the box is $[x_1, y_1, x_2 - x_1, y_2 - y_1]$ in pixels.

**Parameters.**
- `SEQUENCE` — a sequence (ITA — Italy, US — USA); `FRAME_POSITION` — the position of the frame in the sequence (0 — first, 1 — last frame).
- `SHOW_V1` — draw also the boxes of the original annotation v1 (white).

**Expected result.** The frame with the boxes, the lines of its file in the YOLO format with their transformation into pixels, the statistics of the parts and the comparison of the two versions of the annotation.

**What to observe.** 72–76 % of the Italian frames contain no mine (19 % of the USA frames): a detector must learn the frames without objects too. The median box of a mine is about 95 × 80 px in the Italian part (6.6 × 4.2 % of the frame 1440 × 1920) and about 190 × 170 px in the USA part. Between v1 and v2 the boxes of the same mines in the Italian part differ by a median IoU of 0.62 (34 % of them by IoU < 0.5), 1951 boxes were added and 436 removed; in the USA part 3602 of 3856 boxes changed their class (the inverted numbers of the classes in v1). The annotation is a source of errors comparable with the errors of the model.

**Control question 3.** A frame of 1440 × 1920 px has the line `1 0.5 0.25 0.1 0.05`. Which mine is it, and what are its box in pixels and its box in the COCO format?
""")

code(r"""
#@title 4.1.3 Annotation of SULAND frames, statistics, versions v1 and v2 { display-mode: "form" }
SEQUENCE = "ITA-v13" #@param ["ITA-v13", "ITA-v32", "ITA-v2", "ITA-v44", "US2", "US10"]
FRAME_POSITION = 0.3 #@param {type:"slider", min:0, max:1, step:0.05}
SHOW_V1 = True #@param {type:"boolean"}

if "suland_v1" not in globals():
    suland_v1 = aitbc.SULAND("v1")
fr_seq = suland.frames[(suland.frames.seq == SEQUENCE) & (suland.frames.n_boxes > 0)].frame.values
frame_no = int(fr_seq[int(round(FRAME_POSITION * (len(fr_seq) - 1)))])
img, objs = suland.load(SEQUENCE, frame_no)
H, W = img.shape[:2]
vis = aitbc.draw_boxes(img, objs, colors=aitbc.SULAND.COLORS, thickness=4, font=1.4)
objs1 = suland_v1.objects(SEQUENCE, frame_no, W, H)
if SHOW_V1:
    for o in objs1:
        x1, y1, x2, y2 = [int(round(v)) for v in o["box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_AA)
part = suland.frames[suland.frames.seq == SEQUENCE].iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 1]})
ax[0].imshow(vis)
ax[0].set_title(f"{SEQUENCE} ({part.domain}, {part.split}), frame {frame_no}: v2 coloured" + (", v1 white" if SHOW_V1 else ""), fontsize=9)
ax[0].axis("off")
if objs:
    o = objs[0]
    x1, y1, x2, y2 = [int(round(v)) for v in o["box"]]
    m = int(1.0 * max(x2 - x1, y2 - y1))
    ax[1].imshow(vis[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m])
    ax[1].set_title(f"enlarged: {o['cls']} ({aitbc.SULAND.NICKNAMES[o['cls']]}), box {x2 - x1} × {y2 - y1} px", fontsize=9)
ax[1].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"lesson41_annotation_{SEQUENCE}_{frame_no}.png"), dpi=70)
plt.show()

rows = []
for r in suland.boxes[(suland.boxes.seq == SEQUENCE) & (suland.boxes.frame == frame_no)].itertuples():
    b = aitbc.yolo_to_xyxy(r.cx, r.cy, r.w, r.h, W, H)
    rows.append({"YOLO line (v2)": f"{r.cls_id} {r.cx:.4f} {r.cy:.4f} {r.w:.4f} {r.h:.4f}", "class": r.cls,
                 "[x1, y1, x2, y2], px": [round(v) for v in b], "COCO [x, y, w, h], px": [round(v) for v in aitbc.xyxy_to_coco(b)]})
display(pd.DataFrame(rows))
print("Boxes of the same frame in v1 (class, [x1, y1, x2, y2]):", [(o["cls"], [round(v) for v in o["box"]]) for o in objs1] or "none")

# statistics of the parts (annotation v2)
F, Bx = suland.frames, suland.boxes
stats = F.groupby(["domain", "split"]).agg(sequences=("seq", "nunique"), frames=("frame", "size"),
                                           frames_without_mines=("n_boxes", lambda s: f"{100 * (s == 0).mean():.0f} %"),
                                           boxes_PFM1=("n_pfm1", "sum"), boxes_PMA2=("n_pma2", "sum"))
wpx = Bx.w * 1440
hpx = Bx.h * 1920
stats["median box, px"] = [f"{wpx[(Bx.domain == d) & (Bx.split == s)].median():.0f} × {hpx[(Bx.domain == d) & (Bx.split == s)].median():.0f}"
                           for d, s in stats.index]
display(stats.style.set_caption("SULAND_v2: parts of the data set"))


# comparison of the versions of the annotation: a box of v2 is matched with the box of v1 of the largest IoU (IoU ≥ 0.1)
def compare_versions(ga, gb):
    # ga, gb: {(sequence, frame): boxes} of the old and the new annotation
    out = dict(added=0, removed=0, class_changed=0, ious=[])
    for key in set(ga) | set(gb):
        A = [] if key not in ga else [(r.cls, aitbc.yolo_to_xyxy(r.cx, r.cy, r.w, r.h, 1, 1)) for r in ga[key].itertuples()]
        B = [] if key not in gb else [(r.cls, aitbc.yolo_to_xyxy(r.cx, r.cy, r.w, r.h, 1, 1)) for r in gb[key].itertuples()]
        used = set()
        for cb, bb in B:
            best, bi = 0.0, None
            for i, (ca, ba) in enumerate(A):
                if i not in used:
                    v = aitbc.box_iou(ba, bb)
                    if v > best:
                        best, bi = v, i
            if bi is None or best < 0.1:
                out["added"] += 1
            else:
                used.add(bi)
                out["ious"].append(best)
                out["class_changed"] += int(A[bi][0] != cb)
        out["removed"] += len(A) - len(used)
    return out


if "VERSION_CMP" not in globals():
    VERSION_CMP = {}
    for dom in ("data-iid", "data-ood"):
        VERSION_CMP[dom] = compare_versions({k: g for k, g in suland_v1._by_frame.items() if g.domain.iat[0] == dom},
                                            {k: g for k, g in suland._by_frame.items() if g.domain.iat[0] == dom})
cmp_rows = {}
for dom, c in VERSION_CMP.items():
    iou = np.array(c["ious"])
    cmp_rows[dom] = {"boxes v1": int((suland_v1.boxes.domain == dom).sum()), "boxes v2": int((suland.boxes.domain == dom).sum()),
                     "added in v2": c["added"], "removed in v2": c["removed"], "class changed": c["class_changed"],
                     "median IoU of the same box v1-v2": round(float(np.median(iou)), 3),
                     "share of the same boxes with IoU < 0.5": f"{100 * (iou < 0.5).mean():.0f} %"}
display(pd.DataFrame(cmp_rows).T.style.set_caption("Original annotation v1 and refined annotation v2"))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.2))
ax[0].hist(np.clip(wpx, 0, 600), bins=60, alpha=0.7, label="width")
ax[0].hist(np.clip(hpx, 0, 600), bins=60, alpha=0.7, label="height")
ax[0].set_xlabel("size of the box, px (frame 1440 × 1920)")
ax[0].legend()
ax[1].hist(np.array(VERSION_CMP["data-iid"]["ious"]), bins=40)
ax[1].set_xlabel("IoU of the same mine in v1 and v2 (Italy)")
plt.tight_layout()
plt.show()
""", form=True)

md(r"""
### Block 4.1.4. HOG features

**Purpose.** To compute the histogram of oriented gradients (HOG, Dalal and Triggs, 2005) of an image of a military vehicle and of a fragment of a SULAND frame, and to relate its parameters to the length of the feature vector.

**Formulas.**
1. Gradient of brightness in every pixel: $g_x = I(x+1, y) - I(x-1, y)$, $g_y = I(x, y+1) - I(x, y-1)$; magnitude $m = \sqrt{g_x^2 + g_y^2}$; orientation $\theta = \arctan(g_y / g_x)$ in $[0°, 180°)$ (the sign of the edge is not used).
2. **Cell** of $c \times c$ pixels: a histogram of $n_o$ bins of orientation, every pixel adds its magnitude $m$ to the bin of its orientation $\theta$.
3. **Block** of $b \times b$ cells: the histograms of its cells are joined into a vector $v$ and normalised, $v \leftarrow v / \sqrt{\lVert v \rVert^2 + \varepsilon^2}$ (L2-Hys: values above 0.2 are clipped and the vector is normalised again). The normalisation makes the features weakly dependent on the illumination and the contrast; the blocks overlap with a step of one cell.
4. Length of the vector of an image of $S \times S$ pixels:
$$N = \left(\frac{S}{c} - b + 1\right)^2 b^2\, n_o; \qquad S = 64,\; c = 8,\; b = 2,\; n_o = 9:\quad N = 7^2 \cdot 4 \cdot 9 = 1764.$$

**Parameters.**
- `IMAGE` — the image: a military vehicle of the data set of block 4.1.1 or a fragment of a SULAND frame around a mine or without a mine (background).
- `SIZE` — the size $S$ to which the image is reduced; `CELL` — $c$; `BLOCK` — $b$; `ORIENTATIONS` — $n_o$.

**Expected result.** The image, the magnitude of the gradient, the picture of the HOG (in every cell the dominating orientations as segments) and the comparison of the length $N$ by the formula with the length computed by `skimage.feature.hog`.

**What to observe.** The HOG keeps the shape (the contours of the hull, the barrel, the wheels; the rays of the PMA-2) and loses the colour and the small texture. A small cell gives a long vector with details, a large cell a short and coarse vector. The background of grass gives cells with many orientations of similar weight, the mine gives cells with a dominating orientation.

**Control question 4.** What length has the HOG vector for $S = 96$, $c = 8$, $b = 2$, $n_o = 9$, and why does it grow faster than the area of the image in cells?
""")

code(r"""
#@title 4.1.4 Histogram of oriented gradients { display-mode: "form" }
IMAGE = "T-72 (tank)" #@param ["T-72 (tank)", "BM-30 (multiple rocket launcher)", "BTR-80 (armoured personnel carrier)", "SULAND: PFM-1 (butterfly)", "SULAND: PMA-2 (starfish)", "SULAND: background"]
SIZE = 64 #@param {type:"slider", min:32, max:128, step:16}
CELL = 8 #@param [4, 8, 16] {type:"raw"}
BLOCK = 2 #@param [1, 2, 3] {type:"raw"}
ORIENTATIONS = 9 #@param [6, 9, 12, 18] {type:"raw"}

from skimage.feature import hog


def suland_crop(kind, scale=1.5):
    # fragment of a gallery frame of block 0.1: around the first mine of the class, or without mines
    for s, f in GALLERY:
        img, objs = suland.load(s, f)
        if kind == "background":
            x0, y0 = 100, img.shape[0] - 400
            return img[y0:y0 + 220, x0:x0 + 220], f"{s}, frame {f}: background"
        for o in objs:
            if o["cls"] == kind:
                x1, y1, x2, y2 = o["box"]
                side = int(scale * max(x2 - x1, y2 - y1))
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                return img[max(0, cy - side // 2):cy + side // 2, max(0, cx - side // 2):cx + side // 2], f"{s}, frame {f}: {kind}"
    raise ValueError(kind)


if IMAGE.startswith("SULAND"):
    kind = "background" if "background" in IMAGE else IMAGE.split(": ")[1].split(" ")[0]
    src, caption = suland_crop(kind)
else:
    t_ = IMAGE.split(" ")[0]
    i_ = mv.table.index[(mv.table.cls == t_) & (mv.table.split == "train")][3]
    src, caption = mv.image(i_), f"military vehicles: {t_}, image {i_}"
g = cv2.resize(aitbc.gray(src), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
if SIZE // CELL < BLOCK:
    raise ValueError("the image must contain at least BLOCK cells")
t0 = time.time()
vec, hog_img = hog(g, orientations=ORIENTATIONS, pixels_per_cell=(CELL, CELL), cells_per_block=(BLOCK, BLOCK), visualize=True)
dt = (time.time() - t0) * 1000
gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
ax[0].imshow(src)
ax[0].set_title(caption, fontsize=8.5)
ax[1].imshow(g, cmap="gray")
ax[1].set_title(f"grey, {SIZE} × {SIZE} px, cells {CELL} × {CELL}", fontsize=8.5)
for k in range(0, SIZE + 1, CELL):
    ax[1].axhline(k - 0.5, color="y", lw=0.4)
    ax[1].axvline(k - 0.5, color="y", lw=0.4)
ax[2].imshow(np.hypot(gx, gy), cmap="magma")
ax[2].set_title("magnitude of the gradient m", fontsize=8.5)
ax[3].imshow(hog_img, cmap="gray")
ax[3].set_title(f"HOG: {ORIENTATIONS} orientations per cell", fontsize=8.5)
for a in ax:
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson41_hog.png"), dpi=80)
plt.show()
n_formula = aitbc.hog_length(SIZE, CELL, BLOCK, ORIENTATIONS)
display(pd.DataFrame({"value": [f"{SIZE // CELL} × {SIZE // CELL}", f"{SIZE // CELL - BLOCK + 1} × {SIZE // CELL - BLOCK + 1}",
                                n_formula, len(vec), f"{dt:.1f} ms"]},
                     index=["cells", "blocks (step of one cell)", "length N by the formula", "length computed by skimage",
                            "time of the computation for one image"]))
""", form=True)

md(r"""
### Block 4.1.5. SVM and Random Forest classifiers; accuracy, precision, recall, F1

**Purpose.** To train classifiers of military vehicles on HOG vectors and to estimate them by the metrics of classification on the cleaned test part (block 4.1.2).

**Classifiers.**
- **Linear SVM.** The decision function $f(x) = w^\top x + b$; the hyperplane is chosen with the largest margin to the nearest examples of the classes (support vectors): $\min_{w, b}\ \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \max(0,\, 1 - y_i f(x_i))$. Here it is trained by stochastic gradient descent (`SGDClassifier`, hinge loss); several classes — "one class against all".
- **SVM with the RBF kernel.** $f(x) = \sum_i \alpha_i y_i \exp(-\gamma \lVert x - x_i \rVert^2) + b$: the boundary between the classes is curved. $C$ is the price of an error on the training examples: a large $C$ follows the training examples closely.
- **Random Forest.** $T$ decision trees, each trained on a random sample of the examples with a random subset of the features in every node; the class is chosen by the vote of the trees.

**Metrics.** For class $k$: $TP$ — images of class $k$ recognised as $k$, $FP$ — images of other classes recognised as $k$, $FN$ — images of class $k$ recognised as another class.
$$\text{precision}_k = \frac{TP}{TP + FP},\qquad \text{recall}_k = \frac{TP}{TP + FN},\qquad F1_k = \frac{2\,\text{precision}_k\,\text{recall}_k}{\text{precision}_k + \text{recall}_k},\qquad \text{accuracy} = \frac{\text{correct}}{\text{all}}.$$
The macro average is the mean over the classes: every class counts equally, regardless of its number of images. The confusion matrix shows in row $k$ the shares of the images of class $k$ recognised as every class (the diagonal is the recall).

**Parameters.**
- `LEVEL` — 24 types or 7 groups of the course; `N_PER_CLASS` — the largest number of training images of every type (400 — practically all).
- `CLASSIFIER`, `C` (SVM), `N_TREES` (Random Forest). HOG: 64 × 64 px, cells 8 × 8, blocks 2 × 2, 9 orientations (1764 values).

**Expected result.** The table of the metrics, the metrics of every class, the confusion matrix, the most frequent confusions and examples of errors.

**What to observe.** With 150 images per type the SVM with the RBF kernel reaches an accuracy of 0.18 on the 24 types (random choice: 0.04) and 0.39 on the 7 groups (random choice: 0.14); the linear SVM 0.14 (with C = 0.01). The most frequent confusions are between vehicles with a similar silhouette: BTR-60 → BTR-70, BM-30 → Iskander, RS-24 → BM-30, Iskander → RS-24 (long wheeled chassis with a launcher); on the 7 groups IFV → Tank and APC/recon → Tank. The HOG of a thumbnail reduced to 64 × 64 px does not contain the details that distinguish the types. The difference between the accuracy and the macro F1 points to unequal classes.

**Control question 5.** A classifier recognises 90 of 100 tanks; of 120 images recognised as tanks 30 are not tanks. What are the precision, the recall and the F1 for the class "tank"?
""")

code(r"""
#@title 4.1.5 HOG + SVM / Random Forest: training and metrics { display-mode: "form" }
LEVEL_41 = "24 types" #@param ["24 types", "7 groups of the course"]
N_PER_CLASS = 150 #@param {type:"slider", min:25, max:400, step:25}
CLASSIFIER = "RBF SVM" #@param ["RBF SVM", "linear SVM", "Random Forest"]
C = 10 #@param [0.01, 0.1, 1, 10, 100] {type:"raw"}
N_TREES = 300 #@param {type:"slider", min:50, max:500, step:50}

import warnings
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

TRAIN_IDX, TEST_IDX = mv.clean(near=2)          # without exact copies and near duplicates (block 4.1.2)
if "HOG_ALL" not in globals():
    t0 = time.time()
    HOG_ALL = aitbc.hog_features([mv.image(i, size=64, gray=True) for i in range(len(mv.table))], size=64)
    print(f"HOG of {len(mv.table)} images: {HOG_ALL.shape[1]} values, {time.time() - t0:.0f} s")
lab_col = "cls" if LEVEL_41 == "24 types" else "group"
tr = mv.subset(TRAIN_IDX, N_PER_CLASS, seed=0)
y_tr, y_te = mv.table[lab_col].values[tr], mv.table[lab_col].values[TEST_IDX]
if CLASSIFIER == "RBF SVM":
    model = make_pipeline(StandardScaler(), SVC(C=C, gamma="scale"))
elif CLASSIFIER == "linear SVM":
    model = make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", alpha=1.0 / (C * len(tr)), max_iter=100, tol=1e-4, random_state=0))
else:
    model = RandomForestClassifier(N_TREES, n_jobs=-1, random_state=0)
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    model.fit(HOG_ALL[tr], y_tr)
t_fit = time.time() - t0
t0 = time.time()
pred = model.predict(HOG_ALL[TEST_IDX])
t_pred = (time.time() - t0) / len(TEST_IDX) * 1000
labels = sorted(set(y_te) | set(y_tr)) if lab_col == "cls" else list(aitbc.MilitaryVehicles.GROUP_ORDER)
P, R, F1, S = precision_recall_fscore_support(y_te, pred, labels=labels, zero_division=0)
acc = accuracy_score(y_te, pred)
summary = pd.DataFrame({"value": [f"{len(tr)} ({N_PER_CLASS} per type at most)", len(TEST_IDX), f"{1 / len(labels):.3f}",
                                  f"{acc:.3f}", f"{P.mean():.3f}", f"{R.mean():.3f}", f"{F1.mean():.3f}",
                                  f"{t_fit:.1f} s", f"{t_pred:.2f} ms"]},
                       index=["training images", "test images (cleaned)", "accuracy of a random choice", "accuracy",
                              "precision (macro)", "recall (macro)", "F1 (macro)", "time of training", "time of recognition of one image"])
display(summary.style.set_caption(f"HOG + {CLASSIFIER}, {LEVEL_41}"))
per_class = pd.DataFrame({"precision": P, "recall": R, "F1": F1, "test images": S}, index=labels).round(3)
display(per_class.sort_values("F1", ascending=False).style.background_gradient(subset=["F1"], cmap="RdYlGn", vmin=0, vmax=1)
        .format(precision=3).set_caption("Metrics of every class"))
cm = confusion_matrix(y_te, pred, labels=labels)
cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(0.42 * len(labels) + 4, 0.38 * len(labels) + 3))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("recognised class")
ax.set_ylabel("true class")
if len(labels) <= 8:
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if cmn[i, j] > 0.5 else "k")
plt.colorbar(im, fraction=0.03)
ax.set_title(f"Confusion matrix (shares of the true class), accuracy {acc:.3f}", fontsize=9.5)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"lesson41_confusion_{'types' if lab_col == 'cls' else 'groups'}.png"), dpi=80)
plt.show()
off = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j]
display(pd.DataFrame(sorted(off, reverse=True)[:8], columns=["test images", "true class", "recognised as"]))
wrong = np.where(pred != y_te)[0][:8]
fig, ax = plt.subplots(1, len(wrong), figsize=(2.2 * len(wrong), 2.4))
for a, k in zip(np.ravel(ax), wrong):
    a.imshow(mv.image(TEST_IDX[k]))
    a.set_title(f"true: {y_te[k]}\nrecognised: {pred[k]}", fontsize=7.5)
    a.axis("off")
plt.tight_layout()
plt.show()
RESULTS_41 = globals().get("RESULTS_41", {})
RESULTS_41[(LEVEL_41, CLASSIFIER, N_PER_CLASS)] = dict(accuracy=round(acc, 3), f1_macro=round(float(F1.mean()), 3))
per_class.to_csv(os.path.join(REPORT_DIR, f"lesson41_metrics_{CLASSIFIER.replace(' ', '_')}_{'types' if lab_col == 'cls' else 'groups'}.csv"))
update_project_json("4.1", {"lesson_4_1": dict(level=LEVEL_41, classifier=CLASSIFIER, n_per_class=N_PER_CLASS, C=C,
                                               accuracy=round(acc, 3), f1_macro=round(float(F1.mean()), 3))})
""", form=True)

md(r"""
### Block 4.1.6. Division of the data set: random frames or whole sequences

**Purpose.** To measure how the estimation of the quality depends on the way the frames of video sequences are divided into the training and the test part.

**Experiment.** From every one of `N_SEQUENCES` Italian sequences of SULAND a run of `FRAMES_PER_SEQUENCE` consecutive frames with the most mines is taken. From every frame the fragments around the mines (a square of 1.5 × the longer side of the box) and two fragments of the same size without mines are cut and reduced to 64 × 64 px; the classifier works on their HOG vectors. The quality is estimated by cross-validation with `N_FOLDS` folds in two ways:
- **random frames** — every frame goes into a random fold; the neighbouring frames of the same sequence (the same mine, the same background, the same light) are both in the training and in the test part;
- **whole sequences** — all frames of a sequence go into the same fold; the test part contains only places and mines not seen during the training, as in real use.

**Parameters.** `N_SEQUENCES`, `FRAMES_PER_SEQUENCE` (frames of about 0.5 MB are downloaded once), `TASK` — mine or background / PFM-1 or PMA-2, `CLASSIFIER`, `N_FOLDS`.

**Expected result.** The table of the accuracy and of the macro F1 for the two divisions and examples of the fragments.

**What to observe.** With 12 sequences of 30 frames (1099 fragments: 720 background, 306 PMA-2, 73 PFM-1) the SVM with the RBF kernel reaches an accuracy of 0.98 by random frames and 0.83 by whole sequences in the task "mine or background"; for "PFM-1 or PMA-2" the macro F1 falls from 0.98 to 0.46. The division by random frames overestimates the quality: the classifier recognises the place, not the mine. For this reason the authors of SULAND give the parts as whole sequences, and the test of a model is made on sequences (flights, places, days) not used in the training.

**Control question 6.** A data set contains frames of 20 reconnaissance flights over 5 areas. How should the training, validation and test parts be formed, and why?
""")

code(r"""
#@title 4.1.6 Random frames or whole sequences: cross-validation on SULAND fragments { display-mode: "form" }
N_SEQUENCES = 12 #@param {type:"slider", min:4, max:20, step:1}
FRAMES_PER_SEQUENCE = 30 #@param {type:"slider", min:10, max:40, step:5}
TASK = "mine or background" #@param ["mine or background", "PFM-1 or PMA-2"]
CLASSIFIER_46 = "RBF SVM" #@param ["RBF SVM", "linear SVM", "Random Forest"]
N_FOLDS = 4 #@param {type:"slider", min:3, max:6, step:1}

import warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

seq_tab = suland.sequences()
SEQ_POOL = seq_tab[(seq_tab.domain == "data-iid") & (seq_tab.pfm1 > 40) & (seq_tab.pma2 > 40)].sort_values("boxes", ascending=False).seq.tolist()
key = (N_SEQUENCES, FRAMES_PER_SEQUENCE)
CROPS = globals().get("CROPS", {})
if key not in CROPS:
    t0 = time.time()
    CROPS[key] = suland.crop_set(SEQ_POOL[:N_SEQUENCES], FRAMES_PER_SEQUENCE, size=64, seed=0)
    print(f"{len(CROPS[key])} fragments of {CROPS[key].groupby(['seq', 'frame']).ngroups} frames: {time.time() - t0:.0f} s")
crops = CROPS[key]
mask = (crops.label != "background").values if TASK == "PFM-1 or PMA-2" else np.ones(len(crops), bool)
cr = crops[mask].reset_index(drop=True)
y = (cr.label != "background").astype(int).values if TASK == "mine or background" else (cr.label == "PMA-2").astype(int).values
X = aitbc.hog_features(list(cr.img), size=64)
print("Fragments by class:", cr.label.value_counts().to_dict())


def make_model():
    if CLASSIFIER_46 == "RBF SVM":
        return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    if CLASSIFIER_46 == "linear SVM":
        return make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", alpha=1e-3, max_iter=200, tol=1e-4, random_state=0))
    return RandomForestClassifier(300, n_jobs=-1, random_state=0)


n_folds = min(N_FOLDS, cr.seq.nunique())            # a fold must contain at least one whole sequence
if n_folds < N_FOLDS:
    print(f"N_FOLDS = {N_FOLDS} is larger than the number of sequences: {n_folds} folds are used")
rows = {}
for name, groups in (("random frames", (cr.seq + "-" + cr.frame.astype(str)).values), ("whole sequences", cr.seq.values)):
    acc, f1 = [], []
    for k_tr, k_te in StratifiedGroupKFold(n_folds, shuffle=True, random_state=0).split(X, y, groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_model().fit(X[k_tr], y[k_tr])
        p = m.predict(X[k_te])
        acc.append(accuracy_score(y[k_te], p))
        f1.append(f1_score(y[k_te], p, average="macro"))
    rows[name] = {"accuracy": np.mean(acc), "± (folds)": np.std(acc), "F1 (macro)": np.mean(f1)}
LEAK = pd.DataFrame(rows).T
display(LEAK.style.format(precision=3).set_caption(f"{TASK}, HOG + {CLASSIFIER_46}, {N_SEQUENCES} sequences × {FRAMES_PER_SEQUENCE} frames, {n_folds} folds"))
fig, ax = plt.subplots(1, 2, figsize=(15, 3.2), gridspec_kw={"width_ratios": [1, 3]})
LEAK[["accuracy", "F1 (macro)"]].plot.bar(ax=ax[0], rot=0, ylim=(0, 1))
ax[0].set_title("estimated quality", fontsize=9)
show = pd.concat([crops[crops.label == c].head(6) for c in ("PFM-1", "PMA-2", "background")])
tiles = [np.pad(i, 2, constant_values=255) for i in show.img]
ax[1].imshow(np.hstack(tiles), cmap="gray")
ax[1].set_title("fragments 64 × 64 px: 6 PFM-1, 6 PMA-2, 6 background", fontsize=9)
ax[1].axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson41_division.png"), dpi=80)
plt.show()
LEAK.to_csv(os.path.join(REPORT_DIR, "lesson41_division_of_data_set.csv"))
""", form=True)

# =====================================================================================
md(r"""
---
# Lesson 4.2 (Group class). Neural networks: from the neuron to MobileNet, transfer learning and ONNX

**Learning questions**
1. The artificial neuron and the fully connected neural network.
2. The convolutional neural network (CNN): convolution, feature maps, pooling.
3. MobileNet: an architecture for onboard computers of UAVs and ground robots.
4. Transfer learning: the use of a network trained on a large data set for a new task.
5. Export of the trained model to the ONNX format for the onboard computer.

**Order of work of the group.** Every block is run with the default parameters, the result is discussed by the group, then every subgroup changes one parameter (the subgroups take different values) and reports its measurement. The answers to the control questions and the measurements are written in the **Report of Lesson 4.2** at the end of the lesson.
""")

md(r"""
### Block 4.2.1. The artificial neuron

**Purpose.** To study the computation of a neuron and the training of its weights by gradient descent on real data: the fragments of SULAND frames with a mine and without a mine (block 4.1.6).

**Formulas.**
- Neuron: $z = w_1 x_1 + w_2 x_2 + b$, output $y = \varphi(z)$; activation functions: sigmoid $\sigma(z) = 1 / (1 + e^{-z})$, $\tanh z$, ReLU $\max(0, z)$, step $[z > 0]$. The decision "mine" is $z > 0$: the boundary between the classes is the straight line $w_1 x_1 + w_2 x_2 + b = 0$.
- Training of a neuron with the sigmoid (logistic regression) minimises the cross-entropy $L = -\frac{1}{n}\sum_i [t_i \ln y_i + (1 - t_i) \ln (1 - y_i)]$, $t_i \in \{0, 1\}$; its gradient is $\partial L / \partial w_j = \frac{1}{n}\sum_i (y_i - t_i)\, x_{ij}$, $\partial L / \partial b = \frac{1}{n}\sum_i (y_i - t_i)$, and one step of gradient descent is $w \leftarrow w - \eta\, \partial L / \partial w$.
- Inputs: the HOG vector of 1764 values of a fragment is reduced to two numbers $x_1, x_2$ by the principal component analysis (the two directions of the largest variance of the HOG vectors), scaled to zero mean and unit variance.

**Parameters.** `W1`, `W2`, `B` — the weights set by hand; `ACTIVATION`; `TRAIN` — train the weights by gradient descent from zero with `LEARNING_RATE` $\eta$ during `EPOCHS` steps.

**Expected result.** The diagram of the neuron, the fragments on the plane $(x_1, x_2)$ with the boundary of the neuron set by hand (dashed) and of the trained neuron (solid), the accuracy of both and the curve of the loss.

**What to observe.** One neuron separates the classes by a straight line: with two inputs it recognises the mines with an accuracy of about 0.8, the overlapping region remains. A too large $\eta$ makes the loss oscillate, a too small one requires many steps. The activation function does not move the boundary $z = 0$; it changes the output value and its gradient.

**Control question 7.** How many weights has one neuron with the full HOG vector of 1764 values at the input, and how does its boundary between the classes look in this space?
""")

code(r"""
#@title 4.2.1 Neuron with two inputs on SULAND fragments { display-mode: "form" }
W1 = 1.0 #@param {type:"slider", min:-3, max:3, step:0.1}
W2 = -1.0 #@param {type:"slider", min:-3, max:3, step:0.1}
B = 0.0 #@param {type:"slider", min:-3, max:3, step:0.1}
ACTIVATION = "sigmoid" #@param ["sigmoid", "tanh", "ReLU", "step"]
TRAIN = True #@param {type:"boolean"}
LEARNING_RATE = 0.5 #@param [0.01, 0.1, 0.5, 2.0, 8.0] {type:"raw"}
EPOCHS = 200 #@param {type:"slider", min:10, max:1000, step:10}

from sklearn.decomposition import PCA
CROPS = globals().get("CROPS", {})
if not CROPS:
    seq_tab = suland.sequences()
    pool = seq_tab[(seq_tab.domain == "data-iid") & (seq_tab.pfm1 > 40) & (seq_tab.pma2 > 40)].sort_values("boxes", ascending=False).seq.tolist()
    CROPS[(12, 30)] = suland.crop_set(pool[:12], 30, size=64, seed=0)
crops_n = CROPS[(12, 30)] if (12, 30) in CROPS else list(CROPS.values())[0]
Xh = aitbc.hog_features(list(crops_n.img), size=64)
Z2 = PCA(2, random_state=0).fit_transform(Xh)
Z2 = (Z2 - Z2.mean(0)) / Z2.std(0)
t = (crops_n.label != "background").astype(int).values
act = {"sigmoid": lambda z: 1 / (1 + np.exp(-z)), "tanh": np.tanh, "ReLU": lambda z: np.maximum(z, 0), "step": lambda z: (z > 0).astype(float)}[ACTIVATION]
acc_hand = float(((Z2 @ np.array([W1, W2]) + B > 0).astype(int) == t).mean())
w, b, losses = np.zeros(2), 0.0, []
if TRAIN:
    for _ in range(EPOCHS):
        yv = 1 / (1 + np.exp(-(Z2 @ w + b)))
        losses.append(float(-np.mean(t * np.log(yv + 1e-9) + (1 - t) * np.log(1 - yv + 1e-9))))
        w -= LEARNING_RATE * Z2.T @ (yv - t) / len(t)
        b -= LEARNING_RATE * float(np.mean(yv - t))
acc_tr = float(((Z2 @ w + b > 0).astype(int) == t).mean()) if TRAIN else float("nan")
fig, ax = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
aitbc.draw_neuron(ax[0], inputs=("x₁", "x₂"), weights=(f"w₁={W1:g}", f"w₂={W2:g}"), bias=f"b={B:g}", activation=ACTIVATION)
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
if TRAIN:
    ax[1].contourf(xx, yy, 1 / (1 + np.exp(-(w[0] * xx + w[1] * yy + b))), levels=20, cmap="RdBu_r", alpha=0.25)
for lab, c, mk in (("background", "#1565c0", "o"), ("PFM-1", "#c62828", "^"), ("PMA-2", "#ef6c00", "s")):
    m_ = (crops_n.label == lab).values
    ax[1].scatter(Z2[m_, 0], Z2[m_, 1], s=9, c=c, marker=mk, label=lab, alpha=0.6)
xs = np.linspace(-3, 3, 2)
if abs(W2) > 1e-9:
    ax[1].plot(xs, -(W1 * xs + B) / W2, "k--", lw=1.4, label=f"by hand: accuracy {acc_hand:.3f}")
if TRAIN and abs(w[1]) > 1e-9:
    ax[1].plot(xs, -(w[0] * xs + b) / w[1], "k-", lw=2, label=f"trained: accuracy {acc_tr:.3f}")
ax[1].set_xlim(-3, 3)
ax[1].set_ylim(-3, 3)
ax[1].set_xlabel("x₁ (1st principal component of HOG)")
ax[1].set_ylabel("x₂ (2nd principal component)")
ax[1].legend(fontsize=7.5, loc="lower right")
ax[2].plot(losses)
ax[2].set_xlabel("step of gradient descent")
ax[2].set_ylabel("cross-entropy L")
ax[2].set_title(f"η = {LEARNING_RATE:g}: w₁ = {w[0]:.2f}, w₂ = {w[1]:.2f}, b = {b:.2f}" if TRAIN else "training switched off", fontsize=9)
ax[2].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson42_neuron.png"), dpi=80)
plt.show()
print(f"{len(t)} fragments ({int(t.sum())} with a mine). Accuracy of the neuron set by hand: {acc_hand:.3f}"
      + (f"; trained: {acc_tr:.3f}, loss {losses[0]:.3f} → {losses[-1]:.3f}" if TRAIN else ""))
""", form=True)

md(r"""
### Block 4.2.2. The fully connected network

**Purpose.** To train a fully connected (multilayer) network on the HOG vectors of military vehicles and to compare it with the SVM of block 4.1.5.

**Formulas.** Layer $l$: $h^{(l)} = \varphi\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)$, $\varphi$ — ReLU; output layer: $p_k = e^{z_k} / \sum_j e^{z_j}$ (softmax, the probabilities of the $K$ classes). Number of parameters of a layer with $n_{in}$ inputs and $n_{out}$ neurons: $(n_{in} + 1)\, n_{out}$. The weights are trained by the Adam method on the cross-entropy $-\ln p_{\text{true class}}$, by mini-batches of 64 vectors; one epoch is one pass over the training part. Dropout switches off a random share of the neurons in every step of the training and reduces the fitting to the training examples.

**Parameters.** `LEVEL_42`, `N_PER_CLASS_42` — as in block 4.1.5; `HIDDEN` — the numbers of neurons of the hidden layers; `EPOCHS`, `LEARNING_RATE_42`, `DROPOUT`.

**Expected result.** The graph of the network with the sizes of its layers, the number of parameters by the formula and in the model, the curves of the accuracy on the training and the test part and the final metrics.

**What to observe.** The network 1764 → 256 → 64 → 7 has 468 743 parameters for 3569 training images: the accuracy on the training part reaches 0.999 after about 15 epochs, the test accuracy stays at about 0.37, slightly below the SVM of block 4.1.5 (0.39 on the 7 groups). The gap between the two curves is overfitting; dropout and fewer neurons reduce it. With HOG at the input the network cannot obtain more than the features contain.

**Control question 8.** Compute the number of parameters of the network 1764 → 128 → 7 and compare it with the number of training images.
""")

code(r"""
#@title 4.2.2 Fully connected network on HOG vectors { display-mode: "form" }
LEVEL_42 = "7 groups of the course" #@param ["7 groups of the course", "24 types"]
N_PER_CLASS_42 = 150 #@param {type:"slider", min:25, max:400, step:25}
HIDDEN = "256-64" #@param ["64", "128", "256-64", "512-128"]
EPOCHS_42 = 30 #@param {type:"slider", min:5, max:80, step:5}
LEARNING_RATE_42 = 0.001 #@param [0.0003, 0.001, 0.003, 0.01] {type:"raw"}
DROPOUT = 0.3 #@param {type:"slider", min:0, max:0.6, step:0.1}

import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
TRAIN_IDX, TEST_IDX = mv.clean(near=2)          # without exact copies and near duplicates (block 4.1.2)
if "HOG_ALL" not in globals():
    HOG_ALL = aitbc.hog_features([mv.image(i, size=64, gray=True) for i in range(len(mv.table))], size=64)
lab_col = "cls" if LEVEL_42 == "24 types" else "group"
labels = sorted(mv.table.cls.unique()) if lab_col == "cls" else list(aitbc.MilitaryVehicles.GROUP_ORDER)
tr = mv.subset(TRAIN_IDX, N_PER_CLASS_42, seed=0)
mu, sd = HOG_ALL[tr].mean(0), HOG_ALL[tr].std(0) + 1e-6
Xtr = torch.tensor((HOG_ALL[tr] - mu) / sd)
Xte = torch.tensor((HOG_ALL[TEST_IDX] - mu) / sd)
ytr = torch.tensor([labels.index(v) for v in mv.table[lab_col].values[tr]])
yte = np.array([labels.index(v) for v in mv.table[lab_col].values[TEST_IDX]])
sizes = [Xtr.shape[1]] + [int(v) for v in HIDDEN.split("-")] + [len(labels)]
layers = []
for a, b_ in zip(sizes[:-2], sizes[1:-1]):
    layers += [nn.Linear(a, b_), nn.ReLU(), nn.Dropout(DROPOUT)]
layers.append(nn.Linear(sizes[-2], sizes[-1]))
torch.manual_seed(0)
net = nn.Sequential(*layers)
opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE_42)
lossf = nn.CrossEntropyLoss()
hist = []
t0 = time.time()
for ep in range(EPOCHS_42):
    net.train()
    perm = torch.randperm(len(Xtr))
    for s in range(0, len(Xtr), 64):
        bi = perm[s:s + 64]
        opt.zero_grad()
        loss = lossf(net(Xtr[bi]), ytr[bi])
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        acc_tr_ = (net(Xtr).argmax(1) == ytr).float().mean().item()
        p_te = net(Xte).argmax(1).numpy()
    hist.append((acc_tr_, accuracy_score(yte, p_te)))
t_train = time.time() - t0
n_formula = sum((a + 1) * b_ for a, b_ in zip(sizes[:-1], sizes[1:]))
n_model = aitbc.count_parameters(net)[0]
fig, ax = plt.subplots(1, 2, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
names = ["HOG vector"] + [f"hidden {k + 1}, ReLU" for k in range(len(sizes) - 2)] + ["output, softmax"]
aitbc.draw_mlp(ax[0], sizes, names=names, output_names=labels if len(labels) <= 7 else None)
hist_a = np.array(hist)
ax[1].plot(range(1, EPOCHS_42 + 1), hist_a[:, 0], label="training part")
ax[1].plot(range(1, EPOCHS_42 + 1), hist_a[:, 1], label="test part")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("accuracy")
ax[1].set_ylim(0, 1.02)
ax[1].grid(alpha=0.3)
ax[1].legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson42_fully_connected.png"), dpi=80)
plt.show()
f1m = f1_score(yte, p_te, average="macro")
rows = {"layers": " → ".join(map(str, sizes)), "parameters by the formula Σ(n_in + 1)·n_out": n_formula,
        "parameters in the model": n_model, "training images": len(tr),
        "accuracy: training / test part": f"{hist_a[-1, 0]:.3f} / {hist_a[-1, 1]:.3f}", "F1 (macro), test": f"{f1m:.3f}",
        "time of training": f"{t_train:.1f} s"}
ref = {k: v for k, v in globals().get("RESULTS_41", {}).items() if k[0] == LEVEL_42 and k[2] == N_PER_CLASS_42}
for k, v in ref.items():
    rows[f"HOG + {k[1]} (block 4.1.5): accuracy / F1"] = f"{v['accuracy']:.3f} / {v['f1_macro']:.3f}"
display(pd.DataFrame({"value": list(rows.values())}, index=list(rows)))
MLP_RESULT = dict(level=LEVEL_42, layers=sizes, accuracy=round(float(hist_a[-1, 1]), 3), f1_macro=round(float(f1m), 3))
""", form=True)

md(r"""
### Block 4.2.3. The convolutional network: convolution and feature maps

**Purpose.** To see how a convolutional network trained on ImageNet (MobileNetV3-small, 1.2 million images of 1000 classes) transforms an image of a military vehicle layer by layer.

**Formulas.**
- Convolution: the value of the output map $c$ in the point $(x, y)$ is $a_c(x, y) = \varphi\left(\sum_{c'} \sum_{i, j} K_{c, c'}(i, j)\, I_{c'}(x + i, y + j) + b_c\right)$: the same kernel $K$ of $k \times k$ weights is applied in every point. A layer with $C_{in}$ input and $C_{out}$ output maps has $k^2 C_{in} C_{out}$ weights, independent of the size of the image.
- Size of the output map: $W_{out} = \lfloor (W_{in} + 2P - k) / S \rfloor + 1$, $P$ — padding, $S$ — stride; MobileNetV3-small reduces 224 × 224 to 7 × 7 by five layers with $S = 2$.
- Global average pooling turns the last maps (7 × 7 × 576) into a vector of 576 values: the features used in block 4.2.5.

**Parameters.** `TYPE` — the type of the vehicle; `IMAGE_NUMBER`; `LAYER` — the number of the block of the network whose output maps are shown (0 — the first convolution, 12 — the last one); `N_MAPS` — the number of maps.

**Expected result.** The 16 kernels of the first layer, the output maps of the chosen block, the table of the blocks (size of the output, number of parameters), the diagram of the network and the five classes of ImageNet that the network gives for the image.

**What to observe.** The kernels of the first layer are detectors of edges and colour transitions, as the Sobel operator of Topic 3. The maps of the first blocks repeat the contours of the vehicle; the maps of the last blocks have 7 × 7 cells and respond to parts of objects. Among the 1000 classes of ImageNet there are "tank", "half track", "amphibious vehicle", "cannon", "missile", "projectile": the features of ImageNet are close to the task of recognition of military vehicles.

**Control question 9.** What is the size of the output map of a convolution 3 × 3 with stride 2 and padding 1 for an input of 224 × 224, and how many weights has such a layer with 3 input and 16 output maps?
""")

code(r"""
#@title 4.2.3 Kernels and feature maps of MobileNetV3-small { display-mode: "form" }
TYPE = "T-72" #@param ["2S19_MSTA", "30N6E", "BM-30", "BMD", "BMP-1", "BMP-2", "BMP-T15", "BRDM", "BTR-60", "BTR-70", "BTR-80", "D-30", "Iskander", "MT_LB", "Pantsir-S1", "Rs-24", "T-14", "T-62", "T-64", "T-72", "T-80", "T-90", "TOS-1", "Tornado"]
IMAGE_NUMBER = 3 #@param {type:"slider", min:0, max:20, step:1}
LAYER = 1 #@param {type:"slider", min:0, max:12, step:1}
N_MAPS = 16 #@param [8, 16, 32] {type:"raw"}

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
if "MNV3S" not in globals():
    MNV3S_W = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    MNV3S = mobilenet_v3_small(weights=MNV3S_W).eval()
i_img = mv.table.index[(mv.table.cls == TYPE) & (mv.table.split == "train")][IMAGE_NUMBER]
rgb = mv.image(i_img)
x = torch.from_numpy(aitbc.imagenet_batch([rgb], 224))
outs, shapes = {}, []
hooks = [blk.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, o.detach())) for k, blk in enumerate(MNV3S.features)]
with torch.no_grad():
    logits = MNV3S(x)
for h in hooks:
    h.remove()
for k, blk in enumerate(MNV3S.features):
    o = outs[k]
    shapes.append({"block": k, "type": type(blk).__name__, "output H × W × C": f"{o.shape[2]} × {o.shape[3]} × {o.shape[1]}",
                   "parameters": aitbc.count_parameters(blk)[0]})
fig = plt.figure(figsize=(18, 5.2))
ax0 = fig.add_subplot(1, 3, 1)
ax0.imshow(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA))
ax0.set_title(f"{TYPE}: image {i_img}, input 224 × 224 × 3", fontsize=9)
ax0.axis("off")
k1 = MNV3S.features[0][0].weight.detach().numpy()          # 16 x 3 x 3 x 3
k1 = (k1 - k1.min((1, 2, 3), keepdims=True)) / (np.ptp(k1, axis=(1, 2, 3), keepdims=True) + 1e-9)   # every kernel to 0...1
tiles = np.vstack([np.hstack([np.pad(cv2.resize(k1[r * 8 + c].transpose(1, 2, 0), (24, 24), interpolation=cv2.INTER_NEAREST),
                                     ((2, 2), (2, 2), (0, 0)), constant_values=1) for c in range(8)]) for r in range(2)])
ax1 = fig.add_subplot(1, 3, 2)
ax1.imshow(tiles)
ax1.set_title("16 kernels 3 × 3 × 3 of the first convolution (as colour images)", fontsize=9)
ax1.axis("off")
o = outs[LAYER][0].numpy()
order = np.argsort(-o.reshape(o.shape[0], -1).mean(1))[:N_MAPS]
ncol = 8
grid = np.vstack([np.hstack([np.pad(cv2.resize((o[c] - o[c].min()) / (np.ptp(o[c]) + 1e-6), (64, 64), interpolation=cv2.INTER_NEAREST),
                                    2, constant_values=np.nan) for c in order[r * ncol:(r + 1) * ncol]]) for r in range(max(1, len(order) // ncol))])
ax2 = fig.add_subplot(1, 3, 3)
cmap_maps = plt.get_cmap("viridis").copy()
cmap_maps.set_bad("white")
ax2.imshow(np.ma.masked_invalid(grid), cmap=cmap_maps)
ax2.set_title(f"block {LAYER}: {N_MAPS} of {o.shape[0]} maps {o.shape[1]} × {o.shape[2]} with the largest mean", fontsize=9)
ax2.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"lesson42_feature_maps_block{LAYER}.png"), dpi=80)
plt.show()
display(pd.DataFrame(shapes).set_index("block").T)
stages = [("input", 224, 224, 3)] + [(f"block {k}", outs[k].shape[2], outs[k].shape[3], outs[k].shape[1]) for k in (0, 1, 3, 6, 8, 11, 12)] \
    + [("global average pooling", 1, 1, 576), ("classifier ImageNet", 1, 1, 1000)]
fig, ax = plt.subplots(figsize=(18, 3.2))
aitbc.draw_cnn(ax, stages, title=f"MobileNetV3-small: {aitbc.count_parameters(MNV3S)[0]:,} parameters".replace(",", " "))
plt.show()
prob = torch.softmax(logits, 1)[0].numpy()
top = np.argsort(-prob)[:5]
print("Five classes of ImageNet for the image:", "; ".join(f"{MNV3S_W.meta['categories'][j]} {prob[j]:.2f}" for j in top))
""", form=True)

md(r"""
### Block 4.2.4. MobileNet: depthwise separable convolution and the cost of computation

**Purpose.** To compare the number of parameters, the number of operations and the time of inference of MobileNet with other convolutional networks, and to explain the source of the difference.

**Formulas.** A standard convolution $k \times k$ of $C_{in}$ into $C_{out}$ maps of size $H \times W$ needs $k^2 C_{in} C_{out}$ weights and $k^2 C_{in} C_{out} H W$ multiply-accumulate operations (MAC). The **depthwise separable** convolution replaces it by a depthwise convolution $k \times k$ (one kernel per map: $k^2 C_{in}$ weights) and a pointwise convolution $1 \times 1$ ($C_{in} C_{out}$ weights):
$$\frac{k^2 C_{in} + C_{in} C_{out}}{k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2} \approx \frac{1}{9}\ \text{for}\ k = 3.$$
The **inverted residual block** of MobileNet V2/V3 expands the maps by a convolution $1 \times 1$, filters them by a depthwise convolution and projects them back by a convolution $1 \times 1$; when the input and the output have the same size, the input is added to the output (residual connection).

**Parameters.** `INPUT_SIZE` — the size of the input image; `BATCH` — the number of images processed together; `N_RUNS` — the number of repetitions of the measurement of the time.

**Expected result.** The diagram of a real block of MobileNetV3-small with the numbers of weights, the table of five networks (parameters, size of the file of the weights, MAC, time per image on the device of the session) and the dependence of the time on the size of the input.

**What to observe.** MobileNetV3-small has 2.5 million parameters and 56 million MAC per image of 224 × 224, ResNet-50 25.6 million parameters and 4.1 billion MAC: 72 times fewer operations at a lower, but comparable accuracy on ImageNet (67.7 % against 80.9 %). The time does not fall in proportion to the MAC (on a processor of 16 cores 8 ms against 49 ms, 6 times): the depthwise convolutions use the processor less efficiently than the standard ones. The number of operations grows with the square of the size of the input.

**Control question 10.** By how many times does a depthwise separable convolution 3 × 3 reduce the number of weights of a layer with 96 input and 96 output maps?
""")

code(r"""
#@title 4.2.4 Parameters, operations and time of inference of five networks { display-mode: "form" }
INPUT_SIZE = 224 #@param [128, 160, 224, 320] {type:"raw"}
BATCH = 1 #@param [1, 8, 32] {type:"raw"}
N_RUNS = 5 #@param {type:"slider", min:2, max:20, step:1}

import torch.nn as nn
from torchvision import models as tvm


def count_macs(model, size):
    # multiply-accumulate operations of the convolution and linear layers for one image size x size
    macs = []

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            macs.append(o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, nn.Linear):
            macs.append(m.in_features * m.out_features)
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in hs:
        h.remove()
    return sum(macs)


def time_per_image(model, size, batch, runs):
    x = torch.randn(batch, 3, size, size, device=DEVICE)
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / runs / batch * 1000


NETS = {"MobileNetV3-small": tvm.mobilenet_v3_small, "MobileNetV3-large": tvm.mobilenet_v3_large,
        "MobileNetV2": tvm.mobilenet_v2, "ResNet-18": tvm.resnet18, "ResNet-50": tvm.resnet50}
TOP1 = {"MobileNetV3-small": 67.7, "MobileNetV3-large": 75.3, "MobileNetV2": 72.2, "ResNet-18": 69.8, "ResNet-50": 80.9}
rows = []
for name, ctor in NETS.items():
    m = ctor(weights=None).eval()          # architecture only: the weights are not needed to count and to time
    n = aitbc.count_parameters(m)[0]
    rows.append({"network": name, "parameters, million": n / 1e6, "weights float32, MB": n * 4 / 2 ** 20,
                 f"MAC per image {INPUT_SIZE}², million": count_macs(m, INPUT_SIZE) / 1e6,
                 f"time per image, ms ({DEVICE}, batch {BATCH})": time_per_image(m, INPUT_SIZE, BATCH, N_RUNS),
                 "top-1 ImageNet, % (torchvision)": TOP1[name]})
COST = pd.DataFrame(rows).set_index("network")
display(COST.style.format(precision=1).set_caption(f"Input {INPUT_SIZE} × {INPUT_SIZE}; {os.cpu_count()} logical cores" if DEVICE == "cpu" else f"Input {INPUT_SIZE} × {INPUT_SIZE}; GPU"))
small = tvm.mobilenet_v3_small(weights=None)
blk = small.features[3].block                     # inverted residual block 24 -> 88 -> 24 of MobileNetV3-small
c_in, c_exp, c_out = blk[0][0].in_channels, blk[0][0].out_channels, blk[2][0].out_channels
k_dw = blk[1][0].kernel_size[0]
fig, ax = plt.subplots(1, 2, figsize=(18, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
aitbc.draw_mobilenet_block(ax[0], c_in, c_exp, c_out, k_dw, se=False, stride=blk[1][0].stride[0])
ax[0].set_title(f"block 3 of MobileNetV3-small: {aitbc.count_parameters(small.features[3])[0]:,} parameters (with batch normalisation)".replace(",", " "), fontsize=9)
sizes_ = [96, 128, 160, 224, 320]
ax[1].plot(sizes_, [count_macs(small, s) / 1e6 for s in sizes_], "o-")
ax[1].set_xlabel("size of the input, px")
ax[1].set_ylabel("MAC per image, million")
ax[1].set_title("MobileNetV3-small: operations and the size of the input", fontsize=9)
ax[1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson42_mobilenet_cost.png"), dpi=80)
plt.show()
def spaced(n):
    return f"{n:,}".replace(",", " ")


for C_ in (16, 96, 576):
    print(f"3×3 convolution {C_} → {C_} maps: standard {spaced(aitbc.conv_params(3, C_, C_))} weights; depthwise separable "
          f"{spaced(aitbc.conv_params(3, C_, C_, True))} (ratio {aitbc.conv_params(3, C_, C_) / aitbc.conv_params(3, C_, C_, True):.1f})")
COST.to_csv(os.path.join(REPORT_DIR, "lesson42_cost_of_networks.csv"))
""", form=True)

md(r"""
### Block 4.2.5. Transfer learning: features of a network trained on ImageNet

**Purpose.** To recognise military vehicles with the features of MobileNetV3 trained on ImageNet and to compare the result with HOG + SVM on the same training images.

**Method.** The convolutional part of the network with the weights of ImageNet is **frozen** (its weights are not changed): every image (224 × 224, normalised by the mean and the standard deviation of ImageNet) is turned into a vector of features of 576 (small) or 960 (large) values. On these vectors a single linear layer with softmax is trained (multinomial logistic regression, `C` — the inverse strength of the regularisation). This is the first stage of transfer learning; in Notebook 07 (Lesson 4.4) the last blocks of the network are also trained (fine-tuning).

**Parameters.** `LEVEL_45`, `N_PER_CLASS_45` — as in block 4.1.5; `BACKBONE`; `INPUT_SIZE_45`; `C_LOGISTIC`; `LEARNING_CURVE` — repeat the training for 25…400 images per type (the features of all 7751 training images are computed: 2–6 minutes on the processor).

**Expected result.** The table *HOG + SVM — CNN features + linear layer* on the same images, the metrics of the classes, the confusion matrix of the groups and, optionally, the learning curve.

**What to observe.** With 150 images per type the features of MobileNetV3-small give an accuracy of 0.42 on the 24 types and 0.63 on the 7 groups, against 0.18 and 0.39 for HOG + SVM on the same images; with all training images 0.45 and 0.67. The large network gives practically the same result: the limit is set by the data (small images, contradictory labels, renderings), not by the size of the network. For comparison, the authors of the data set report 54 % on the 24 types for the network ViT-B/16.

**Control question 11.** Why do the features of ImageNet, where there is no class "BTR-80", help to recognise a BTR-80, and in which conditions does this help stop?
""")

code(r"""
#@title 4.2.5 Frozen MobileNetV3 + trained linear layer { display-mode: "form" }
LEVEL_45 = "24 types" #@param ["24 types", "7 groups of the course"]
N_PER_CLASS_45 = 150 #@param {type:"slider", min:25, max:400, step:25}
BACKBONE = "MobileNetV3-small" #@param ["MobileNetV3-small", "MobileNetV3-large"]
INPUT_SIZE_45 = 224 #@param [128, 160, 224] {type:"raw"}
C_LOGISTIC = 0.05 #@param [0.01, 0.05, 0.2, 1.0] {type:"raw"}
LEARNING_CURVE = False #@param {type:"boolean"}

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BACKBONES = globals().get("BACKBONES", {})
if BACKBONE not in BACKBONES:
    BACKBONES[BACKBONE] = (mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1) if BACKBONE.endswith("small")
                           else mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)).eval()
backbone = BACKBONES[BACKBONE]
FEATS = globals().get("FEATS", {})
fkey = (BACKBONE, INPUT_SIZE_45)
FEATS.setdefault(fkey, {})


def features_of(idx):
    # CNN features of the images idx (computed once, kept in FEATS)
    need = [i for i in idx if i not in FEATS[fkey]]
    if need:
        t0 = time.time()
        f = aitbc.cnn_features(backbone, [mv.image(i) for i in need], size=INPUT_SIZE_45, batch=64, device=DEVICE)
        FEATS[fkey].update(zip(need, f))
        print(f"features of {len(need)} images ({f.shape[1]} values): {time.time() - t0:.0f} s on {DEVICE}")
    return np.stack([FEATS[fkey][i] for i in idx])


TRAIN_IDX, TEST_IDX = mv.clean(near=2)          # without exact copies and near duplicates (block 4.1.2)
lab_col = "cls" if LEVEL_45 == "24 types" else "group"
labels = sorted(mv.table.cls.unique()) if lab_col == "cls" else list(aitbc.MilitaryVehicles.GROUP_ORDER)
tr = mv.subset(TRAIN_IDX, N_PER_CLASS_45, seed=0)
F_tr, F_te = features_of(tr), features_of(TEST_IDX)
y_tr, y_te = mv.table[lab_col].values[tr], mv.table[lab_col].values[TEST_IDX]
t0 = time.time()
HEAD = make_pipeline(StandardScaler(), LogisticRegression(C=C_LOGISTIC, max_iter=3000)).fit(F_tr, y_tr)
t_fit = time.time() - t0
pred = HEAD.predict(F_te)
P, R, F1, S = precision_recall_fscore_support(y_te, pred, labels=labels, zero_division=0)
acc = accuracy_score(y_te, pred)
if "HOG_ALL" not in globals():
    HOG_ALL = aitbc.hog_features([mv.image(i, size=64, gray=True) for i in range(len(mv.table))], size=64)
svm = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(HOG_ALL[tr], y_tr)
p_svm = svm.predict(HOG_ALL[TEST_IDX])
_, _, F1s, _ = precision_recall_fscore_support(y_te, p_svm, labels=labels, zero_division=0)
cmpt = pd.DataFrame({"features": ["HOG 64 × 64 (1764 values)", f"{BACKBONE}, ImageNet, frozen ({F_tr.shape[1]} values)"],
                     "classifier": ["RBF SVM, C = 10", f"linear layer + softmax, C = {C_LOGISTIC}"],
                     "accuracy": [accuracy_score(y_te, p_svm), acc], "F1 (macro)": [F1s.mean(), F1.mean()]}).set_index("features")
display(cmpt.style.format(precision=3).set_caption(f"{LEVEL_45}: {len(tr)} training images ({N_PER_CLASS_45} per type at most), "
                                                    f"{len(TEST_IDX)} test images; training of the linear layer {t_fit:.1f} s"))
per_class = pd.DataFrame({"precision": P, "recall": R, "F1": F1, "test images": S}, index=labels).sort_values("F1")
if len(per_class) > 8:
    display(pd.concat([per_class.head(4), per_class.tail(4)]).style.format(precision=3).set_caption("4 worst and 4 best classes"))
else:
    display(per_class.style.format(precision=3).set_caption("Metrics of every class"))
g_true = pd.Series(y_te).map(aitbc.MilitaryVehicles.GROUPS).values if lab_col == "cls" else y_te
g_pred = pd.Series(pred).map(aitbc.MilitaryVehicles.GROUPS).values if lab_col == "cls" else pred
G = list(aitbc.MilitaryVehicles.GROUP_ORDER)
cm = confusion_matrix(g_true, g_pred, labels=G)
cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(1, 2 if LEARNING_CURVE else 1, figsize=(15 if LEARNING_CURVE else 7.5, 5.6), squeeze=False)
ax = ax[0]
ax[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(len(G)):
    for j in range(len(G)):
        ax[0].text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if cmn[i, j] > 0.5 else "k")
ax[0].set_xticks(range(len(G)))
ax[0].set_xticklabels(G, rotation=60, ha="right", fontsize=8)
ax[0].set_yticks(range(len(G)))
ax[0].set_yticklabels(G, fontsize=8)
ax[0].set_title(f"groups ({'from the recognised types' if lab_col == 'cls' else 'recognised directly'}): accuracy {np.mean(g_true == g_pred):.3f}", fontsize=9)
if LEARNING_CURVE:
    curve = []
    F_all = features_of(TRAIN_IDX)
    pos = {i: k for k, i in enumerate(TRAIN_IDX)}
    for n in (25, 50, 100, 150, 250, 400):
        s_ = mv.subset(TRAIN_IDX, n, seed=0)
        h_ = make_pipeline(StandardScaler(), LogisticRegression(C=C_LOGISTIC, max_iter=3000)).fit(F_all[[pos[i] for i in s_]], mv.table[lab_col].values[s_])
        curve.append((len(s_), accuracy_score(y_te, h_.predict(F_te))))
    ax[1].plot(*zip(*curve), "o-")
    ax[1].set_xlabel("training images")
    ax[1].set_ylabel("test accuracy")
    ax[1].set_title(f"learning curve, {LEVEL_45}", fontsize=9)
    ax[1].grid(alpha=0.3)
    print("Learning curve (training images, accuracy):", [(n, round(a, 3)) for n, a in curve])
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson42_transfer_learning.png"), dpi=80)
plt.show()
HEAD_INFO = dict(level=LEVEL_45, backbone=BACKBONE, input_size=INPUT_SIZE_45, labels=list(HEAD.classes_),
                 accuracy=round(acc, 3), f1_macro=round(float(F1.mean()), 3), n_train=len(tr))
""", form=True)

md(r"""
### Block 4.2.6. Export of the model to ONNX and check of the inference

**Purpose.** To join the frozen network and the trained linear layer of block 4.2.5 into one model, to save it in the ONNX format and to check that the inference programs of the onboard computer (ONNX Runtime, OpenCV DNN) give the same result as PyTorch.

**ONNX** (Open Neural Network Exchange) describes a model as a graph of standard operations (Conv, HardSwish, GlobalAveragePool, Gemm, Softmax…) with the weights; the file is executed without PyTorch by ONNX Runtime, OpenCV DNN, TensorRT and other programs on processors, GPUs and NPUs of onboard computers. The *opset* is the version of the set of operations: the program of inference must support the opset of the file.

**Model.** `image` (N × 3 × S × S, float32, normalised as for ImageNet) → convolutional part → global average pooling → linear layer → softmax → `probabilities` (N × K). The standardisation of the features of block 4.2.5 is included in the weights of the linear layer: $z = W\,\frac{f - \mu}{\sigma} + b = \frac{W}{\sigma} f + \left(b - W \frac{\mu}{\sigma}\right)$.

**Parameters.**
- `EXPORTER` — the classical exporter of PyTorch (tracing of the model, `dynamo=False`) or the new one (`dynamo=True`, opset not lower than 18; by default it writes the weights into a separate file `.onnx.data`, here `external_data=False` keeps the model in one file).
- `OPSET` — the version of the operations; `DYNAMIC_BATCH` — the number of images in the input is free; `N_CHECK` — the number of test images for the check.

**Expected result.** The file of the model and the file of its labels in `models` of the project, the check by the ONNX checker, the largest difference of the probabilities of PyTorch, ONNX Runtime and OpenCV DNN, the agreement of the recognised classes and the time per image.

**What to observe.** With the classical exporter (opset 17) the differences of the probabilities of ONNX Runtime and OpenCV DNN are of the order of $10^{-6}$ (rounding of float32), and the classes agree for all images. The file of the new exporter (opset 18) is executed correctly by ONNX Runtime, but in the check of the course OpenCV 4.12 (the version of Colab) did not read it, and OpenCV 5.0 read it and gave wrong probabilities (the classes agreed for 1–2 of 64 images) without any error message. For this reason a model is accepted for the onboard computer only after the check of its file by the program of inference of that computer. The file of MobileNetV3-small with the linear layer is about 3.6 MB. The preprocessing (size, mean, standard deviation, order of the channels RGB) is not a part of the file: it is written in the file of the labels and must be repeated exactly on the onboard computer.

**Control question 12.** Which information must accompany the ONNX file so that the onboard computer gives the same classes as the model in the notebook?
""")

code(r"""
#@title 4.2.6 Export of the recogniser to ONNX, check with ONNX Runtime and OpenCV DNN { display-mode: "form" }
EXPORTER = "classical (dynamo=False)" #@param ["classical (dynamo=False)", "new (dynamo=True)"]
OPSET = 17 #@param [13, 17, 18] {type:"raw"}
DYNAMIC_BATCH = True #@param {type:"boolean"}
N_CHECK = 64 #@param {type:"slider", min:16, max:256, step:16}

import importlib.util, subprocess
need = [p for p, mod in (("onnx", "onnx"), ("onnxruntime", "onnxruntime"), ("onnxscript", "onnxscript")) if importlib.util.find_spec(mod) is None]
if need:
    print("Installing:", ", ".join(need))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + need, check=True)
import onnx
import onnxruntime as ort
if "HEAD" not in globals():
    raise RuntimeError("Run block 4.2.5 first: it trains the linear layer of the recogniser.")
scaler, logreg = HEAD[0], HEAD[1]
backbone = BACKBONES[HEAD_INFO["backbone"]]
S_IN = HEAD_INFO["input_size"]


class Recogniser(torch.nn.Module):
    def __init__(self, net, scaler, logreg):
        super().__init__()
        self.features, self.pool = net.features, net.avgpool
        Wt = logreg.coef_ / scaler.scale_[None, :]
        bt = logreg.intercept_ - (logreg.coef_ * scaler.mean_ / scaler.scale_).sum(1)
        self.head = torch.nn.Linear(Wt.shape[1], Wt.shape[0])
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(Wt, dtype=torch.float32))
            self.head.bias.copy_(torch.tensor(bt, dtype=torch.float32))

    def forward(self, image):
        return torch.softmax(self.head(self.pool(self.features(image)).flatten(1)), dim=1)


model = Recogniser(backbone, scaler, logreg).eval()
tag = f"military_vehicles_{'types' if HEAD_INFO['level'] == '24 types' else 'groups'}_{HEAD_INFO['backbone'].replace('-', '').lower()}_{S_IN}"
onnx_path = os.path.join(MODEL_DIR, tag + ".onnx")
dummy = torch.from_numpy(aitbc.imagenet_batch([mv.image(TEST_IDX[0])], S_IN))
kw = dict(input_names=["image"], output_names=["probabilities"], opset_version=OPSET)
if DYNAMIC_BATCH:
    kw["dynamic_axes"] = {"image": {0: "batch"}, "probabilities": {0: "batch"}}
import warnings
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")        # the classical exporter is marked as deprecated since PyTorch 2.9
    if EXPORTER.startswith("new"):
        kw["external_data"] = False        # the new exporter otherwise writes the weights into a separate file .onnx.data
    torch.onnx.export(model, (dummy,), onnx_path, dynamo=EXPORTER.startswith("new"), **kw)
t_exp = time.time() - t0
m_onnx = onnx.load(onnx_path)
onnx.checker.check_model(m_onnx)
ops = pd.Series([n.op_type for n in m_onnx.graph.node]).value_counts()
opset_file = max(o.version for o in m_onnx.opset_import if o.domain in ("", "ai.onnx"))
labels_json = dict(labels=HEAD_INFO["labels"], input="image: N x 3 x S x S float32, RGB", size=S_IN,
                   mean=aitbc.IMAGENET_MEAN.tolist(), std=aitbc.IMAGENET_STD.tolist(), resize="cv2.INTER_AREA to S x S",
                   output="probabilities: N x K (softmax)", backbone=HEAD_INFO["backbone"] + ", ImageNet weights, frozen",
                   head="linear layer trained on " + str(HEAD_INFO["n_train"]) + " images", data="military vehicles (Kricheli et al., 2024)",
                   test_accuracy=HEAD_INFO["accuracy"], opset=opset_file)
json.dump(labels_json, open(os.path.join(MODEL_DIR, tag + "_labels.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

idx = np.sort(np.random.default_rng(0).choice(TEST_IDX, min(N_CHECK, len(TEST_IDX)), replace=False))   # random test images
xb = aitbc.imagenet_batch([mv.image(i) for i in idx], S_IN)
with torch.no_grad():
    t0 = time.time()
    p_torch = model(torch.from_numpy(xb)).numpy()
    t_torch = (time.time() - t0) / len(idx) * 1000
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
if DYNAMIC_BATCH:
    t0 = time.time()
    p_ort = sess.run(None, {"image": xb})[0]
else:
    t0 = time.time()
    p_ort = np.concatenate([sess.run(None, {"image": xb[k:k + 1]})[0] for k in range(len(xb))])
t_ort = (time.time() - t0) / len(idx) * 1000
size_mb = sum(os.path.getsize(f_) for f_ in (onnx_path, onnx_path + ".data") if os.path.exists(f_)) / 2 ** 20
rows = {"file": f"{onnx_path} ({size_mb:.2f} MB)", "exporter, opset of the file": f"{EXPORTER}, {opset_file}",
        "time of the export": f"{t_exp:.1f} s", "operations of the graph": ", ".join(f"{k} {v}" for k, v in ops.head(8).items()),
        "ONNX Runtime: largest difference of the probabilities": f"{np.abs(p_ort - p_torch).max():.2e}",
        "ONNX Runtime: agreement of the classes": f"{np.mean(p_ort.argmax(1) == p_torch.argmax(1)):.3f}",
        "time per image: PyTorch / ONNX Runtime (CPU), ms": f"{t_torch:.1f} / {t_ort:.1f}"}
try:
    net_cv = cv2.dnn.readNetFromONNX(onnx_path)
    p_cv = []
    t0 = time.time()
    for k in range(len(xb)):
        net_cv.setInput(xb[k:k + 1])
        p_cv.append(net_cv.forward())
    t_cv = (time.time() - t0) / len(idx) * 1000
    p_cv = np.concatenate(p_cv)
    rows["OpenCV DNN: largest difference of the probabilities"] = f"{np.abs(p_cv - p_torch).max():.2e}"
    rows["OpenCV DNN: agreement of the classes; time per image, ms"] = f"{np.mean(p_cv.argmax(1) == p_torch.argmax(1)):.3f}; {t_cv:.1f}"
except Exception as e:
    rows["OpenCV DNN"] = f"the file was not read by OpenCV {cv2.__version__}: {str(e).splitlines()[0][:120]}"
y_chk = mv.table["cls" if HEAD_INFO["level"] == "24 types" else "group"].values[idx]
cls_onnx = np.array(HEAD_INFO["labels"])[p_ort.argmax(1)]
rows[f"accuracy of the ONNX model on {len(idx)} random test images"] = f"{np.mean(cls_onnx == y_chk):.3f}"
display(pd.DataFrame({"value": list(rows.values())}, index=list(rows)))
k_show = min(6, len(idx))
fig, ax = plt.subplots(1, k_show, figsize=(2.6 * k_show, 2.8))
for a, k in zip(np.ravel(ax), range(k_show)):
    a.imshow(mv.image(idx[k]))
    j = p_ort[k].argmax()
    a.set_title(f"true: {y_chk[k]}\nONNX: {HEAD_INFO['labels'][j]} ({p_ort[k, j]:.2f})", fontsize=8, color="k" if HEAD_INFO["labels"][j] == y_chk[k] else "#c62828")
    a.axis("off")
plt.tight_layout()
plt.show()
update_project_json("4.2", {"lesson_4_2": dict(mlp=globals().get("MLP_RESULT"), transfer=HEAD_INFO,
                                               onnx=os.path.relpath(onnx_path, PROJECT_ROOT), opset=opset_file, exporter=EXPORTER,
                                               max_difference_onnxruntime=float(np.abs(p_ort - p_torch).max()))})
""", form=True)

md(r"""
### Report of Lesson 4.2

Double-click this cell and write the answers.

1. Neuron (block 4.2.1): weights set by hand and trained, accuracy, effect of the learning rate; answer to control question 7:
2. Fully connected network (block 4.2.2): layers, number of parameters, accuracy on the training and the test part; answer to control question 8:
3. Convolutional network (block 4.2.3): sizes of the maps of the blocks, classes of ImageNet for the chosen vehicle; answer to control question 9:
4. MobileNet (block 4.2.4): parameters, MAC and time of the five networks on the device of the session; answer to control question 10:
5. Transfer learning (block 4.2.5): HOG + SVM and CNN features on the same images, worst classes; answer to control question 11:
6. ONNX (block 4.2.6): file, opset, differences of the probabilities, time per image; answer to control question 12:

---
## Self-check of Lessons 4.1–4.2

1. A data set of frames of FPV drones over a training area contains 30 flights. Which checks are made before the training (duplicates, labels, empty frames), and how is the test part formed?
2. Why does the accuracy alone not suffice to estimate a recogniser of 24 types of vehicles with unequal numbers of images?
3. What do the HOG features describe, and why do they recognise a mine on grass better than the type of a tank in a web photograph?
4. What is common and what is different in a fully connected and in a convolutional network, and why is the second one used for images?
5. Why is MobileNet used on the onboard computer of a UAV or a ground robot instead of ResNet-50, and what must be checked after the export to ONNX?

## Data sources and licences

The course library reads the files directly from the repositories of the data sets; the data are not redistributed with the notebook. When images or results are published, the data sets are cited.

1. **Military vehicles.** Kricheli J. S., Vo K., Datta A., Ozgur S., Shakarian P. Error Detection and Constraint Recovery in Hierarchical Multi-Label Classification without Prior Knowledge. arXiv:2407.15192 (2024). https://arxiv.org/abs/2407.15192 · Data: https://huggingface.co/datasets/leibnitz-lab/military_vehicles · Licence declared by the authors: MIT; the photographs were collected by the authors from web sites, the rights to the individual photographs are not stated: use them only for training and research.
2. **SULAND.** Vivoli E., Bertini M., Capineri L. Deep Learning-Based Real-Time Detection of Surface Landmines Using Optical Imaging. *Remote Sensing*, 16(4), 677 (2024). https://doi.org/10.3390/rs16040677 · Refined annotation SULAND_v2: Lekhak S., Pulakurthi P. R., Joshi L., Bhatta R., Ientilucci E. J. SULAND v2: A Refined RGB Dataset and Deep Learning Object Detection Benchmark for UAV/UGV-Based SUrface LANDmine Detection Under Domain Shift. arXiv:2607.28996 (2026) · Data: https://huggingface.co/datasets/SagarLekhak/SULAND_v2_RGB_Surface_Landmine_Dataset · Licence CC BY 4.0. The mines are surrogates; the results are not an evidence of the readiness of a detector for humanitarian demining.
3. **ImageNet weights of MobileNetV3, MobileNetV2, ResNet** — torchvision (BSD-3-Clause); Howard A. et al. Searching for MobileNetV3. *ICCV 2019*, 1314–1324.
4. **HOG.** Dalal N., Triggs B. Histograms of Oriented Gradients for Human Detection. *CVPR 2005*, 886–893.

**Next notebook.** Notebook 07 — Topic 4, Lessons 4.3 and 4.4 (practical classes): classification of thermal signatures by HOG + SVM (target — background) with the analysis of errors; transfer learning of MobileNet on military vehicles with fine-tuning and export to ONNX.
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
    out = sys.argv[1] if len(sys.argv) > 1 else "AITBC_NB06_Military_object_recognition_ML_and_CNN.ipynb"
    nb = build(out)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    print(f"written {out}: {len(nb.cells)} cells ({n_md} text, {len(nb.cells) - n_md} code)")
