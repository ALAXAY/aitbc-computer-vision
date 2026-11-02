# Builder of Notebook 05 of the course "AI Technologies (Basic Course)" (AIT-BC).
# Topic 3, Lessons 3.1 (lecture), 3.3 and 3.4 (practical). All exercises use real open data sets.
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
# AI Technologies (Basic Course) · Notebook 05
## Topic 3. Classical methods of feature extraction in images

| Lesson | Type, duration | Title |
|---|---|---|
| **3.1** | Lecture, 2 h | Extraction of edges, silhouettes and contours of objects by classical methods |
| **3.3** | Practical class, 2 h | Features of silhouettes and thermal signatures in thermal frames |
| **3.4** | Practical class, 2 h | Stages of the stabilisation of the video stream of an FPV drone |

Course: *AI Technologies (Basic Course)*, AIT-BC · Content module 1 · Kruty Heroes Military Institute of Telecommunications and Information Technology, Kyiv, 2026.

**How to work with this notebook**

1. Run the cells from top to bottom: *Runtime → Run all*, or `Shift+Enter` cell by cell. Section 0 is always run first; after a restart of the runtime it is run again.
2. Every block begins with a text cell: the purpose, the parameters of the form, the formulas, the expected result and the control question. The answers are written in the text cell **Report** at the end of each practical lesson.
3. The blocks of the lecture are demonstrations: change the parameters of the forms and observe the result. Cells marked **(code shown)** in the practical lessons are the practical skills: read the code, change it and run it again. Cells marked **[auto]** run again after every change of a parameter.
4. The results are saved in the course project on Google Drive (`data`, `outputs`, `project.json`), created in Notebook 02.

**Data.** Only real frames are used, with the annotation made by the authors of the data sets:
- **Kust4K** — pixel-aligned visible and thermal frames of a UAV over city roads, day and night, with pixel labels (the boundaries of the labelled regions are the reference for the edge detectors);
- **HIT-UAV** — thermal frames of a reconnaissance UAV at 60–130 m, day and night, with boxes of persons, cars and bicycles, flight altitude and camera angle in the file names;
- **M3OT** — thermal frames of a DJI Mavic 3T (black-hot palette) with vehicle tracks;
- **TII-RATM** — the video stream of the onboard camera of an FPV racing quadrotor flown by a human pilot (125 frames per second, wide-angle lens) with the measurements of its gyroscope at 500 Hz, which serve as an independent reference of the rotation of the camera.

**Contents**

- **0. Preparation** — 0.1 runtime, course library and data sets · 0.2 course project
- **Lesson 3.1 (lecture)**
  - 3.1.1 The edge of an object and the gradient of brightness (Sobel)
  - 3.1.2 Edges by the Sobel and Canny methods: comparison with the boundaries of the annotation
  - 3.1.3 Binarisation of an image, global threshold of Otsu
  - 3.1.4 Contour analysis and geometric characteristics of objects
- **Lesson 3.3 (practical class)**
  - 3.3.1 Geometric transformations: scaling, rotation, translation, perspective transformation
  - 3.3.2 Preparation of a thermal frame: grey levels, CLAHE, polarity of thermal signatures
  - 3.3.3 Morphological processing of the binary mask
  - 3.3.4 Contours and their filtering by geometric characteristics
  - 3.3.5 Visualisation: contours and boxes on the original frame
- **Lesson 3.4 (practical class)**
  - 3.4.1 Reading a sequence of frames of a test FPV video fragment
  - 3.4.2 ORB keypoints on a pair of consecutive frames
  - 3.4.3 Matching of keypoints and quantitative estimation of the inter-frame shift
  - 3.4.4 Homography and geometric alignment of frames
  - 3.4.5 Saving of the stabilised video stream and comparative analysis
- **Summary**: self-check of Topic 3, data sources and licences
""")

# =====================================================================================
md(r"""
---
# 0. Preparation

### Block 0.1. Runtime, course library and data sets

**Purpose.** To prepare the runtime: the libraries, the course library `aitbc` (version 2.4.0 adds the functions of Topic 3 and the reader of the FPV data set TII-RATM) and the open data sets of the course.

**Parameters.**
- `CACHE_ON_DRIVE` — keep the downloaded frames on Google Drive (they remain after the end of the session); otherwise they are kept on the disk of the session.
- `PREFETCH_ALL` — download now the frames of Lessons 3.1 and 3.3 (about 35 MB); the FPV frames of Lesson 3.4 (about 55 MB in the cache, streamed once from an archive of about 200 MB) are downloaded in block 3.4.1.

**Output.** A table of versions and the number of frames in the cache.
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

t0 = time.time()
hit, kust, m3ot = aitbc.HITUAV(), aitbc.Kust4K(), aitbc.M3OT()
tii = aitbc.TIIRacing("flight-16p-trackRATM")
KUST_EDGE = list(aitbc.Kust4K.DAY[:6]) + list(aitbc.Kust4K.NIGHT[:6])          # 6 day and 6 night pairs
_im = hit.images
_busy = (_im.Person + _im.Car) >= 4
HIT_EVAL_NIGHT = _im[(_im.split == "test") & _im.night & _busy].sample(40, random_state=21).file.tolist()
HIT_EVAL_DAY = _im[(_im.split == "test") & ~_im.night & _busy].sample(24, random_state=21).file.tolist()
HIT_POLARITY = _im[_im.split != "train"].sample(160, random_state=11).file.tolist()
HIT_OBLIQUE = ["0_80_40_0_06933.jpg", "0_120_30_0_09449.jpg", "1_110_30_0_02555.jpg", "0_100_40_0_03347.jpg"]
if PREFETCH_ALL:
    kust.prefetch(KUST_EDGE)
    hit.prefetch(HIT_EVAL_NIGHT + HIT_EVAL_DAY + HIT_POLARITY + HIT_OBLIQUE)
    for f in (300, 500):
        m3ot.prefetch("1-08", "ir", start=f, count=1)
VERSIONS = {"Python": platform.python_version(), "NumPy": np.__version__, "OpenCV": cv2.__version__,
            "pandas": pd.__version__, "Matplotlib": matplotlib.__version__, "aitbc": aitbc.__version__}
n_files, size_mb = aitbc.cache_size()
display(pd.DataFrame({"Value": list(VERSIONS.values()) + ["Google Colab" if IN_COLAB else "local Jupyter",
                                                           f"{aitbc.DATA_ROOT}: {n_files} files, {size_mb:.0f} MB",
                                                           f"{time.time() - t0:.0f} s"]},
                     index=list(VERSIONS) + ["environment", "cache of the data sets", "preparation time"]))
""", form=True)

md(r"""
### Block 0.2. Course project

**Purpose.** To connect the course project created in Notebook 02. The results of this notebook are saved in it: figures and tables in `outputs`, the FPV video fragment in `data/video`, the stabilised video in `outputs/video` and the parameters of the lessons in `project.json`.

**Parameters.** `USE_DRIVE` — the project is on Google Drive (Colab asks for access to the Google account); otherwise it is on the disk of the session. `PROJECT_PATH` — folder of the project; the same as in Notebooks 02–04.

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
# Lesson 3.1 (Lecture). Extraction of edges, silhouettes and contours of objects by classical methods

**Learning questions**
1. The concept of the edge of an object in an image and the basic approaches to its algorithmic extraction.
2. Extraction of the edges of an object by the Sobel and Canny methods with regard to their practical application.
3. Methods of binarisation of an image, including the global threshold processing by the Otsu method.
4. Contour analysis and computation of the geometric characteristics of the extracted objects.

**Classical methods** compute features of an image by fixed formulas: the gradient of brightness, a threshold, a contour and its geometric characteristics. They need no training data, work in milliseconds on an onboard computer and their result can be explained step by step. Their parameters (thresholds, sizes of windows) are chosen by the developer; the lecture shows by measurements on real frames how strongly the result depends on them.
""")

code(r"""
#@title Frames of Lesson 3.1 (run once) { display-mode: "form" }
SOURCES_31 = ["Kust4K: day, visible", "Kust4K: night, visible", "Kust4K: day, thermal (luminance of the palette)",
              "Kust4K: night, thermal (luminance of the palette)", "HIT-UAV: night, thermal (white-hot)",
              "HIT-UAV: day, thermal (white-hot)", "M3OT 1-08: dusk, thermal (black-hot)"]


def frame_31(source, number=1):
    # dict(img, grey, labels, objects, caption, altitude, angle) of a real frame of a data set
    k = int(number) - 1
    if source.startswith("Kust4K"):
        fid = (aitbc.Kust4K.DAY if "day" in source else aitbc.Kust4K.NIGHT)[k % 6]
        p = kust.pair(fid)
        img = p["rgb"] if "visible" in source else p["tir_gray"]
        return dict(img=img, grey=aitbc.gray(img), labels=p["labels"], objects=[], altitude=None, angle=None,
                    caption=f"Kust4K {fid} ({'night' if p['night'] else 'day'}), {'visible' if 'visible' in source else 'thermal'}")
    if source.startswith("HIT-UAV"):
        files = HIT_EVAL_NIGHT if "night" in source else HIT_EVAL_DAY
        f = files[k % len(files)]
        g, objs = hit.load(f)
        r = hit.images.set_index("file").loc[f]
        return dict(img=g, grey=g, labels=None, objects=objs, altitude=int(r.altitude), angle=int(r.angle),
                    caption=f"HIT-UAV {f}: {r.altitude} m, camera angle {r.angle}°, {'night' if r.night else 'day'}")
    f0 = (300, 500)[k % 2]
    g = aitbc.to_white_hot(m3ot.frames("1-08", "ir", start=f0, count=1)[0], "black-hot")[..., 0]
    return dict(img=g, grey=g, labels=None, objects=m3ot.objects(m3ot.tracks("1-08", "ir"), f0 + 1), altitude=None, angle=None,
                caption=f"M3OT 1-08 (dusk), thermal frame {f0 + 1}, inverted to white-hot")


print("Sources of Lesson 3.1:", "; ".join(SOURCES_31))
""", form=True)

md(r"""
## Block 3.1.1. The edge of an object and the gradient of brightness

**Edge.** An edge is a line in the image along which the brightness changes abruptly: the boundary between an object and its background, between two surfaces or the boundary of a shadow. In a thermal frame an edge separates surfaces of different temperature or emissivity. The edge is found where the derivative of the brightness is large. A digital image is a grid of samples; the derivative is replaced by a difference, combined with smoothing across the direction of the difference (Sobel operator):

$$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = K_x^{\mathsf T}, \qquad G_x = I * K_x, \quad G_y = I * K_y, \tag{1}$$

$$|G| = \sqrt{G_x^2 + G_y^2}, \qquad \theta = \operatorname{atan2}(G_y, G_x), \tag{2}$$

where $*$ is the convolution, $|G|$ the magnitude and $\theta$ the direction of the gradient; the edge runs perpendicular to $\theta$.

**Parameters.** `SOURCE`, `NUMBER` — frame; `BLUR` — size of the Gaussian smoothing before the operator (0 — without it); `SOBEL_KSIZE` — size of the Sobel kernel (1 — a central difference without smoothing); `ROW` — row of the profile as a share of the height of the frame.

**Expected result.** The frame, $G_x$, $G_y$, $|G|$, the direction $\theta$ in colour (brightness = magnitude), the profile of the brightness and of $|G|$ along the chosen row.

**What to observe.** $G_x$ responds to vertical edges and $G_y$ to horizontal ones; the sign shows whether the brightness grows or falls. The magnitude is large not only at the boundaries of objects but also at road markings, shadows and texture of vegetation. Without smoothing, the noise of the sensor gives small gradients in every pixel of a thermal frame. A larger kernel suppresses the noise and makes the edges thicker. The operator gives a value in every pixel, so a threshold or a thinning of the edges is needed to obtain lines (block 3.1.2).

**Control question 1.** Why is the derivative of the brightness computed with smoothing across its direction? What happens to the magnitude of the gradient on an edge after inversion of the palette of a thermal frame (white-hot → black-hot)?
""")

code(r"""
#@title 3.1.1 Gradient of brightness: Sobel operator { run: "auto" }
SOURCE = "Kust4K: day, visible" #@param ["Kust4K: day, visible", "Kust4K: night, visible", "Kust4K: day, thermal (luminance of the palette)", "Kust4K: night, thermal (luminance of the palette)", "HIT-UAV: night, thermal (white-hot)", "HIT-UAV: day, thermal (white-hot)", "M3OT 1-08: dusk, thermal (black-hot)"]
NUMBER = 1 #@param {type:"slider", min:1, max:6, step:1}
BLUR = 3 #@param [0, 3, 5, 7] {type:"raw"}
SOBEL_KSIZE = 3 #@param [1, 3, 5, 7] {type:"raw"}
ROW = 0.5 #@param {type:"slider", min:0.05, max:0.95, step:0.05}

fr = frame_31(SOURCE, NUMBER)
g = fr["grey"]
gs = cv2.GaussianBlur(g, (BLUR, BLUR), 0) if BLUR else g
gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE)                  # formula (1)
gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE)
mag, theta = np.hypot(gx, gy), np.arctan2(gy, gx)                         # formula (2)
hsv = np.dstack([((theta + np.pi) / (2 * np.pi) * 179).astype(np.uint8), np.full(g.shape, 255, np.uint8),
                 np.clip(mag / np.percentile(mag, 99) * 255, 0, 255).astype(np.uint8)])
direction = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
y = int(ROW * (g.shape[0] - 1))

fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
lim = np.percentile(np.abs(np.r_[gx.ravel(), gy.ravel()]), 99)
panels = [(fr["img"], "frame", None), (gx, "G_x (red: brightness grows to the right)", "RdBu_r"),
          (gy, "G_y (red: brightness grows downwards)", "RdBu_r"), (mag, "magnitude |G|", "magma"),
          (direction, "direction θ (colour), magnitude (brightness)", None)]
for a, (im_, t_, cm) in zip(ax.ravel(), panels):
    if cm == "RdBu_r":
        a.imshow(im_, cmap=cm, vmin=-lim, vmax=lim)
    elif cm == "magma":
        a.imshow(im_, cmap=cm, vmax=np.percentile(im_, 99))
    else:
        a.imshow(im_, cmap="gray" if im_.ndim == 2 else None)
    a.axhline(y, color="cyan", lw=0.8)
    a.set_title(t_, fontsize=10)
    a.axis("off")
a = ax[1, 2]
a.plot(g[y].astype(float), color="0.3", lw=0.9, label="brightness I")
a2 = a.twinx()
a2.plot(mag[y], color="C3", lw=0.9, label="|G|")
a.set_xlabel("column x")
a.set_ylabel("brightness, grey levels")
a2.set_ylabel("|G|", color="C3")
a.set_title(f"Profile of row y = {y}", fontsize=10)
a.grid(alpha=0.3)
fig.suptitle(fr["caption"], fontsize=11)
plt.tight_layout()
plt.show()
print(f"|G|: median {np.median(mag):.1f}, 90th percentile {np.percentile(mag, 90):.1f}, 99th percentile {np.percentile(mag, 99):.1f} "
      f"(the strongest edges are few: the median is {np.percentile(mag, 99) / max(np.median(mag), 1e-6):.0f} times smaller than the 99th percentile)")
""")

md(r"""
## Block 3.1.2. Edges by the Sobel and Canny methods: comparison with the boundaries of the annotation

**Canny method.** (a) Gaussian smoothing; (b) gradient (1)–(2); (c) *non-maximum suppression*: a pixel is kept when its $|G|$ is not smaller than the values of its two neighbours along the direction $\theta$ — the edges become one pixel thick; (d) *hysteresis* with two thresholds:

$$|G| \ge T_{high} \;\Rightarrow\; \text{strong edge}, \qquad T_{low} \le |G| < T_{high} \;\Rightarrow\; \text{edge only if connected to a strong edge}. \tag{3}$$

A frequently used rule for the thresholds is $T_{low} = (1 - 0.33)\,m$, $T_{high} = (1 + 0.33)\,m$, where $m$ is the median brightness of the frame.

**Reference.** The pixel labels of Kust4K give the boundaries between the labelled regions (road, building, car, tree, person…). A detected edge pixel is correct when a boundary pixel lies within the tolerance $\tau$:

$$P = \frac{|E \cap B_\tau|}{|E|}, \qquad R = \frac{|B \cap E_\tau|}{|B|}, \qquad F_1 = \frac{2PR}{P + R}, \tag{4}$$

where $E$ is the set of detected edge pixels, $B$ the set of boundary pixels and the index $\tau$ means the dilation by $\tau$ pixels.

**Parameters.** `SOURCE`, `NUMBER` — frame of Kust4K; `CANNY_LOW`, `CANNY_HIGH` — thresholds (3); `MEDIAN_RULE` — thresholds from the median of the frame; `BLUR`; `L2_GRADIENT` — magnitude by (2) instead of $|G_x| + |G_y|$; `SOBEL_SHARE` — share of the pixels with the largest $|G|$ kept as Sobel edges; `TOLERANCE_PX` — $\tau$ in (4).

**Expected result.** The reference boundaries, the edges of Canny and of the Sobel threshold, the measures (4) for the frame and a table for 12 frames (6 day, 6 night, visible and thermal): fixed thresholds, the median rule and the best threshold chosen separately for every frame.

**What to observe.** Both methods find many edges that are not boundaries of labelled regions (road markings, shadows, texture), therefore $P$ is low: with the thresholds 50/150 $F_1$ is 0.33–0.48 in the four groups of frames, and even the best threshold chosen for every frame separately gives 0.41–0.61. No pair of thresholds suits all frames: the best $T_{low}$ ranges from 5 to 100 levels between the frames. The median rule does not solve this problem: in all four groups it is below the best threshold of the frame, most of all for night thermal frames (0.41 against 0.61). At night the thermal frame gives better edges of roads and buildings than the visible one ($F_1$ 0.48 against 0.36 with the fixed thresholds), whose median brightness is only about 17 levels. These limits of fixed parameters are one of the reasons for the methods with training of Topics 4 and 5.

**Control question 2.** Why does non-maximum suppression make the edges thin? What does the hysteresis (3) give compared with one threshold?
""")

code(r"""
#@title 3.1.2 Sobel and Canny compared with the boundaries of the annotation { run: "auto" }
SOURCE = "Kust4K: day, visible" #@param ["Kust4K: day, visible", "Kust4K: night, visible", "Kust4K: day, thermal (luminance of the palette)", "Kust4K: night, thermal (luminance of the palette)"]
NUMBER = 1 #@param {type:"slider", min:1, max:6, step:1}
CANNY_LOW = 50 #@param {type:"slider", min:0, max:250, step:5}
CANNY_HIGH = 150 #@param {type:"slider", min:5, max:255, step:5}
MEDIAN_RULE = False #@param {type:"boolean"}
BLUR = 5 #@param [0, 3, 5, 7] {type:"raw"}
L2_GRADIENT = True #@param {type:"boolean"}
SOBEL_SHARE = 0.10 #@param {type:"slider", min:0.02, max:0.3, step:0.01}
TOLERANCE_PX = 3 #@param {type:"slider", min:1, max:6, step:1}


def canny_edges(g, low, high, median_rule=False):
    gs = cv2.GaussianBlur(g, (BLUR, BLUR), 0) if BLUR else g
    if median_rule:
        m = float(np.median(gs))
        low, high = int(max(0, 0.67 * m)), int(min(255, 1.33 * m))
    return cv2.Canny(gs, low, max(high, low + 1), L2gradient=L2_GRADIENT), (low, high)


def sobel_edges(g, share):
    gs = cv2.GaussianBlur(g, (BLUR, BLUR), 0) if BLUR else g
    mag = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1))
    return (mag > np.percentile(mag, 100 * (1 - share))).astype(np.uint8) * 255


fr = frame_31(SOURCE, NUMBER)
ref = aitbc.label_boundaries(fr["labels"], ignore=(0,))
ec, (lo, hi) = canny_edges(fr["grey"], CANNY_LOW, CANNY_HIGH, MEDIAN_RULE)
es = sobel_edges(fr["grey"], SOBEL_SHARE)
sc, ss = aitbc.edge_map_scores(ec, ref, TOLERANCE_PX), aitbc.edge_map_scores(es, ref, TOLERANCE_PX)   # formula (4)

fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
for a, (im_, t_) in zip(ax, ((fr["img"], fr["caption"]), (ref, "boundaries of the labelled regions (reference)"),
                             (ec, f"Canny {lo}/{hi}: P = {sc['precision']:.2f}, R = {sc['recall']:.2f}, F1 = {sc['f1']:.2f}"),
                             (es, f"Sobel, {SOBEL_SHARE:.0%} largest |G|: F1 = {ss['f1']:.2f}"))):
    a.imshow(im_, cmap="gray" if im_.ndim == 2 else None)
    a.set_title(t_, fontsize=9)
    a.axis("off")
plt.tight_layout()
plt.show()

rows = []
for fid in KUST_EDGE:
    p = kust.pair(fid)
    b = aitbc.label_boundaries(p["labels"], ignore=(0,))
    for ch, im_ in (("visible", p["rgb"]), ("thermal", p["tir_gray"])):
        gg = aitbc.gray(im_)
        f_fixed = aitbc.edge_map_scores(canny_edges(gg, CANNY_LOW, CANNY_HIGH)[0], b, TOLERANCE_PX)
        f_med = aitbc.edge_map_scores(canny_edges(gg, 0, 0, True)[0], b, TOLERANCE_PX)["f1"]
        best = max((aitbc.edge_map_scores(canny_edges(gg, t, 2 * t)[0], b, TOLERANCE_PX)["f1"], t)
                   for t in (5, 10, 20, 30, 40, 60, 80, 100, 130))
        rows.append(dict(time="night" if p["night"] else "day", channel=ch, median_brightness=float(np.median(gg)),
                         P_fixed=f_fixed["precision"], R_fixed=f_fixed["recall"], F1_fixed=f_fixed["f1"], F1_median_rule=f_med,
                         F1_best_of_frame=best[0], best_T_low=best[1],
                         F1_Sobel=aitbc.edge_map_scores(sobel_edges(gg, SOBEL_SHARE), b, TOLERANCE_PX)["f1"]))
tab = pd.DataFrame(rows)
summary = tab.groupby(["time", "channel"]).median(numeric_only=True)
summary["best_T_low_range"] = tab.groupby(["time", "channel"]).best_T_low.agg(lambda s: f"{s.min()}–{s.max()}")
display(summary.round(3).style.set_caption(f"Medians over 6 frames of every group; fixed thresholds {CANNY_LOW}/{CANNY_HIGH}, τ = {TOLERANCE_PX} px"))
tab.to_csv(os.path.join(REPORT_DIR, "lesson31_edges.csv"), index=False)
""")

md(r"""
## Block 3.1.3. Binarisation of an image, global threshold of Otsu

**Binarisation** divides the pixels into two classes by a threshold $t$: $b(x, y) = 1$ if $I(x, y) > t$, otherwise 0. **Otsu's method** chooses the threshold that separates the histogram into two classes with the largest between-class variance:

$$t^* = \arg\max_t \sigma_B^2(t), \qquad \sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\,\big(\mu_0(t) - \mu_1(t)\big)^2, \tag{5}$$

where $\omega_0, \omega_1$ are the shares of the pixels below and above $t$ and $\mu_0, \mu_1$ their mean brightness. The method has no parameters and works well when the histogram has two modes of comparable size, for example a warm road and a cool field.

**Small targets.** A person seen from 100 m occupies a few tens of pixels — far less than 1 % of the frame — and does not form its own mode of the histogram. Local contrast is therefore extracted first: the *white top-hat* $I - (I \circ S)$ (difference between the frame and its morphological opening with a structuring element $S$ larger than the target) keeps only bright details smaller than $S$, and the threshold is applied to it (Lesson 3.3).

**Parameters.** `SOURCE`, `NUMBER` — thermal frame; `PREPROCESSING` — none or white top-hat; `TOP_HAT_M` — size of $S$ in metres (converted to pixels with the GSD of the frame of HIT-UAV).

**Expected result.** The frame, its histogram with the curve $\sigma_B^2(t)$ and the threshold $t^*$, the binary mask; for HIT-UAV — the share of the annotated persons and vehicles whose centre lies in the mask. The threshold of formula (5) is compared with `cv2.threshold(..., THRESH_OTSU)`.

**What to observe.** Without preprocessing, Otsu's threshold divides the frame into large warm and cool areas (roads, roofs, fields): in the default night frame the mask covers 60 % of the frame (43–81 % in six night frames of the set). Most persons lie in this mask, but together with the warm ground, so the mask does not separate them. After the top-hat with $S$ = 3 m the histogram is dominated by the background close to zero, and the mask covers 1–11 % of the frame; in the default frame it still contains 9 of the 10 annotated persons. In the full pipeline on the 40 night test frames of HIT-UAV (block 3.3.4) the global Otsu threshold gives the recall of persons 0.25, the top-hat with a threshold 0.84.

**Control question 3.** For which histogram does Otsu's method give a good threshold? Why is it not suitable for small warm targets without preprocessing?
""")

code(r"""
#@title 3.1.3 Otsu's threshold { run: "auto" }
SOURCE = "HIT-UAV: night, thermal (white-hot)" #@param ["HIT-UAV: night, thermal (white-hot)", "HIT-UAV: day, thermal (white-hot)", "Kust4K: night, thermal (luminance of the palette)", "M3OT 1-08: dusk, thermal (black-hot)"]
NUMBER = 1 #@param {type:"slider", min:1, max:6, step:1}
PREPROCESSING = "none" #@param ["none", "white top-hat"]
TOP_HAT_M = 3.0 #@param {type:"slider", min:1, max:15, step:0.5}

fr = frame_31(SOURCE, NUMBER)
g = cv2.GaussianBlur(fr["grey"], (3, 3), 0)
if PREPROCESSING == "white top-hat":
    gsd_c = hit.gsd(fr["altitude"], fr["angle"]) if fr["altitude"] else 0.15
    se_px = max(3, int(round(TOP_HAT_M / gsd_c))) | 1
    src = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_px, se_px)))
    note = f"white top-hat, S = {TOP_HAT_M:g} m = {se_px} px"
else:
    src, note = g, "without preprocessing"
t_star, sigma_b, hist = aitbc.otsu(src)                                   # formula (5)
t_cv, MASK_313 = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
FRAME_313 = fr

fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
ax[0].imshow(fr["img"], cmap="gray")
ax[0].set_title(fr["caption"], fontsize=9)
ax[1].bar(np.arange(256), hist / hist.sum(), width=1.0, color="0.55")
ax[1].set_yscale("log")
a2 = ax[1].twinx()
a2.plot(sigma_b, color="C3", lw=1.2)
a2.set_ylabel("σ_B²(t)", color="C3")
ax[1].axvline(t_star, color="C3", ls="--")
ax[1].set_title(f"Histogram ({note}) and σ_B²(t): t* = {t_star}", fontsize=9)
ax[1].set_xlabel("grey level t")
ax[1].set_ylabel("share of pixels (logarithmic)")
ax[2].imshow(MASK_313, cmap="gray")
ax[2].set_title(f"mask I > t*: {np.mean(MASK_313 > 0):.1%} of the pixels", fontsize=9)
for a in (ax[0], ax[2]):
    a.axis("off")
plt.tight_layout()
plt.show()

rows = {"threshold t* by formula (5)": t_star, "threshold of cv2.threshold (THRESH_OTSU)": int(t_cv),
        "share of the pixels in the mask": f"{np.mean(MASK_313 > 0):.1%}"}
for cls in ("Person", "Car", "vehicle"):
    obs = [o for o in fr["objects"] if o["cls"] == cls]
    if obs:
        inside = [MASK_313[int(min(max((o["box"][1] + o["box"][3]) / 2, 0), MASK_313.shape[0] - 1)),
                           int(min(max((o["box"][0] + o["box"][2]) / 2, 0), MASK_313.shape[1] - 1))] > 0 for o in obs]
        rows[f"annotated {cls} with the centre in the mask"] = f"{sum(inside)} of {len(obs)}"
display(pd.DataFrame({"Value": rows}))
""")

md(r"""
## Block 3.1.4. Contour analysis and geometric characteristics of objects

**Contour.** A contour is the closed boundary of a connected region of a binary mask, stored as a list of boundary points (`cv2.findContours`). The mode `RETR_EXTERNAL` keeps only outer contours, `RETR_TREE` also the contours of holes with their hierarchy. For every contour the geometric characteristics are computed:

$$A, \quad P, \quad \frac{L}{W}, \quad e = \frac{A}{A_{box}}, \quad s = \frac{A}{A_{hull}}, \quad c = \frac{4\pi A}{P^2}, \tag{6}$$

where $A$ is the area and $P$ the perimeter, $L \times W$ the size of the minimum-area rotated rectangle, $e$ the extent (filling of the bounding box), $s$ the solidity (filling of the convex hull) and $c$ the circularity (1 for a circle). With the ground sampling distance (Lesson 1.1) the area and the sizes are converted to metres:

$$A_{m^2} = A_{px}\,\mathrm{GSD}^2, \qquad L_m = L_{px}\,\mathrm{GSD}. \tag{7}$$

The sizes in metres do not depend on the flight altitude and can be compared with the known sizes of objects.

**Parameters.** `RETRIEVAL` — mode of `findContours`; `MIN_AREA_PX` — smallest area of a contour shown in the table; `TABLE_ROWS` — number of rows. The mask of block 3.1.3 is used.

**Expected result.** The contours of the mask on the frame, the table of characteristics (6)–(7) of the largest contours and the sizes of the annotated objects of HIT-UAV in metres, computed from the rotated boxes of the annotation of the frames with a vertical view (camera angle 80–90°).

**What to observe.** The annotation of frames with a vertical view gives the reference sizes (medians): a person seen from above is about 0.9 × 0.7 m (0.7 m²; 90 % of the persons below 1.2 m²), a bicycle with its rider about 2.4 × 1.0 m (2.3 m²), a car about 5.9 × 2.7 m (16 m²; 10 % of the cars below 7 m²). These values are used for the filtering of contours in block 3.3.4: a contour of 0.1–2.5 m² is a candidate person, of 2.5–40 m² a candidate vehicle; bicycles lie near the boundary of the two classes. Contours of large warm surfaces (roofs, roads) have areas of hundreds of m² (in the default frame the two largest contours are about 730 and 770 m²) and are removed by the limit of the area.

**Control question 4.** Which characteristics of (6) do not depend on the size of an object in pixels? Why is the area in m² (7) a better feature than the area in pixels for frames taken from different altitudes?
""")

code(r"""
#@title 3.1.4 Contours and their geometric characteristics { run: "auto" }
RETRIEVAL = "RETR_EXTERNAL" #@param ["RETR_EXTERNAL", "RETR_LIST", "RETR_TREE"]
MIN_AREA_PX = 4 #@param {type:"slider", min:1, max:100, step:1}
TABLE_ROWS = 10 #@param {type:"slider", min:5, max:30, step:1}

fr = FRAME_313
cnts, hier = cv2.findContours(MASK_313, getattr(cv2, RETRIEVAL), cv2.CHAIN_APPROX_SIMPLE)
gsd_c = hit.gsd(fr["altitude"], fr["angle"]) if fr["altitude"] else None
feats = []
for c in cnts:
    if cv2.contourArea(c) < MIN_AREA_PX:
        continue
    f = aitbc.contour_features(c, gsd_c, gsd_c) if gsd_c else aitbc.contour_features(c)      # formulas (6)-(7)
    feats.append(f)
F = pd.DataFrame(feats)
vis = cv2.cvtColor(fr["grey"], cv2.COLOR_GRAY2RGB)
cv2.drawContours(vis, [c for c in cnts if cv2.contourArea(c) >= MIN_AREA_PX], -1, (255, 60, 60), 1)
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(vis)
ax.set_title(f"{fr['caption']}: {len(feats)} contours of {len(cnts)} with A ≥ {MIN_AREA_PX} px ({RETRIEVAL})", fontsize=9)
ax.axis("off")
plt.show()
if len(F):
    cols = ["area_px", "perimeter_px", "length_px", "width_px", "aspect", "extent", "solidity", "circularity"] + (["area_m2", "length_m", "width_m"] if gsd_c else [])
    display(F.sort_values("area_px", ascending=False)[cols].head(TABLE_ROWS).reset_index(drop=True).round(2)
            .style.set_caption("Largest contours of the mask of block 3.1.3" + (f" (GSD at the centre {gsd_c:.3f} m/px)" if gsd_c else "")))

o = hit.objects[(hit.objects.angle >= 80) & hit.objects.cls.isin(["Person", "Bicycle", "Car", "OtherVehicle"])].dropna(subset=["width_px"]).copy()
o["gsd"] = [hit.gsd(a, b) for a, b in zip(o.altitude, o.angle)]
o["length_m"], o["width_m"] = o.length_px * o.gsd, o.width_px * o.gsd
o["area_m2"] = o.length_m * o.width_m
SIZE_REF = o.groupby("cls")[["length_m", "width_m", "area_m2"]].quantile([0.1, 0.5, 0.9]).unstack().round(2)
SIZE_REF.insert(0, ("objects", ""), o.groupby("cls").size())
display(SIZE_REF.style.set_caption("Sizes of the annotated objects of HIT-UAV in metres (rotated boxes, camera angle 80–90°): percentiles 10, 50, 90"))
""")

md(r"""
### Questions for self-check of Lesson 3.1

1. What is an edge of an object in a visible frame and in a thermal frame? Which physical quantities change across it?
2. Write the Sobel kernels and the formulas of the magnitude and direction of the gradient.
3. List the four stages of the Canny method and explain the role of each of them.
4. Why can no pair of thresholds of the Canny method be chosen that suits day and night frames of the same scene?
5. Write the criterion of Otsu's method. When is it suitable, and why are small warm targets not separated by it?
6. Which geometric characteristics of a contour separate a person from a car in a thermal frame taken from a UAV, and why must the area be converted to square metres?
""")


# =====================================================================================
md(r"""
---
# Lesson 3.3 (Practical class). Features of silhouettes and thermal signatures in thermal frames

**Learning questions**
1. Basic geometric transformations of images: scaling, rotation, translation, perspective transformations.
2. Preparation of a thermal frame: conversion to grey levels, application of CLAHE.
3. Morphological processing of the obtained binary mask.
4. Detection of contours and their filtering by geometric characteristics.
5. Visualisation of the results of the processing by overlaying contours or bounding boxes on the original frame.

**Task of the lesson.** A classical module of detection of thermal signatures is built step by step: frame → grey levels in the white-hot convention → local contrast and threshold → morphology → contours → filtering by sizes in metres → persons and vehicles on the frame. Every step is checked on the annotated night frames of HIT-UAV.

**Result of the lesson.** The detection pipeline with the chosen parameters, its recall and precision on 40 annotated night frames, the figure with the detections and the table of contours in `outputs`, the parameters in `project.json`.
""")

md(r"""
## Task 3.3.1. Geometric transformations: scaling, rotation, translation, perspective transformation

**Affine transformation.** Scaling by $s$, rotation by $\alpha$ about the centre $c$ and translation by $t$:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = s\begin{bmatrix} \cos\alpha & \sin\alpha \\ -\sin\alpha & \cos\alpha \end{bmatrix}\left(\begin{bmatrix} x \\ y \end{bmatrix} - c\right) + c + t, \tag{8}$$

the matrix $2 \times 3$ of `cv2.getRotationMatrix2D(c, α, s)` with $t$ added to its last column ($\alpha > 0$ — counter-clockwise). The boxes of the annotation are transformed by their four corners: after a rotation the axis-aligned box of the corners contains, in addition to the object, a part of the background.

**Perspective transformation to the top view.** A camera at the height $h$ looks forward with the depression angle $\vartheta$ (the camera angle of HIT-UAV; 90° — vertical view). A point $(X, Y)$ of the flat ground is mapped to the image by the homography

$$\tilde{\mathbf x} = H_g \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}, \qquad H_g = K\,[\,\mathbf r_1 \;\; \mathbf r_2 \;\; -h\,\mathbf r_3\,], \tag{9}$$

where $K$ is the camera matrix (from the field of view 25° × 20° of the Zenmuse XT2) and $\mathbf r_i$ the columns of the rotation of the camera; $H_g^{-1}$ gives the top view with a constant GSD.

**Parameters.** `FRAME` — an oblique frame of HIT-UAV (altitude and camera angle from the file name); `SCALE`, `ANGLE_DEG`, `SHIFT_X`, `SHIFT_Y` — parameters of (8); `MAX_RANGE_M` — largest ground distance shown in the top view.

**Expected result.** The frame with the boxes, the result of (8) with the transformed boxes, the top view (9) with the boxes; the growth of the boxes after rotation, measured on the annotation of HIT-UAV; the length of cars in metres in four bands of rows of oblique frames for three models of the GSD.

**What to observe.** In the top view the lines of a parking lot become parallel and perpendicular, as on the ground. Cars in the far part of the frame are stretched along the line of sight: the perspective transformation is exact only for points of the ground, while the box of a car also contains its side. The rotation of a frame by 15°, 30° and 45° increases the area of the boxes of the annotation (median) by 1.5, 1.9 and 2.0 times — an augmentation by rotation needs rotated boxes or a recomputation of the labels. For the length of cars in oblique frames: one GSD of the centre of the frame makes the cars of the far rows 26 % shorter than those of the near rows (4.0 m against 5.4 m); the GSD of the row computed from the slant range (model «range of the row») gives 4.6–4.9 m in all bands; the ground-plane model, which also divides by the sine of the angle of view, overestimates the far cars (6.7 m). The model «range of the row» is used in task 3.3.4.

**Control question 5.** Why does an axis-aligned box grow after rotation of a frame? Why is a perspective transformation to the top view exact for road markings and not for cars?
""")

code(r"""
#@title 3.3.1 Geometric transformations of a thermal frame (code shown) { run: "auto" }
FRAME = "0_80_40_0_06933.jpg: day, 80 m, 40°, parking lot" #@param ["0_80_40_0_06933.jpg: day, 80 m, 40°, parking lot", "0_120_30_0_09449.jpg: day, 120 m, 30°, parking lot and buildings", "1_110_30_0_02555.jpg: night, 110 m, 30°, persons on a sports ground", "0_100_40_0_03347.jpg: day, 100 m, 40°, sports ground"]
SCALE = 1.0 #@param {type:"slider", min:0.5, max:2.0, step:0.05}
ANGLE_DEG = 30 #@param {type:"slider", min:-180, max:180, step:5}
SHIFT_X = 0 #@param {type:"slider", min:-200, max:200, step:10}
SHIFT_Y = 0 #@param {type:"slider", min:-200, max:200, step:10}
MAX_RANGE_M = 250 #@param {type:"slider", min:50, max:500, step:10}

f = FRAME.split(":")[0]
g, objs = hit.load(f)
alt, ang = int(f.split("_")[1]), int(f.split("_")[2])
H_, W_ = g.shape
boxes = np.array([o["box"] for o in objs], float).reshape(-1, 4)
colors = [hit.COLORS.get(o["cls"], (255, 255, 255)) for o in objs]

M = cv2.getRotationMatrix2D(((W_ - 1) / 2, (H_ - 1) / 2), ANGLE_DEG, SCALE)          # formula (8)
M[:, 2] += (SHIFT_X, SHIFT_Y)
aff = cv2.warpAffine(g, M, (W_, H_))
boxes_aff = aitbc.transform_boxes(boxes, np.vstack([M, [0, 0, 1]]))

K_xt2 = aitbc.camera_matrix(W_, H_, hit.CAMERA["hfov_deg"], hit.CAMERA["vfov_deg"])
Hg, (Wt, Ht), gsd_top, ext = aitbc.ground_homography(alt, ang, K_xt2, (W_, H_), max_range_m=MAX_RANGE_M)   # formula (9)
top = cv2.warpPerspective(g, Hg, (Wt, Ht))
boxes_top = aitbc.transform_boxes(boxes, Hg)


def draw(ax, img, bxs, title):
    ax.imshow(img, cmap="gray")
    for b, c in zip(bxs, colors):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=np.array(c) / 255, lw=0.9))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


fig, ax = plt.subplots(1, 3, figsize=(19, 6), gridspec_kw=dict(width_ratios=[1, 1, Wt / Ht * H_ / W_]))
draw(ax[0], g, boxes, f"{f}: {alt} m, camera angle {ang}°")
draw(ax[1], aff, boxes_aff, f"(8): s = {SCALE:g}, α = {ANGLE_DEG}°, t = ({SHIFT_X}, {SHIFT_Y}) px")
draw(ax[2], top, boxes_top, f"(9): top view, {gsd_top:.3f} m/px, ground distance {ext[2]:.0f}–{ext[3]:.0f} m"
     + (" (MAX_RANGE_M is below the nearest row)" if ext[3] > MAX_RANGE_M + 1 else ""))
plt.tight_layout()
plt.show()

ov = hit.objects[(hit.objects.angle >= 80) & hit.objects.cls.isin(["Person", "Car"])]
rows = []
for a_ in (15, 30, 45, ANGLE_DEG):
    r_ = math.radians(a_)
    wn = np.abs(ov.w * math.cos(r_)) + np.abs(ov.h * math.sin(r_))
    hn = np.abs(ov.w * math.sin(r_)) + np.abs(ov.h * math.cos(r_))
    rows.append({"rotation, deg": a_, "median growth of the box area": float(np.median(wn * hn / (ov.w * ov.h)))})
display(pd.DataFrame(rows).drop_duplicates().set_index("rotation, deg").round(2)
        .style.set_caption(f"Boxes of {len(ov)} persons and cars of HIT-UAV (vertical views) after rotation"))

cars = hit.objects[(hit.objects.cls == "Car") & (hit.objects.angle <= 50)].dropna(subset=["length_px"]).copy()
cars["row"] = (cars.y1 + cars.y2) / 2
across, along = aitbc.oblique_gsd(cars.altitude.values, cars.angle.values, cars.row.values)
cars["central GSD of the frame"] = cars.length_px * [hit.gsd(a, b) for a, b in zip(cars.altitude, cars.angle)]
cars["range of the row"] = cars.length_px * across
cars["ground plane (range and foreshortening)"] = cars.length_px * np.sqrt(across * along)
cars["band of rows"] = pd.cut(cars.row, [0, 128, 256, 384, 512], labels=["0–127 (far)", "128–255", "256–383", "384–511 (near)"])
GSD_CHECK = cars.groupby("band of rows", observed=True)[["central GSD of the frame", "range of the row", "ground plane (range and foreshortening)"]].median()
display(GSD_CHECK.round(2).style.set_caption(f"Median length of {len(cars)} cars of HIT-UAV in oblique frames (camera angle 30–50°), m"))
""")

md(r"""
## Task 3.3.2. Preparation of a thermal frame: grey levels, CLAHE and the polarity of thermal signatures

**Grey levels in one convention.** The thermal cameras of the data sets record the frames in different palettes: white-hot (HIT-UAV: warm = bright), black-hot (M3OT: warm = dark) and a pseudo-colour palette (Kust4K: purple = cold, yellow = warm). Before the processing the frame is converted to one channel of grey levels in the white-hot convention: the luminance of the palette, inverted for black-hot (`aitbc.to_white_hot`). CLAHE (Lesson 2.4) may be applied afterwards.

**Polarity of a signature.** An object is *warm* in the frame when its brightest part differs from the background more than its darkest part:

$$\text{warm:}\quad P_{90}(\text{box}) - \operatorname{med}(\text{ring}) \;\ge\; \operatorname{med}(\text{ring}) - P_{10}(\text{box}), \tag{10}$$

where $P_{90}$, $P_{10}$ are the percentiles of the brightness inside the box of the annotation and the ring is the background around the box.

**Parameters.** `SOURCE`, `NUMBER` — thermal frame; `APPLY_CLAHE`, `CLIP_LIMIT`, `TILES` — CLAHE after the conversion.

**Expected result.** The frame in its palette, the grey frame in the white-hot convention (and after CLAHE), their histograms; the shares of warm and cold signatures of persons, cars and bicycles on 160 frames of HIT-UAV by day and at night.

**What to observe.** At night almost all persons are warm (97 %), while 86 % of cars and 82 % of bicycles are *colder* than the asphalt that kept the heat of the day. By day the picture changes: the ground is heated by the sun, and only two thirds of persons are warmer than their background. A detector of warm spots therefore finds persons at night, but must look for cold spots to find parked cars and fails for persons by day (block 3.3.4). CLAHE raises the visible contrast of the background, which also raises the contrast of the clutter; its effect on the detection is measured in block 3.3.4.

**Control question 6.** Why are parked cars colder than the road at night? Which polarity of the mask is needed to find persons and parked cars in one night frame?
""")

code(r"""
#@title 3.3.2 Preparation of a thermal frame and the polarity of signatures (code shown) { run: "auto" }
SOURCE = "HIT-UAV: night, test frames (white-hot)" #@param ["HIT-UAV: night, test frames (white-hot)", "HIT-UAV: day, test frames (white-hot)", "M3OT 1-08: dusk (black-hot)", "Kust4K: night (pseudo-colour palette)"]
NUMBER = 1 #@param {type:"slider", min:1, max:40, step:1}
APPLY_CLAHE = False #@param {type:"boolean"}
CLIP_LIMIT = 2.0 #@param {type:"slider", min:1, max:8, step:0.5}
TILES = 8 #@param {type:"slider", min:2, max:16, step:1}

GSD_ASSUMED = 0.145      # m/px of M3OT and Kust4K: thermal camera f = 9.1 mm, pixel 12 um, altitude about 110 m (assumption)
k = NUMBER - 1
if SOURCE.startswith("HIT-UAV"):
    files = HIT_EVAL_NIGHT if "night" in SOURCE else HIT_EVAL_DAY
    f = files[k % len(files)]
    raw, objs = hit.load(f)
    r = hit.images.set_index("file").loc[f]
    shown, palette, alt, ang = raw, hit.PALETTE, int(r.altitude), int(r.angle)
    caption = f"HIT-UAV {f}: {alt} m, {ang}°, {'night' if r.night else 'day'}"
elif SOURCE.startswith("M3OT"):
    f0 = (300, 500)[k % 2]
    shown = m3ot.frames("1-08", "ir", start=f0, count=1)[0]
    objs, palette, alt, ang, f = m3ot.objects(m3ot.tracks("1-08", "ir"), f0 + 1), m3ot.PALETTE, None, None, f"m3ot_1-08_{f0 + 1}"
    caption = f"M3OT 1-08, thermal frame {f0 + 1} (black-hot)"
else:
    fid = aitbc.Kust4K.NIGHT[k % 6]
    p = kust.pair(fid)
    shown, palette, alt, ang, f = p["tir"], kust.PALETTE, None, None, f"kust4k_{fid}"
    objs = aitbc.boxes_from_labels(p["labels"], (3, 4, 5, 7), kust.CLASSES, min_area=20)
    caption = f"Kust4K {fid} (night), pseudo-colour palette"
grey = aitbc.to_white_hot(shown, palette)[..., 0]                         # one channel, warm = bright
prepared = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(TILES, TILES)).apply(grey) if APPLY_CLAHE else grey
PREP_33 = dict(file=f, grey=prepared, shown=shown, objects=objs, altitude=alt, angle=ang, caption=caption,
               gsd=hit.gsd(alt, ang) if alt else GSD_ASSUMED, clahe=APPLY_CLAHE)

fig, ax = plt.subplots(1, 4, figsize=(19, 4.3), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]))
ax[0].imshow(shown, cmap="gray" if shown.ndim == 2 else None)
ax[0].set_title(f"{caption}\npalette: {palette}", fontsize=8.5)
ax[1].imshow(grey, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("grey levels, white-hot (warm = bright)", fontsize=9)
ax[2].imshow(prepared, cmap="gray", vmin=0, vmax=255)
ax[2].set_title(f"after CLAHE c = {CLIP_LIMIT:g}, {TILES}×{TILES}" if APPLY_CLAHE else "without CLAHE", fontsize=9)
for a in ax[:3]:
    a.axis("off")
ax[3].hist(grey.ravel(), bins=64, range=(0, 255), alpha=0.6, label="white-hot")
if APPLY_CLAHE:
    ax[3].hist(prepared.ravel(), bins=64, range=(0, 255), alpha=0.6, label="after CLAHE")
ax[3].legend(fontsize=8)
ax[3].set_xlabel("grey level")
ax[3].set_title("histograms", fontsize=9)
plt.tight_layout()
plt.show()

if "POLARITY" not in globals():
    recs = []
    for fp in HIT_POLARITY:
        gp, op_ = hit.load(fp)
        night = bool(hit.images.set_index("file").loc[fp].night)
        for ob in op_:
            if ob["cls"] not in ("Person", "Car", "Bicycle"):
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in ob["box"]]
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            inner = gp[max(0, y1):y2, max(0, x1):x2].astype(float)
            pad = max(4, int(0.5 * max(x2 - x1, y2 - y1)))
            X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(gp.shape[1], x2 + pad), min(gp.shape[0], y2 + pad)
            ring = gp[Y1:Y2, X1:X2].astype(float)
            keep = np.ones(ring.shape, bool)
            keep[max(0, y1) - Y1:y2 - Y1, max(0, x1) - X1:x2 - X1] = False
            bg = np.median(ring[keep])
            warm = np.percentile(inner, 90) - bg >= bg - np.percentile(inner, 10)         # formula (10)
            recs.append(dict(time="night" if night else "day", cls=ob["cls"], warm=warm))
    POLARITY = pd.DataFrame(recs)
pol = POLARITY.groupby(["cls", "time"]).warm.agg(["size", "mean"]).rename(columns={"size": "objects", "mean": "warm share"})
pol["cold share"] = 1 - pol["warm share"]
display(pol.round(2).style.set_caption(f"Polarity (10) of the signatures on {len(HIT_POLARITY)} frames of HIT-UAV (validation and test parts)"))
""")

md(r"""
## Task 3.3.3. Morphological processing of the binary mask

**Morphological operations** with a structuring element $S$ (a disc of the given diameter):

$$I \ominus S = \min_{s \in S} I(x + s), \quad I \oplus S = \max_{s \in S} I(x + s), \quad I \circ S = (I \ominus S) \oplus S, \quad I \bullet S = (I \oplus S) \ominus S, \tag{11}$$

erosion, dilation, opening (removes bright details smaller than $S$) and closing (fills dark gaps smaller than $S$). The *white top-hat* $I - I \circ S$ keeps warm details smaller than $S$, the *black top-hat* $I \bullet S - I$ cold details. The same operations applied to the binary mask remove isolated pixels (opening) and join the parts of one object (closing).

**Pipeline of the mask.** prepared frame (3.3.2) → white / black top-hat with $S$ given in metres → threshold → opening → closing.

**Parameters.** `POLARITY` — warm, cold or both kinds of signatures (10); `MASK_METHOD` — threshold: Otsu on the whole frame, Otsu on the top-hat, mean + $k\sigma$ of the top-hat, adaptive local mean; `SE_M` — diameter of $S$ in metres; `K`; `OPEN_PX`, `CLOSE_PX` — sizes of the opening and closing of the mask in pixels (1 — not applied).

**Expected result.** The stages of the processing and a table for each stage: number of connected regions, share of pixels in the mask and the annotated objects whose centre lies in the mask.

**What to observe.** The top-hat removes the large warm surfaces (roads, roofs) and keeps the small signatures. In the default frame a threshold of mean + 4σ of the top-hat leaves 1.5 % of the pixels. The opening 3 × 3 removes single pixels of noise (60 → 45 regions in the default frame) and separates touching blobs: on the 40 night frames of block 3.3.4 the recall of persons is 0.84 with the opening and 0.81 without it. Check with `OPEN_PX = 5` how many distant persons, which are only 2–4 px wide, remain in the mask. The closing joins the parts of a car, but can merge persons standing close to each other. With `POLARITY = cold` the parked cars of a night frame appear in the mask.

**Control question 7.** Which operation (11) is used to remove isolated noise pixels from a mask, and which to fill gaps inside an object? Why is the size of $S$ set in metres?
""")

code(r"""
#@title 3.3.3 Morphological processing of the mask (code shown) { run: "auto" }
POLARITY = "warm" #@param ["warm", "cold", "both"]
MASK_METHOD = "top-hat + mean + k*std" #@param ["Otsu (global)", "top-hat + Otsu", "top-hat + mean + k*std", "adaptive (local mean)"]
SE_M = 3.0 #@param {type:"slider", min:1, max:15, step:0.5}
K = 4.0 #@param {type:"slider", min:2, max:8, step:0.5}
OPEN_PX = 3 #@param [1, 3, 5] {type:"raw"}
CLOSE_PX = 1 #@param [1, 3, 5, 7] {type:"raw"}


def mask_stages(grey, gsd, polarity, method, se_m, k, open_px, close_px):
    se_px = max(3, int(round(se_m / gsd))) | 1
    raw = aitbc.thermal_mask(grey, polarity, method, se_px=se_px, k=k)        # top-hat (11) and threshold
    opened = raw if open_px <= 1 else cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    closed = opened if close_px <= 1 else cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    return dict(raw=raw, opened=opened, closed=closed, se_px=se_px)


P33 = dict(polarity=POLARITY, method=MASK_METHOD, se_m=SE_M, k=K, open_px=OPEN_PX, close_px=CLOSE_PX)
st = mask_stages(PREP_33["grey"], PREP_33["gsd"], **P33)
MASK_33 = st["closed"]
g = cv2.GaussianBlur(PREP_33["grey"], (3, 3), 0)
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (st["se_px"], st["se_px"]))
th_w = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
th_b = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)

fig, ax = plt.subplots(1, 5, figsize=(20, 4.3))
for a, (im_, t_) in zip(ax, ((PREP_33["grey"], PREP_33["caption"]), (th_w if POLARITY != "cold" else th_b, f"{'white' if POLARITY != 'cold' else 'black'} top-hat, S = {SE_M:g} m = {st['se_px']} px"),
                             (st["raw"], f"threshold: {MASK_METHOD}"), (st["opened"], f"opening {OPEN_PX}×{OPEN_PX}"),
                             (st["closed"], f"closing {CLOSE_PX}×{CLOSE_PX}"))):
    a.imshow(im_, cmap="gray" if im_ is not th_w and im_ is not th_b else "magma")
    a.set_title(t_, fontsize=8.5)
    a.axis("off")
plt.tight_layout()
plt.show()


def centre_hits(mask, objs, classes):
    obs = [o for o in objs if o["cls"] in classes]
    hits = sum(mask[int(np.clip((o["box"][1] + o["box"][3]) / 2, 0, mask.shape[0] - 1)),
                    int(np.clip((o["box"][0] + o["box"][2]) / 2, 0, mask.shape[1] - 1))] > 0 for o in obs)
    return f"{hits} of {len(obs)}" if obs else "—"


rows = []
for name in ("raw", "opened", "closed"):
    m = st[name]
    n, _ = cv2.connectedComponents(m)
    rows.append({"stage": {"raw": "threshold", "opened": "after opening", "closed": "after closing"}[name], "regions": n - 1,
                 "share of pixels": f"{(m > 0).mean():.2%}", "persons with the centre in the mask": centre_hits(m, PREP_33["objects"], ("Person", "person")),
                 "vehicles with the centre in the mask": centre_hits(m, PREP_33["objects"], ("Car", "OtherVehicle", "vehicle", "car", "truck"))})
display(pd.DataFrame(rows).set_index("stage"))
""")

md(r"""
## Task 3.3.4. Contours and their filtering by geometric characteristics

**Filtering.** For every contour of the mask the characteristics (6) are computed, and its area and sizes are converted to metres (7) with the GSD of its row. A contour is kept as a thermal signature when

$$A_{min} \le A_{m^2} \le A_{max}, \qquad \frac{L}{W} \le r_{max}, \qquad s \ge s_{min}, \tag{12}$$

and it is assigned the class *person* when $A_{m^2} <$ `PERSON_MAX_M2`, otherwise *vehicle* (block 3.1.4: a person about 1 m², a car about 13 m²).

**Evaluation.** A detection is correct when its centre lies in a box of the annotation (enlarged by 2 px); the recall is counted separately for persons and vehicles; the precision is the share of correct detections; the class is correct when a person is detected as a person and a vehicle as a vehicle.

**Parameters.** `AREA_MIN_M2`, `AREA_MAX_M2`, `MAX_ASPECT`, `MIN_SOLIDITY` — limits (12); `PERSON_MAX_M2` — boundary of the classes; `GSD_MODEL` — GSD of the row of the contour (task 3.3.1); `EVALUATE`, `EVAL_SET` — evaluation on 40 night or 24 day test frames of HIT-UAV with the mask of block 3.3.3 (about 5 s); `RUN_ABLATION` — the same evaluation with one stage changed at a time (about 30 s).

**Expected result.** The kept and the rejected contours of the current frame with the reasons of rejection; the recall, precision and class correctness on the chosen set; the table of variants.

**What to observe.** On the 40 night test frames (343 persons, 74 vehicles) the pipeline finds 0.84 of the persons and 0.65 of the vehicles with a precision of 0.33; the class is correct for 84 % of the correct detections. The false detections are lamps, warm parts of buildings and edges of trees that have the size of a person. Without the geometric filter the recall of persons grows to 0.92, but the precision falls to 0.22. The global Otsu threshold instead of the top-hat gives a recall of 0.25, CLAHE before the mask lowers it to 0.60 (as in Lesson 2.4). The mask of warm and cold signatures raises the recall of vehicles to 0.80 but lowers that of persons to 0.57: the cold details merge with warm persons and add false contours. The central GSD of the frame instead of the GSD of the row changes the result little on these frames (0.83 against 0.84). By day the recall of persons falls to 0.20 (89 persons on 24 frames): a third of the persons are not warmer than the heated ground (task 3.3.2). The classical pipeline is a fast and explainable detector of warm targets at night; the precision and the work by day require the features learned by the networks of Topics 4 and 5.

**Control question 8.** Which false detections are removed by the limits of the area in m², and which remain? Why does the recall by day fall so strongly?
""")

code(r"""
#@title 3.3.4 Contours and their filtering (code shown) { run: "auto" }
AREA_MIN_M2 = 0.1 #@param {type:"slider", min:0.02, max:1.0, step:0.02}
AREA_MAX_M2 = 40.0 #@param {type:"slider", min:5, max:100, step:5}
MAX_ASPECT = 5.0 #@param {type:"slider", min:1.5, max:10, step:0.5}
MIN_SOLIDITY = 0.6 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
PERSON_MAX_M2 = 2.5 #@param {type:"slider", min:0.5, max:6, step:0.25}
GSD_MODEL = "range of the row" #@param ["central GSD of the frame", "range of the row", "ground plane (range and foreshortening)"]
EVALUATE = True #@param {type:"boolean"}
EVAL_SET = "night: 40 test frames" #@param ["night: 40 test frames", "day: 24 test frames"]
RUN_ABLATION = True #@param {type:"boolean"}

F33 = dict(area_min=AREA_MIN_M2, area_max=AREA_MAX_M2, max_aspect=MAX_ASPECT, min_solidity=MIN_SOLIDITY,
           person_max=PERSON_MAX_M2, gsd_model=GSD_MODEL)


def row_gsd(model, alt, ang, row, gsd_default):
    if alt is None:
        return gsd_default, gsd_default
    if model.startswith("central"):
        v = hit.gsd(alt, ang)
        return v, v
    across, along = aitbc.oblique_gsd(alt, ang, row)
    return (float(across), float(across)) if model.startswith("range") else (float(across), float(along))


def detect_signatures(grey, alt, ang, gsd_default, mask_par, filt, clahe=False):
    if clahe:
        grey = cv2.createCLAHE(2.0, (8, 8)).apply(grey)
    mask = mask_stages(grey, hit.gsd(alt, ang) if alt else gsd_default, **mask_par)["closed"]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 2:
            continue
        cy = aitbc.contour_features(c)["cy"]
        f = aitbc.contour_features(c, *row_gsd(filt["gsd_model"], alt, ang, cy, gsd_default))     # formulas (6)-(7)
        reason = ("area < min" if f["area_m2"] < filt["area_min"] else "area > max" if f["area_m2"] > filt["area_max"]
                  else "elongated" if f["aspect"] > filt["max_aspect"] else "low solidity" if f["solidity"] < filt["min_solidity"] else "")
        f.update(keep=reason == "", reason=reason, contour=c, cls="Person" if f["area_m2"] < filt["person_max"] else "vehicle")   # (12)
        out.append(f)
    return out


def evaluate_pipeline(files, mask_par, filt, clahe=False):
    c = dict(P=0, Pf=0, V=0, Vf=0, det=0, ok=0, cls=0)
    for fname in files:
        gg, objs = hit.load(fname)
        r = hit.images.set_index("file").loc[fname]
        dets = [d for d in detect_signatures(gg, int(r.altitude), int(r.angle), None, mask_par, filt, clahe) if d["keep"]]
        c["det"] += len(dets)
        used = set()
        for d in dets:
            hits = [j for j, o in enumerate(objs) if o["box"][0] - 2 <= d["cx"] <= o["box"][2] + 2 and o["box"][1] - 2 <= d["cy"] <= o["box"][3] + 2]
            if hits:
                c["ok"] += 1
                used.update(hits)
                c["cls"] += (d["cls"] == "Person") == (objs[hits[0]]["cls"] == "Person")
        for j, o in enumerate(objs):
            if o["cls"] == "Person":
                c["P"] += 1
                c["Pf"] += j in used
            elif o["cls"] in ("Car", "OtherVehicle"):
                c["V"] += 1
                c["Vf"] += j in used
    return {"recall of persons": c["Pf"] / max(c["P"], 1), "recall of vehicles": c["Vf"] / max(c["V"], 1),
            "precision": c["ok"] / max(c["det"], 1), "class correct": c["cls"] / max(c["ok"], 1),
            "detections": c["det"], "persons": c["P"], "vehicles": c["V"]}


DETS_33 = detect_signatures(PREP_33["grey"], PREP_33["altitude"], PREP_33["angle"], PREP_33["gsd"], P33, F33)
kept = [d for d in DETS_33 if d["keep"]]
vis_ = cv2.cvtColor(PREP_33["grey"], cv2.COLOR_GRAY2RGB)
cv2.drawContours(vis_, [d["contour"] for d in DETS_33 if not d["keep"]], -1, (230, 40, 40), 1)
cv2.drawContours(vis_, [d["contour"] for d in kept if d["cls"] == "Person"], -1, (0, 229, 255), 2)
cv2.drawContours(vis_, [d["contour"] for d in kept if d["cls"] != "Person"], -1, (255, 235, 59), 2)
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(vis_)
ax.set_title(f"{PREP_33['caption']}: kept {len(kept)} (cyan — person, yellow — vehicle), rejected {len(DETS_33) - len(kept)} (red)", fontsize=9)
ax.axis("off")
plt.show()
print("Rejected contours by reason:", pd.Series([d["reason"] for d in DETS_33 if not d["keep"]]).value_counts().to_dict())

if EVALUATE:
    files = HIT_EVAL_NIGHT if EVAL_SET.startswith("night") else HIT_EVAL_DAY
    EVAL_33 = evaluate_pipeline(files, P33, F33)
    display(pd.DataFrame({"Value": EVAL_33}).round(3).style.set_caption(f"Evaluation on HIT-UAV, {EVAL_SET}"))
    if RUN_ABLATION:
        variants = {"pipeline of blocks 3.3.3-3.3.4": (P33, F33, False),
                    "without geometric filter (12)": (P33, dict(F33, area_min=0, area_max=1e9, max_aspect=1e9, min_solidity=0), False),
                    "global Otsu instead of the top-hat": (dict(P33, method="Otsu (global)"), F33, False),
                    "top-hat + Otsu": (dict(P33, method="top-hat + Otsu"), F33, False),
                    "CLAHE before the mask": (P33, F33, True),
                    "without opening": (dict(P33, open_px=1), F33, False),
                    "structuring element 8 m": (dict(P33, se_m=8.0), F33, False),
                    "polarity: warm and cold": (dict(P33, polarity="both"), F33, False),
                    "central GSD of the frame": (P33, dict(F33, gsd_model="central GSD of the frame"), False)}
        t0 = time.time()
        ABL_33 = pd.DataFrame({name: evaluate_pipeline(files, mp, fp, cl) for name, (mp, fp, cl) in variants.items()}).T
        display(ABL_33.round(3).style.set_caption(f"Variants of the pipeline, {EVAL_SET} ({time.time() - t0:.0f} s)"))
        ABL_33.to_csv(os.path.join(REPORT_DIR, "lesson33_variants.csv"))
""")

md(r"""
## Task 3.3.5. Visualisation: contours and boxes on the original frame

**Purpose.** To present the result of the module in the form used by an operator: the original frame in its palette with the contours or boxes of the detected signatures, the class and the size in metres, and — for checking — the boxes of the annotation.

**Parameters.** `OVERLAY` — contours, boxes or both; `SHOW_ANNOTATION` — boxes of the annotation (white, thin); `LABELS` — class and area in m² near each detection; `SAVE` — save the figure and the table of detections in the project.

**Expected result.** The figure `outputs/figures/lesson33_<frame>.png`, the table `outputs/reports/lesson33_detections.csv` (class, centre, box, area and sizes in metres, aspect, solidity) and the record of the lesson in `project.json`.

**What to observe.** On a night frame the persons are found as small warm contours; the boxes of the annotation show the missed persons (for example adjacent persons merged into one contour) and the false detections without an annotation box.
""")

code(r"""
#@title 3.3.5 Visualisation of the results (code shown) { run: "auto" }
OVERLAY = "contours and boxes" #@param ["contours", "boxes", "contours and boxes"]
SHOW_ANNOTATION = True #@param {type:"boolean"}
LABELS = True #@param {type:"boolean"}
SAVE = True #@param {type:"boolean"}

base = PREP_33["shown"]
out = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) if base.ndim == 2 else base.copy()
if SHOW_ANNOTATION:
    for o in PREP_33["objects"]:
        x1, y1, x2, y2 = [int(round(v)) for v in o["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
for d in kept:
    col = (0, 229, 255) if d["cls"] == "Person" else (255, 235, 59)
    if OVERLAY != "boxes":
        cv2.drawContours(out, [d["contour"]], -1, col, 1)
    if OVERLAY != "contours":
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), col, 1)
    if LABELS:
        cv2.putText(out, f"{'P' if d['cls'] == 'Person' else 'V'} {d['area_m2']:.1f}", (int(d["box"][0]), max(8, int(d["box"][1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
fig, ax = plt.subplots(figsize=(11, 8.8))
ax.imshow(out)
ax.set_title(f"{PREP_33['caption']}: {sum(d['cls'] == 'Person' for d in kept)} persons (P), {sum(d['cls'] != 'Person' for d in kept)} vehicles (V); "
             f"white — annotation", fontsize=9)
ax.axis("off")
if SAVE:
    name = os.path.splitext(os.path.basename(str(PREP_33["file"])))[0]
    fig_path = os.path.join(FIG_DIR, f"lesson33_{name}.png")
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
plt.show()
table = pd.DataFrame([{k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in d.items() if k_ not in ("contour", "keep", "reason")} for d in kept])
display(table.head(10))
if SAVE:
    table.to_csv(os.path.join(REPORT_DIR, "lesson33_detections.csv"), index=False)
    update_project_json("3.3", {"lesson_3_3": dict(frame=PREP_33["caption"], clahe=PREP_33["clahe"], mask=P33, filter=F33,
                                                   detections=len(kept), figure=os.path.relpath(fig_path, PROJECT_ROOT),
                                                   evaluation=({k_: round(float(v), 3) for k_, v in EVAL_33.items()} if EVALUATE else None))})
""")

md(r"""
### Report of Lesson 3.3

Double-click this cell and write the answers.

1. Frame, parameters of (8) and (9), growth of the boxes after rotation, the GSD model chosen and why; answer to control question 5:
2. Polarity of the signatures of persons and cars by day and at night; answer to control question 6:
3. Parameters of the mask and the effect of each stage; answer to control question 7:
4. Limits (12), recall, precision and class correctness on the night and day sets, the most important variant of the table; answer to control question 8:
5. Saved figure and table of detections:
""")


# =====================================================================================
md(r"""
---
# Lesson 3.4 (Practical class). Stages of the stabilisation of the video stream of an FPV drone

**Learning questions**
1. Reading of a sequence of frames of a test FPV video fragment.
2. Detection of ORB keypoints on a pair of consecutive frames.
3. Matching of the detected keypoints and quantitative estimation of the inter-frame shift.
4. Computation of the homography matrix and geometric alignment of frames.
5. Saving of the stabilised video stream and comparative analysis of the results of the processing.

**Data.** Flight 16p of the TII-RATM data set: a human FPV pilot flies a racing quadrotor through the gates of an indoor arena (31 s, 337 m). The onboard camera (Arducam IMX219, 640 × 480 px, wide-angle lens 155° × 115°, tilted up by 50°) records 125 frames per second; the gyroscope of the flight controller measures the angular rate at 500 Hz. The gyroscope is an independent sensor: it is used here as the reference of the rotation of the camera between frames, which the image methods estimate from the frames alone.

**Result of the lesson.** The FPV fragment in `data/video`, the measured dependence of the matching on the frame rate, the comparison of the motion models, the stabilised video (original and stabilised side by side) in `outputs/video`, the table of the comparative analysis in `outputs/reports` and the parameters in `project.json`.
""")

md(r"""
## Task 3.4.1. Reading a sequence of frames of a test FPV video fragment

**Purpose.** To prepare a test FPV fragment as a video file, as it is recorded by the video receiver of an operator, to read it frame by frame with OpenCV and to estimate how fast the camera rotates between frames.

**Rotation between frames.** With the frame rate $f$ and every $n$-th frame used, the interval is $\Delta t = n / f$, and the rotation of the camera between two used frames is

$$\varphi \approx |\boldsymbol\omega|\,\Delta t, \tag{13}$$

where $\boldsymbol\omega$ is the angular rate measured by the gyroscope (in the axes of the camera: x — right, y — down, z — along the optical axis).

**Parameters.** `SEGMENT` — fragment of the flight (A — take-off and the first gates, B — slalom between gates; 4 s each); `FRAME_STEP` — every $n$-th frame (1 — 125 fps, 2 — 63 fps, 4 — 31 fps, 8 — 16 fps, as with a slower camera or a reduced video link).

**Expected result.** The video file `data/video/fpv_tii16p_<A|B>_<fps>fps.mp4` and its properties read by OpenCV, frames of the fragment, the angular rate of the camera and the distribution of the rotation (13) between the used frames.

**What to observe.** The first run downloads the camera archive of the flight once (about 200 MB, 1–3 min) and keeps the frames of both fragments in the cache. The pilot turns the drone at up to about 790 °/s. In fragment B the median rotation between consecutive frames is 1.0° at 126 fps (95th percentile 6°), 2.0° at 63 fps, 4.1° at 32 fps and 8.6° at 16 fps (95th percentile 47°): the content of consecutive frames overlaps less and less, which is the main condition of the matching of task 3.4.3.

**Control question 9.** How does the rotation between frames change when every 4th frame is used? Why is the frame rate of an FPV camera more important for stabilisation than its resolution?
""")

code(r"""
#@title 3.4.1 Test FPV video fragment (code shown) { display-mode: "form" }
SEGMENT = "B: slalom between gates (15.0-19.0 s)" #@param ["B: slalom between gates (15.0-19.0 s)", "A: take-off and first gates (0.5-4.5 s)"]
FRAME_STEP = 2 #@param [1, 2, 4, 8] {type:"raw"}

t_start, t_end = tii.SEGMENTS[tii.flight][SEGMENT]
t0 = time.time()
clip_all, clip_tab = tii.clip(t_start, t_end, step=1)                    # frames of the data set, 125 fps
times_all = clip_tab.t.values
fps_cam = 1.0 / np.median(np.diff(times_all))
frames_src = clip_all[::FRAME_STEP]
fps_eff = fps_cam / FRAME_STEP
seg = SEGMENT[0]
video_path = os.path.join(PROJECT_ROOT, "data", "video", f"fpv_tii16p_{seg}_{fps_eff:.0f}fps.mp4")
aitbc.write_video(video_path, frames_src, fps_eff)
info = aitbc.video_info(video_path)
cap = cv2.VideoCapture(video_path)                                         # reading frame by frame
video_frames = []
while True:
    ok, fr_ = cap.read()
    if not ok:
        break
    video_frames.append(cv2.cvtColor(fr_, cv2.COLOR_BGR2RGB))
cap.release()

imu = tii.imu()
R_cb = tii.camera_to_body()                                                # camera axes -> body axes
w_cam = imu[["gyro_x", "gyro_y", "gyro_z"]].values @ R_cb                 # rows: (R_cb^T w_body)^T
m = (imu.t >= t_start) & (imu.t <= t_end)
times = times_all[::FRAME_STEP]
phi = np.array([np.degrees(np.linalg.norm(aitbc.gyro_rotation(imu.t.values, w_cam, a, b))) for a, b in zip(times[:-1], times[1:])])   # (13)
ROT_STEPS = {}
for st_ in (1, 2, 4, 8):
    tt_ = times_all[::st_]
    ph_ = [np.degrees(np.linalg.norm(aitbc.gyro_rotation(imu.t.values, w_cam, a, b))) for a, b in zip(tt_[:-1], tt_[1:])]
    ROT_STEPS[st_] = {"fps": round(fps_cam / st_), "median φ, °": float(np.median(ph_)), "95th percentile φ, °": float(np.percentile(ph_, 95))}
FPV = dict(segment=SEGMENT, step=FRAME_STEP, fps=fps_eff, frames=video_frames, times=times, path=video_path, imu_t=imu.t.values,
           w_cam=w_cam, clip_all=[cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in clip_all], times_all=times_all)   # grey frames at 125 fps
del clip_all, frames_src

fig = plt.figure(figsize=(19, 7.5))
gs_ = fig.add_gridspec(2, 6)
for j, i in enumerate(np.linspace(0, len(video_frames) - 1, 6).astype(int)):
    a = fig.add_subplot(gs_[0, j])
    a.imshow(video_frames[i])
    a.set_title(f"frame {i}, t = {times[i]:.2f} s", fontsize=8.5)
    a.axis("off")
a = fig.add_subplot(gs_[1, :4])
for k_, lab in enumerate(("ω_x (tilt)", "ω_y (pan)", "ω_z (roll about the optical axis)")):
    a.plot(imu.t[m], np.degrees(w_cam[m.values, k_]), lw=0.7, label=lab)
a.set_xlabel("time of the flight, s")
a.set_ylabel("angular rate, °/s")
a.legend(fontsize=8)
a.grid(alpha=0.3)
a.set_title("Gyroscope in the axes of the camera", fontsize=9)
a = fig.add_subplot(gs_[1, 4:])
a.hist(phi, bins=40, color="C2")
a.set_xlabel(f"rotation between used frames φ (13), ° (Δt = {1000 / fps_eff:.1f} ms)")
a.set_title(f"median {np.median(phi):.2f}°, 95th percentile {np.percentile(phi, 95):.2f}°", fontsize=9)
a.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"lesson34_fragment_{seg}.png"), dpi=90)
plt.show()
display(pd.DataFrame({"Value": {"fragment": SEGMENT, "frames read from the video file": len(video_frames),
                                "frame rate of the camera (time stamps), fps": round(fps_cam, 1), "used frame rate, fps": round(fps_eff, 1),
                                "size of a frame": f"{info['width']} × {info['height']}", "codec / size of the file": f"{info['codec']} / {info['size_mb']:.1f} MB",
                                "largest angular rate in the fragment, °/s": round(float(np.degrees(np.linalg.norm(w_cam[m.values], axis=1)).max())),
                                "time of preparation, s": round(time.time() - t0)}}))
display(pd.DataFrame(ROT_STEPS).T.rename_axis("frame step").round(2).style.set_caption("Rotation of the camera between used frames (13) for every frame step"))
""", form=True)

md(r"""
## Task 3.4.2. ORB keypoints on a pair of consecutive frames

**ORB** (Oriented FAST and Rotated BRIEF). (a) *FAST*: a pixel is a corner when on the circle of 16 pixels around it there are at least 9 consecutive pixels brighter or darker than its brightness by more than the threshold $t$; (b) a pyramid of images with the scale factor $s$ gives keypoints of different sizes; (c) the orientation of a keypoint is the direction of the intensity centroid of its neighbourhood,

$$\theta = \operatorname{atan2}(m_{01}, m_{10}), \qquad m_{pq} = \sum_{x, y} x^p y^q I(x, y); \tag{14}$$

(d) *rotated BRIEF*: the descriptor is a string of 256 bits, each bit compares the brightness of two pixels of a fixed pattern turned by $\theta$. The descriptor does not change when the frame turns, which is necessary for an FPV camera that rolls.

**Lens distortion.** The wide-angle lens bends straight lines; the frames are corrected with the camera matrix and the distortion coefficients of the data set (`cv2.initUndistortRectifyMap`). A homography (task 3.4.4) describes the motion of a camera without distortion.

**Parameters.** `PAIR_POSITION` — position of the pair in the fragment; `N_FEATURES` — largest number of keypoints; `FAST_THRESHOLD` — $t$; `SCALE_FACTOR`, `N_LEVELS` — pyramid; `UNDISTORT` — correction of the distortion.

**Expected result.** The keypoints on both frames (circle — size, line — orientation), their number, the coverage of the frame (share of the cells of a grid 8 × 6 with at least one keypoint) and the time of the detection.

**What to observe.** The keypoints concentrate on the gates, the lamps and the edges of windows; the floor and the walls without texture have few of them. A lower FAST threshold gives more keypoints and a better coverage, but more of them lie on noise and on motion blur. After the correction of the distortion the black corners of the frame contain no keypoints.

**Control question 10.** Why is the orientation (14) computed for every keypoint? What limits the use of keypoints on the floor of the arena?
""")

code(r"""
#@title 3.4.2 ORB keypoints (code shown) { run: "auto" }
PAIR_POSITION = 0.5 #@param {type:"slider", min:0, max:1, step:0.01}
N_FEATURES = 1000 #@param {type:"slider", min:200, max:3000, step:100}
FAST_THRESHOLD = 20 #@param {type:"slider", min:5, max:60, step:1}
SCALE_FACTOR = 1.2 #@param {type:"slider", min:1.1, max:2.0, step:0.05}
N_LEVELS = 8 #@param {type:"slider", min:1, max:12, step:1}
UNDISTORT = True #@param {type:"boolean"}

K_cam, D_cam = tii.calibration()
map1, map2, K_und = aitbc.undistort_maps(K_cam, D_cam, (640, 480), alpha=0.0)
K_use = K_und if UNDISTORT else K_cam


def prepare(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return cv2.remap(g, map1, map2, cv2.INTER_LINEAR) if UNDISTORT else g


levels = min(N_LEVELS, int(math.log(480 / 63.0) / math.log(SCALE_FACTOR)) + 1)    # the smallest level must hold a 31 x 31 patch
if levels < N_LEVELS:
    print(f"N_LEVELS reduced to {levels}: with the scale factor {SCALE_FACTOR} a deeper level would be smaller than the ORB patch")
orb = cv2.ORB_create(nfeatures=N_FEATURES, scaleFactor=SCALE_FACTOR, nlevels=levels, fastThreshold=FAST_THRESHOLD)
i_a = int(round(PAIR_POSITION * (len(FPV["frames"]) - 2)))
i_b = i_a + 1
ga, gb = prepare(FPV["frames"][i_a]), prepare(FPV["frames"][i_b])
t_orb = aitbc.time_call(lambda: orb.detectAndCompute(ga, None), repeats=3)
ka, da = orb.detectAndCompute(ga, None)
kb, db = orb.detectAndCompute(gb, None)


def coverage(kps, shape, nx=8, ny=6):
    cells = {(int(k.pt[0] * nx / shape[1]), int(k.pt[1] * ny / shape[0])) for k in kps}
    return len(cells) / (nx * ny)


PAIR = dict(ia=i_a, ib=i_b, ga=ga, gb=gb, ka=ka, da=da, kb=kb, db=db, K=K_use, undistort=UNDISTORT)
fig, ax = plt.subplots(1, 2, figsize=(18, 6.8))
for a, g_, k_, i_ in ((ax[0], ga, ka, i_a), (ax[1], gb, kb, i_b)):
    a.imshow(cv2.drawKeypoints(cv2.cvtColor(g_, cv2.COLOR_GRAY2RGB), k_, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    a.set_title(f"frame {i_} (t = {FPV['times'][i_]:.3f} s): {len(k_)} keypoints, coverage {coverage(k_, g_.shape):.0%}", fontsize=9)
    a.axis("off")
plt.tight_layout()
plt.show()
print(f"Time of ORB detection and description of one frame: {t_orb:.1f} ms; distortion corrected: {UNDISTORT}")
""")

md(r"""
## Task 3.4.3. Matching of keypoints and quantitative estimation of the inter-frame shift

**Matching.** The distance of two binary descriptors is the Hamming distance — the number of different bits:

$$d_H(\mathbf a, \mathbf b) = \sum_{i=1}^{256} a_i \oplus b_i, \qquad \text{match if } \frac{d_1}{d_2} < \rho, \tag{15}$$

where $d_1$, $d_2$ are the distances to the nearest and the second nearest descriptor of the other frame (ratio test). Cross-check keeps a pair only when each descriptor is the nearest one for the other.

**Shift and rotation of the content.** The displacement of a matched pair is $\Delta \mathbf p = \mathbf p_b - \mathbf p_a$; the median of the displacements is the estimate of the inter-frame shift. A similarity transformation estimated by RANSAC (`cv2.estimateAffinePartial2D`) also gives the rotation of the content about the image centre; it equals the roll of the camera about the optical axis with the opposite sign and is compared with the gyroscope.

**Parameters.** `RATIO` — $\rho$; `CROSS_CHECK`; `SHOW_MATCHES` — number of drawn pairs; `COMPARE_STEPS` — the same measurements on 60 pairs of the fragment for every frame step 1, 2, 4, 8 (about 15 s).

**Expected result.** The matched pairs, the field of displacements, the histogram of the shifts; for the pair — the median shift and the rotation of the content compared with the gyroscope; the table of the matching for four frame rates.

**What to observe.** At 126 fps the median shift of the matched keypoints is about 4 px, the share of inliers about 0.8 and 3–7 % of the pairs fail (fragments A and B); at 16 fps the shift is about 26 px, the share of inliers falls to about 0.45 and 27–33 % of the pairs fail. The rotation of the content follows the roll measured by the gyroscope with the opposite sign: at 63 fps the correlation is 0.90–0.94 and the median error 0.2–0.3°. At 16 fps in fragment B the estimate of the rotation becomes unreliable (correlation −0.66): strongly rotated frames give many wrong matches. The frame rate determines whether a stabilisation by image matching is possible.

**Control question 11.** Why is the ratio test (15) used instead of a fixed threshold of the Hamming distance? Why does the rotation of the content have the opposite sign to the roll of the camera?
""")

code(r"""
#@title 3.4.3 Matching and the inter-frame shift (code shown) { run: "auto" }
RATIO = 0.75 #@param {type:"slider", min:0.5, max:0.95, step:0.01}
CROSS_CHECK = False #@param {type:"boolean"}
SHOW_MATCHES = 80 #@param {type:"slider", min:10, max:300, step:10}
COMPARE_STEPS = True #@param {type:"boolean"}

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=CROSS_CHECK)


def match(da, db):
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return []
    if CROSS_CHECK:
        return sorted(bf.match(da, db), key=lambda m: m.distance)
    return [p[0] for p in bf.knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < RATIO * p[1].distance]   # (15)


def roll_gyro(t1, t2):
    return np.degrees(aitbc.gyro_rotation(FPV["imu_t"], FPV["w_cam"], t1, t2)[2])


mm = match(PAIR["da"], PAIR["db"])
pa = np.float32([PAIR["ka"][m.queryIdx].pt for m in mm])
pb = np.float32([PAIR["kb"][m.trainIdx].pt for m in mm])
A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0) if len(mm) >= 3 else (None, None)
d = pb - pa
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw=dict(width_ratios=[2, 1, 0.8]))
sel = mm[:SHOW_MATCHES]
ax[0].imshow(cv2.drawMatches(PAIR["ga"], PAIR["ka"], PAIR["gb"], PAIR["kb"], sel, None, matchColor=(0, 255, 0), flags=2))
ax[0].set_title(f"{len(mm)} matches ({'cross-check' if CROSS_CHECK else f'ratio test ρ = {RATIO}'}), {len(sel)} drawn", fontsize=9)
ax[0].axis("off")
ax[1].imshow(PAIR["gb"], cmap="gray")
if len(mm):
    good = inl.ravel() > 0 if inl is not None else np.ones(len(mm), bool)
    ax[1].quiver(pa[good, 0], pa[good, 1], d[good, 0], d[good, 1], color="lime", angles="xy", scale_units="xy", scale=1, width=0.003)
    ax[1].quiver(pa[~good, 0], pa[~good, 1], d[~good, 0], d[~good, 1], color="red", angles="xy", scale_units="xy", scale=1, width=0.003)
ax[1].set_title("displacements (green — inliers, red — outliers)", fontsize=9)
ax[1].axis("off")
ax[2].hist(np.hypot(d[:, 0], d[:, 1]), bins=40, color="C0")
ax[2].set_xlabel("|Δp|, px")
ax[2].set_title("shifts of the matched keypoints", fontsize=9)
plt.tight_layout()
plt.show()
ang_img = math.degrees(math.atan2(A[1, 0], A[0, 0])) if A is not None else float("nan")
roll = roll_gyro(FPV["times"][PAIR["ia"]], FPV["times"][PAIR["ib"]])
display(pd.DataFrame({"Value": {"matches": len(mm), "inliers of the similarity (RANSAC, 3 px)": int(inl.sum()) if inl is not None else 0,
                                "median shift, px": round(float(np.median(np.hypot(d[:, 0], d[:, 1]))), 1) if len(mm) else None,
                                "median shift (x, y), px": tuple(np.round(np.median(d, axis=0), 1)) if len(mm) else None,
                                "rotation of the content, °": round(ang_img, 2),
                                "roll of the camera by the gyroscope, °": round(roll, 2)}}))

if COMPARE_STEPS:
    t0 = time.time()
    rows = []
    clip, tt = FPV["clip_all"], FPV["times_all"]
    for step in (1, 2, 4, 8):
        starts = np.linspace(0, len(clip) - step - 1, 60).astype(int)
        recs = []
        for s_ in starts:
            ga_, gb_ = prepare(clip[s_]), prepare(clip[s_ + step])
            (k1, d1), (k2, d2) = orb.detectAndCompute(ga_, None), orb.detectAndCompute(gb_, None)
            m_ = match(d1, d2)
            r_ = dict(matches=len(m_), inliers=0, share=np.nan, shift_px=np.nan, ang=np.nan, roll=roll_gyro(tt[s_], tt[s_ + step]))
            if len(m_) >= 10:
                q1 = np.float32([k1[m.queryIdx].pt for m in m_])
                q2 = np.float32([k2[m.trainIdx].pt for m in m_])
                A_, in_ = cv2.estimateAffinePartial2D(q1, q2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if A_ is not None:
                    r_.update(inliers=int(in_.sum()), share=float(in_.mean()), ang=math.degrees(math.atan2(A_[1, 0], A_[0, 0])),
                              shift_px=float(np.median(np.hypot(*(q2 - q1)[in_.ravel() > 0].T))))
            recs.append(r_)
        R = pd.DataFrame(recs)
        okr = R[R.inliers >= 15]
        rows.append({"frame step": step, "fps": round(fps_cam / step), "median matches": R.matches.median(),
                     "median share of inliers": okr.share.median(), "median shift, px": okr.shift_px.median(),
                     "failed pairs (< 15 inliers)": float((R.inliers < 15).mean()),
                     "corr(content rotation, −roll)": float(np.corrcoef(okr.ang, -okr.roll)[0, 1]) if len(okr) > 3 else np.nan,
                     "median |error| of the rotation, °": float(np.median(np.abs(okr.ang + okr.roll))) if len(okr) else np.nan,
                     "median |roll| by the gyroscope, °": float(np.median(np.abs(R.roll)))})
    MATCH_STEPS = pd.DataFrame(rows).set_index("frame step")
    display(MATCH_STEPS.round(3).style.set_caption(f"Matching on 60 pairs of fragment {SEGMENT[0]} for every frame step ({time.time() - t0:.0f} s)"))
    MATCH_STEPS.to_csv(os.path.join(REPORT_DIR, "lesson34_matching.csv"))
""")

md(r"""
## Task 3.4.4. Homography and geometric alignment of frames

**Models of the inter-frame motion** (homogeneous coordinates $\tilde{\mathbf x} = (x, y, 1)^{\mathsf T}$):
- translation — 2 parameters $(t_x, t_y)$;
- similarity — 4 parameters (shift, rotation, scale);
- homography — 8 parameters:

$$\tilde{\mathbf x}_b \sim H\,\tilde{\mathbf x}_a, \qquad H = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix}. \tag{16}$$

A homography describes exactly the images of a camera that only rotates ($H = K R K^{-1}$) and the images of a flat scene. It is estimated by RANSAC: a model is computed from a random minimal set of $m$ pairs ($m$ = 4 for $H$), the pairs with the reprojection error below the threshold are counted, and the best model is refined on its inliers. The number of iterations for the probability $p$ to draw at least one set without outliers when the share of inliers is $w$:

$$N = \frac{\ln(1 - p)}{\ln(1 - w^m)}. \tag{17}$$

**Alignment.** The first frame of the pair is transformed by the model (`cv2.warpPerspective`) and compared with the second one.

**Parameters.** `RANSAC_THRESHOLD` — largest reprojection error of an inlier, px; `CONFIDENCE` — $p$ in (17).

**Expected result.** The table of the three models: inliers, median reprojection error, mean absolute difference of the frames after alignment; the difference of the frames before and after the alignment with the homography; the rotation of the camera computed from $H$ (pure rotation model) compared with the gyroscope; $N$ by (17).

**What to observe.** In the default pair the homography has the most inliers (109 of 146 against 93 for the similarity) and the smallest difference after alignment (2.6 grey levels against 3.6 without alignment). The residual difference remains at the near objects (gates, the floor), because the drone also moves forward at several metres per second and the scene of a hall is not flat (parallax). For the same reason the rotation computed from $H$ agrees with the gyroscope only approximately; on 55 pairs of fragment B the correlation is 0.85 for the roll, 0.84 for the pan and 0.57 for the tilt, which is distorted most by the forward motion. With a lower frame rate the homography is fitted to fewer inliers and can align a pair worse than the similarity. With $w$ = 0.6 and $p$ = 0.99 formula (17) gives 34 iterations for a homography, with $w$ = 0.75 — 13.

**Control question 12.** In which two cases does a homography describe the motion between two frames exactly? Why does the alignment of an FPV frame with a homography leave a residual difference at near objects?
""")

code(r"""
#@title 3.4.4 Homography and alignment (code shown) { run: "auto" }
RANSAC_THRESHOLD = 3.0 #@param {type:"slider", min:0.5, max:10, step:0.5}
CONFIDENCE = 0.99 #@param {type:"slider", min:0.9, max:0.999, step:0.001}

ga, gb = PAIR["ga"], PAIR["gb"]
valid_a = cv2.remap(np.full(ga.shape, 255, np.uint8), map1, map2, cv2.INTER_NEAREST) > 0 if PAIR["undistort"] else np.ones(ga.shape, bool)


def reproj_error(Hm, p1, p2):
    q = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    return np.hypot(*(q - p2).T)


models = {}
t_med = np.median(pb - pa, axis=0)
models["translation (median shift)"] = np.array([[1, 0, t_med[0]], [0, 1, t_med[1]], [0, 0, 1]], float)
A_s, _ = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD, confidence=CONFIDENCE)
models["similarity (4 parameters)"] = np.vstack([A_s, [0, 0, 1]])
H_, inl_h = cv2.findHomography(pa, pb, cv2.RANSAC, RANSAC_THRESHOLD, confidence=CONFIDENCE)    # formula (16)
if H_ is None:
    H_, inl_h = models["similarity (4 parameters)"], np.ones((len(pa), 1), np.uint8)
    print("The homography could not be estimated for this pair; the similarity is used instead.")
models["homography (8 parameters)"] = H_
diff0 = np.abs(gb.astype(np.float32) - ga.astype(np.float32))[valid_a].mean()
rows, warped = [], {}
for name, Hm in models.items():
    e = reproj_error(Hm, pa, pb)
    wa = cv2.warpPerspective(ga, Hm, (ga.shape[1], ga.shape[0]))
    wv = cv2.warpPerspective(valid_a.astype(np.uint8), Hm, (ga.shape[1], ga.shape[0])) > 0
    warped[name] = (wa, wv)
    rows.append({"model": name, "inliers (error < threshold)": int((e < RANSAC_THRESHOLD).sum()),
                 "median reprojection error, px": float(np.median(e)),
                 "mean |difference| after alignment": float(np.abs(gb.astype(np.float32) - wa.astype(np.float32))[wv & valid_a].mean())})
rows.insert(0, {"model": "without alignment", "inliers (error < threshold)": int((np.hypot(*(pb - pa).T) < RANSAC_THRESHOLD).sum()),
                "median reprojection error, px": float(np.median(np.hypot(*(pb - pa).T))), "mean |difference| after alignment": float(diff0)})
display(pd.DataFrame(rows).set_index("model").round(2).style.set_caption(f"Pair of frames {PAIR['ia']}–{PAIR['ib']}, {len(pa)} matches"))

wa, wv = warped["homography (8 parameters)"]
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
ax[0].imshow(np.dstack([gb, ga, ga]))
ax[0].set_title("before alignment: frame b (red) over frame a (cyan)", fontsize=9)
ax[1].imshow(np.abs(gb.astype(np.int16) - ga.astype(np.int16)), cmap="magma", vmin=0, vmax=80)
ax[1].set_title(f"|b − a|: mean {diff0:.1f}", fontsize=9)
dd = np.abs(gb.astype(np.int16) - wa.astype(np.int16)) * (wv & valid_a)
ax[2].imshow(dd, cmap="magma", vmin=0, vmax=80)
ax[2].set_title(f"|b − H(a)| after the homography: mean {dd[wv & valid_a].mean():.1f}", fontsize=9)
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()

rv_img = aitbc.rotation_from_homography(H_, PAIR["K"])
rv_gyro = aitbc.gyro_rotation(FPV["imu_t"], FPV["w_cam"], FPV["times"][PAIR["ia"]], FPV["times"][PAIR["ib"]])
w_in = float(inl_h.mean())
display(pd.DataFrame({"from the homography (pure rotation)": np.degrees(-rv_img), "gyroscope": np.degrees(rv_gyro)},
                     index=["tilt about x, °", "pan about y, °", "roll about the optical axis, °"]).round(2)
        .style.set_caption("Rotation of the camera between the frames of the pair" + ("" if PAIR["undistort"] else " (distortion not corrected: K of the lens used)")))
n_iter = 1 if w_in ** 4 >= 1 - 1e-9 else (math.inf if w_in <= 0 else math.ceil(math.log(1 - CONFIDENCE) / math.log(1 - w_in ** 4)))
print(f"Share of inliers of the homography w = {w_in:.2f}: N = ln(1 − p)/ln(1 − w⁴) = {n_iter} iterations for p = {CONFIDENCE} (formula 17)")

t0 = time.time()
cmp_rows = []
for s_ in np.linspace(0, len(FPV["frames"]) - 2, 60).astype(int):
    g1, g2 = prepare(FPV["frames"][s_]), prepare(FPV["frames"][s_ + 1])
    (k1, d1), (k2, d2) = orb.detectAndCompute(g1, None), orb.detectAndCompute(g2, None)
    m_ = match(d1, d2)
    if len(m_) < 15:
        continue
    q1 = np.float32([k1[m.queryIdx].pt for m in m_])
    q2 = np.float32([k2[m.trainIdx].pt for m in m_])
    Hq, iq = cv2.findHomography(q1, q2, cv2.RANSAC, RANSAC_THRESHOLD, confidence=CONFIDENCE)
    if Hq is None or iq.sum() < 15:
        continue
    rv_h = aitbc.rotation_from_homography(Hq, PAIR["K"])
    if not np.all(np.isfinite(rv_h)):
        continue
    cmp_rows.append(np.r_[np.degrees(-rv_h),
                          np.degrees(aitbc.gyro_rotation(FPV["imu_t"], FPV["w_cam"], FPV["times"][s_], FPV["times"][s_ + 1]))])
C = np.array(cmp_rows).reshape(-1, 6)
ROT_CHECK = pd.DataFrame({"correlation": [np.corrcoef(C[:, j], C[:, j + 3])[0, 1] for j in range(3)],
                          "median |error|, °": [np.median(np.abs(C[:, j] - C[:, j + 3])) for j in range(3)],
                          "median |rotation| by the gyroscope, °": [np.median(np.abs(C[:, j + 3])) for j in range(3)]},
                         index=["tilt about x", "pan about y", "roll about the optical axis"])
display(ROT_CHECK.round(2).style.set_caption(f"Rotation from the homography against the gyroscope on {len(C)} pairs of the fragment ({time.time() - t0:.0f} s)"))
""")

md(r"""
## Task 3.4.5. Saving of the stabilised video stream and comparative analysis

**Stabilisation by smoothing of the trajectory.** For every pair of consecutive frames the similarity transformation gives the shift and rotation of the content $(\Delta x_k, \Delta y_k, \Delta\alpha_k)$ (task 3.4.3; when a pair fails, the motion of the previous pair is kept). Their sums form the trajectory of the camera in the image, which is smoothed by a moving average over $2r + 1$ frames:

$$\mathbf T_k = \sum_{j \le k} (\Delta x_j, \Delta y_j, \Delta\alpha_j), \qquad \bar{\mathbf T}_k = \frac{1}{2r + 1}\sum_{j = k - r}^{k + r} \mathbf T_j, \tag{18}$$

and every frame is transformed by the correction $\bar{\mathbf T}_k - \mathbf T_k$: the slow, intended motion of the pilot remains, the fast oscillations are removed. The borders of the transformed frames that have no data are black or are cut off by a zoom.

**Measures of the comparison.**
- *high-frequency jitter* — standard deviation of the trajectory around its moving average with $r$ = 0.1 s, in pixels (shift) and degrees (rotation); for the stabilised video the motion is measured again on its frames (keypoints only in the valid part; pairs without an estimate count as zero motion);
- *jitter of the rotation by the gyroscope* — the same measure for the rotation of the content computed from the gyroscope, $-\int \omega_z\,dt$, before and after the correction: an independent check that does not depend on the image matching;
- *inter-frame transformation fidelity* (higher means a steadier video):

$$\mathrm{ITF} = \frac{1}{N - 1}\sum_{k=2}^{N} \mathrm{PSNR}(I_{k-1}, I_k); \tag{19}$$

- *share of valid pixels* — the part of the frame that keeps image data after the correction.

**Parameters.** `SMOOTHING_RADIUS_S` — $r$ of (18) in seconds; `BORDER` — black borders or a zoom; `ZOOM`; `RUN_RADIUS_SWEEP` — the measures for $r$ = 0.1, 0.25 and 0.5 s (about 20 s); `SAVE_VIDEO` — the original and the stabilised video side by side in `outputs/video`.

**Expected result.** The trajectories $x$, $y$, $\alpha$ before and after smoothing, frames of the original and of the stabilised video, the table of the measures and its file `outputs/reports/lesson34_stabilisation.csv`, the video file and the record in `project.json`.

**What to observe.** At 63 fps and $r$ = 0.25 s the high-frequency jitter falls by 43–53 %: in fragment B from 33 to 19 px in shift and from 2.8° to 1.6° in rotation, in fragment A from 33 to 16 px and from 2.7° to 1.3°. The gyroscope confirms the result independently: the jitter of the rotation falls from 3.1° to 2.0° (B) and from 2.5° to 1.4° (A). About 87–89 % of the pixels remain valid. ITF grows only by about 0.3 dB, because the forward motion and the parallax of the near gates remain. A larger $r$ cuts off a larger part of the frame (74 % of valid pixels at $r$ = 0.5 s) and delays the intended turns of the pilot. In fragment B at $r$ = 0.5 s the motion measured again on the stabilised frames even grows (48 px), because more pairs fail at the wide black borders (23 of 239), while the gyroscope shows a further small decrease (1.9°): a measure computed from the processed frames themselves needs an independent check. At 32 and 16 fps more pairs fail and the trajectory becomes less reliable. An operator of an FPV drone needs a moderate smoothing without delay; a recorded video for analysis allows a stronger one.

**Control question 13.** Why is the trajectory (18) smoothed instead of compensating every inter-frame motion completely? What limits the radius of the smoothing for a pilot who controls the drone by the video?
""")

code(r"""
#@title 3.4.5 Stabilised video and comparative analysis (code shown) { display-mode: "form" }
SMOOTHING_RADIUS_S = 0.25 #@param {type:"slider", min:0.05, max:1.0, step:0.05}
BORDER = "black borders" #@param ["black borders", "crop by zoom"]
ZOOM = 1.15 #@param {type:"slider", min:1.0, max:1.5, step:0.05}
MOTION_MODEL = "similarity (shift and rotation)" #@param ["similarity (shift and rotation)", "translation (shift only)"]
RUN_RADIUS_SWEEP = True #@param {type:"boolean"}
SAVE_VIDEO = True #@param {type:"boolean"}

greys = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in FPV["frames"]]
fps = FPV["fps"]
orb_s = cv2.ORB_create(1000)


def interframe_motion(frames, masks=None):
    # keypoints only inside the valid part of a frame: the edges of black borders must not be matched
    er = np.ones((15, 15), np.uint8)
    kd = [orb_s.detectAndCompute(g, None if masks is None else cv2.erode(masks[k].astype(np.uint8) * 255, er)) for k, g in enumerate(frames)]
    T, ok = [np.zeros(3)], [True]
    for k in range(1, len(frames)):
        (k1, d1), (k2, d2) = kd[k - 1], kd[k]
        m_ = match(d1, d2)
        A_ = None
        if len(m_) >= 10:
            q1 = np.float32([k1[m.queryIdx].pt for m in m_])
            q2 = np.float32([k2[m.trainIdx].pt for m in m_])
            A_, in_ = cv2.estimateAffinePartial2D(q1, q2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if A_ is not None and in_.sum() < 15:
                A_ = None
        if A_ is None:
            T.append(T[-1].copy())
            ok.append(False)
        elif MOTION_MODEL.startswith("translation"):
            dxy = np.median((q2 - q1)[in_.ravel() > 0], axis=0)
            T.append(np.array([dxy[0], dxy[1], 0.0]))
            ok.append(True)
        else:
            T.append(np.array([A_[0, 2], A_[1, 2], math.atan2(A_[1, 0], A_[0, 0])]))
            ok.append(True)
    return np.array(T), np.array(ok)


def moving_average(x, r):
    if r < 1:
        return x.copy()
    xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    k = np.ones(2 * r + 1) / (2 * r + 1)
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def stabilise(frames, T, r, zoom=1.0):
    traj = np.cumsum(T, 0)
    smooth = moving_average(traj, r)                                    # formula (18)
    corr = smooth - traj
    h, w = frames[0].shape[:2]
    Z = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoom)
    out, valid = [], []
    for f, (dx, dy, da) in zip(frames, corr):
        M = np.array([[math.cos(da), -math.sin(da), dx], [math.sin(da), math.cos(da), dy], [0, 0, 1]])
        M = (np.vstack([Z, [0, 0, 1]]) @ M)[:2]
        out.append(cv2.warpAffine(f, M, (w, h)))
        valid.append(cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h)) > 0)
    return out, valid, traj, smooth


def jitter(T, ok, r):
    # high-frequency part of the trajectory; pairs without an estimate count as zero motion (no accumulation of errors)
    T = np.where(ok[:, None], T, 0.0)
    traj = np.cumsum(T, 0)
    e = (traj - moving_average(traj, r)).std(0)
    return float(np.hypot(e[0], e[1])), float(np.degrees(e[2]))


# independent reference of the rotation of the content: alpha_k = -(roll of the camera), integrated gyroscope
roll_steps = [aitbc.gyro_rotation(FPV["imu_t"], FPV["w_cam"], a, b)[2] for a, b in zip(FPV["times"][:-1], FPV["times"][1:])]
alpha_gyro = -np.r_[0.0, np.cumsum(roll_steps)]


def hf_std(x, r):
    return float(np.degrees((x - moving_average(x[:, None], r)[:, 0]).std()))


def compare(r_s, zoom):
    r = max(1, int(round(r_s * fps)))
    out, valid, traj, smooth = stabilise(greys, T_raw, r, zoom)
    T_st, ok_st = interframe_motion(out, valid)
    r_j = max(2, int(round(0.1 * fps)))
    j0, j1 = jitter(T_raw, ok_raw, r_j), jitter(T_st, ok_st, r_j)
    corr_alpha = (smooth - traj)[:, 2]
    return out, valid, traj, smooth, {"radius r, s": r_s, "radius r, frames": r,
                                      "jitter of shift: original, px": j0[0], "jitter of shift: stabilised, px": j1[0],
                                      "jitter of rotation: original, °": j0[1], "jitter of rotation: stabilised, °": j1[1],
                                      "jitter of rotation by the gyroscope: original, °": hf_std(alpha_gyro, r_j),
                                      "jitter of rotation by the gyroscope: stabilised, °": hf_std(alpha_gyro + corr_alpha, r_j),
                                      "pairs without estimate: original / stabilised": f"{int((~ok_raw).sum())} / {int((~ok_st).sum())}",
                                      "ITF original, dB": aitbc.interframe_psnr(greys), "ITF stabilised, dB": aitbc.interframe_psnr(out, valid),
                                      "share of valid pixels": float(np.mean([v.mean() for v in valid]))}


t0 = time.time()
T_raw, ok_raw = interframe_motion(greys)
zoom = ZOOM if BORDER == "crop by zoom" else 1.0
stab, valid, traj, smooth, res = compare(SMOOTHING_RADIUS_S, zoom)
print(f"Motion model: {MOTION_MODEL}; pairs without a reliable estimate (motion of the previous pair kept): {int((~ok_raw).sum())} of {len(greys) - 1}")

fig, ax = plt.subplots(1, 3, figsize=(19, 3.8))
tt = FPV["times"]
for j, (lab, sc) in enumerate((("x, px", 1), ("y, px", 1), ("α, °", 180 / math.pi))):
    ax[j].plot(tt, traj[:, j] * sc, lw=0.8, label="trajectory T_k")
    ax[j].plot(tt, smooth[:, j] * sc, lw=1.6, label=f"smoothed, r = {SMOOTHING_RADIUS_S:g} s")
    ax[j].set_xlabel("time of the flight, s")
    ax[j].set_ylabel(lab)
    ax[j].grid(alpha=0.3)
ax[0].legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"lesson34_trajectory_{FPV['segment'][0]}.png"), dpi=90)
plt.show()
idx = np.linspace(0, len(greys) - 1, 5).astype(int)
fig, ax = plt.subplots(2, 5, figsize=(19, 6))
for j, i in enumerate(idx):
    ax[0, j].imshow(greys[i], cmap="gray")
    ax[0, j].set_title(f"original, frame {i}", fontsize=8.5)
    ax[1, j].imshow(stab[i], cmap="gray")
    ax[1, j].set_title(f"stabilised, frame {i}", fontsize=8.5)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.show()

rows = [res]
if RUN_RADIUS_SWEEP:
    for r_s in (0.1, 0.25, 0.5):
        if abs(r_s - SMOOTHING_RADIUS_S) > 1e-9:
            rows.append(compare(r_s, zoom)[4])
STAB = pd.DataFrame(rows).set_index("radius r, s").sort_index()
display(STAB.style.format(precision=2).set_caption(f"Fragment {FPV['segment'][0]}, {fps:.0f} fps, {BORDER} ({time.time() - t0:.0f} s)"))
STAB.to_csv(os.path.join(REPORT_DIR, "lesson34_stabilisation.csv"))
if SAVE_VIDEO:
    side = [np.hstack([g_, s_]) for g_, s_ in zip(greys, stab)]
    out_path = os.path.join(VIDEO_DIR, f"fpv_stabilised_{FPV['segment'][0]}_{fps:.0f}fps.mp4")
    aitbc.write_video(out_path, side, fps)
    print(f"Saved: {out_path} (left — original, right — stabilised), {aitbc.video_info(out_path)['size_mb']:.1f} MB")
update_project_json("3.4", {"lesson_3_4": dict(fragment=FPV["segment"], frame_step=FPV["step"], fps=round(fps, 1),
                                               video=os.path.relpath(FPV["path"], PROJECT_ROOT), smoothing_radius_s=SMOOTHING_RADIUS_S,
                                               border=BORDER, motion_model=MOTION_MODEL,
                                               measures={k_: (round(float(v), 3) if not isinstance(v, str) else v) for k_, v in res.items()},
                                               stabilised_video=os.path.relpath(out_path, PROJECT_ROOT) if SAVE_VIDEO else None)})
""", form=True)

md(r"""
### Report of Lesson 3.4

Double-click this cell and write the answers.

1. Fragment, frame step, rotation between frames by the gyroscope; answer to control question 9:
2. Parameters of ORB, number and coverage of the keypoints; answer to control question 10:
3. Matches, shift, rotation compared with the gyroscope, table for four frame rates; answer to control question 11:
4. Comparison of the motion models, rotation from the homography and from the gyroscope; answer to control question 12:
5. Radius of the smoothing, jitter, ITF and valid pixels before and after the stabilisation, saved video; answer to control question 13:

---
## Self-check of Topic 3

1. A thermal frame of a UAV at night contains persons, parked cars and a warm road. Which steps of the classical processing are needed to obtain the boxes of the persons, and which polarity is needed for the cars?
2. Why do the Canny edges agree only partly with the boundaries of the annotation, and why can no thresholds be chosen for all frames?
3. Which geometric characteristics of contours are computed in metres, and which model of the GSD is used for an oblique frame?
4. An FPV video is transmitted at 30 frames per second instead of 120. How does this change the matching of keypoints and the stabilisation?
5. When is a homography an exact model of the motion between two frames, and why does a stabilised FPV video still contain residual motion?

## Data sources and licences

The course library reads the files directly from the repositories of the authors; the data are not redistributed with the notebook. When images or results are published, the data sets are cited.

1. **HIT-UAV.** Suo J., Wang T., Zhang X., Chen H., Zhou W., Shi W. HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection. *Scientific Data*, 10, 227 (2023). https://doi.org/10.1038/s41597-023-02066-6 · Data: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset · Licence CC BY 4.0.
2. **Kust4K.** Ouyang J., Wang Q., Shang Y., Jin P., Zhong H., Zhou L., Shen T. An RGB-TIR dataset from UAV platform for robust urban traffic scenes semantic segmentation. *Scientific Data*, 12, 1701 (2025). https://doi.org/10.1038/s41597-025-05994-7 · Data: https://doi.org/10.6084/m9.figshare.29476610 · Licence CC BY 4.0.
3. **M3OT.** Nie Z., Xue L., Fang Z., Ren J., Wei Y., Zheng J. M3OT: A Multi-Drone Multi-Modality dataset for Multi-Object Tracking. *Scientific Data*, 12, 1927 (2025). https://doi.org/10.1038/s41597-025-06204-0 · Data: https://doi.org/10.6084/m9.figshare.28308887 · Licence CC BY 4.0.
4. **TII-RATM.** Bosello M., Aguiari D., Keuter Y., Pallotta E., Kiade S., Caminati G., Pinzarrone F., Halepota J., Panerati J., Pau G. Race Against the Machine: A Fully-Annotated, Open-Design Dataset of Autonomous and Piloted High-Speed Flight. *IEEE Robotics and Automation Letters*, 9(4), 3799–3806 (2024). https://doi.org/10.1109/LRA.2024.3371288 · Data: https://github.com/tii-racing/drone-racing-dataset (release v3.0.0) · Licence CC BY 4.0.

**Next notebook.** Notebook 06 — Topic 4, Lessons 4.1 (lecture) and 4.2 (group class): machine learning in the recognition of military objects, HOG features, SVM and Random Forest, metrics of classification; convolutional networks, MobileNet, transfer learning and export to ONNX.
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
    out = sys.argv[1] if len(sys.argv) > 1 else "AITBC_NB05_Edges_contours_and_FPV_stabilisation.ipynb"
    nb = build(out)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    print(f"written {out}: {len(nb.cells)} cells ({n_md} text, {len(nb.cells) - n_md} code)")
