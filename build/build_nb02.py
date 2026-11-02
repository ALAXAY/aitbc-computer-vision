# Builder of Notebook 02 of the course "AI Technologies (Basic Course)" (AIT-BC).
# Topic 1, Lessons 1.3 and 1.4 (practical). All exercises use real open data sets.
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
# AI Technologies (Basic Course) · Notebook 02
## Topic 1. Fundamentals of computer vision in Google Colab for UAV and UGV tasks

| Lesson | Type, duration | Title |
|---|---|---|
| **1.3** | Practical class, 2 h | Setting up Google Colab and first work with frames of electro-optical systems |
| **1.4** | Practical class, 2 h | Processing of video streams of UAVs and ground robotic platforms |

Course: *AI Technologies (Basic Course)*, AIT-BC · Content module 1 · Kruty Heroes Military Institute of Telecommunications and Information Technology, Kyiv, 2026.

**How to work with this notebook**

1. Run the cells from top to bottom: *Runtime → Run all*, or `Shift+Enter` cell by cell. After a restart of the runtime run the cells again from task 1.3.2.
2. Every task begins with a text cell: the purpose of the task, the parameters of the form, the expected result and the control question. Answers to the control questions are written in the text cell **Report** at the end of each lesson.
3. Parameters are set in the forms of the code cells. Cells marked **[auto]** run again after every change of a parameter.
4. Code cells marked **(code shown)** are part of the practical skills of the lesson: read them, change them and run them again.
5. The results of the lessons are saved in the course project on Google Drive (`data`, `frames`, `models`, `outputs`); Notebook 01 created the structure, task 1.3.5 checks it and creates it again if it is missing.

**Data.** Real open data sets with manual annotation (see "Data sources and licences" at the end):
- **HIT-UAV** — thermal images and 5 thermal videos of a DJI Matrice M210 V2 UAV (60–130 m, day and night);
- **Kust4K** — pixel-aligned visible and thermal image pairs of a UAV over city roads;
- **VisDrone** — visible images of camera drones over cities;
- **M3OT** — synchronised visible and thermal image sequences of two DJI Mavic 3T drones (100–120 m, 10 frames per second, day, dusk and night) with vehicle tracks;
- **GOOSE** — image sequences of the front camera of a ground vehicle in fields and forests.

**Contents**

- **Lesson 1.3 (practical)** — 1.3.1 own notebook and GPU · 1.3.2 Python and the computer vision libraries · 1.3.3 visible frame of a UAV · 1.3.4 thermal frame and comparison with the visible one · 1.3.5 project structure on Google Drive · report
- **Lesson 1.4 (practical)** — 1.4.1 Google Drive and a UAV video fragment · 1.4.2 decomposition of a video into frames with a given sampling rate · 1.4.3 grid of frames for visual analysis · 1.4.4 thermal video fragment by the same method · 1.4.5 saving of the frame sets · 1.4.6 additional task: video of a ground robotic platform · report
- **Final self-check**, data sources and licences, next notebook
""")

# =====================================================================================
md(r"""
---
# Lesson 1.3 (Practical class). Setting up Google Colab and first work with frames of electro-optical systems

**Learning questions**
1. Creation of an own notebook in the Google Colab environment and activation of the GPU mode.
2. Checking the version of Python and installation of the basic stack of computer vision libraries.
3. Loading a test frame from the electro-optical system of a UAV and its display in the notebook.
4. Loading a test thermal frame and its comparison with the image in the visible spectrum.
5. Forming the structure of the project in Google Drive: data, frames, models, outputs.

**Result of the lesson.** A working notebook with the GPU runtime, a table of the versions of the libraries, a visible and a thermal frame of a UAV with their properties, and the project structure on Google Drive with the first saved frames.
""")

md(r"""
## Task 1.3.1. Own notebook in Google Colab and the GPU mode

**Steps.**
1. Open https://colab.research.google.com and sign in with a Google account.
2. *File → New notebook in Drive*: the notebook is created in the folder *My Drive / Colab Notebooks*. Rename it: *File → Rename*, for example `AITBC_<surname>_lesson_1_3.ipynb`. This notebook (Notebook 02) is opened by *File → Upload notebook* or from Google Drive.
3. *Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save*. The runtime restarts.
4. Run the cell below. It reports whether the notebook runs in Colab and whether a GPU is available, and measures the speed of a matrix multiplication on the CPU and on the GPU.

**Parameters.** `MATRIX_SIZE` — size $n$ of the square matrices of the test; `REPEATS` — number of repetitions.

**Expected result.** In the GPU runtime the table shows the name of the GPU (for example *Tesla T4*) and a speed of several TFLOP/s against tens or hundreds of GFLOP/s of the CPU. A matrix product of size $n$ takes $2n^3$ floating-point operations, therefore the speed is

$$P = \frac{2 n^3}{t}, \tag{1}$$

FLOP/s, where $t$ is the time of one multiplication, s.

**Control question 1.** Why is the first run of an operation on the GPU slower than the following runs, and why does the code call `torch.cuda.synchronize()` before stopping the timer?
""")

code(r"""
#@title 1.3.1 Runtime and GPU check { display-mode: "form" }
MATRIX_SIZE = 2048 #@param {type:"slider", min:256, max:8192, step:256}
REPEATS = 5 #@param {type:"slider", min:1, max:20, step:1}

import sys, os, platform, time, json, math, shutil, subprocess, importlib
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
try:
    import torch
    CUDA_OK = torch.cuda.is_available()
except ImportError:
    torch, CUDA_OK = None, False

rows = [("Environment", "Google Colab" if IN_COLAB else "local Jupyter")]
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=15).stdout.strip()
except (FileNotFoundError, subprocess.TimeoutExpired):
    out = ""
rows.append(("GPU (nvidia-smi)", out or "not available: Runtime → Change runtime type → T4 GPU"))


def matmul_speed(device):
    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    (a @ b).sum().item()                                   # warm-up: memory allocation, kernel selection
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        c = a @ b
    if device == "cuda":
        torch.cuda.synchronize()                           # wait until the GPU has finished
    t = (time.perf_counter() - t0) / REPEATS
    return t, 2 * MATRIX_SIZE ** 3 / t                     # formula (1)


if torch is not None:
    t_cpu, p_cpu = matmul_speed("cpu")
    rows.append((f"CPU: {MATRIX_SIZE} x {MATRIX_SIZE} matrix product", f"{t_cpu * 1000:.1f} ms, {p_cpu / 1e9:.1f} GFLOP/s"))
    if CUDA_OK:
        t_gpu, p_gpu = matmul_speed("cuda")
        rows.append((f"GPU: {MATRIX_SIZE} x {MATRIX_SIZE} matrix product", f"{t_gpu * 1000:.2f} ms, {p_gpu / 1e12:.2f} TFLOP/s"))
        rows.append(("Acceleration t_CPU / t_GPU", f"{t_cpu / t_gpu:.0f}"))
display(pd.DataFrame(rows, columns=["Item", "Value"]).set_index("Item"))
""", form=True)

md(r"""
## Task 1.3.2. Python version and the basic stack of computer vision libraries

**Purpose.** To check the versions of Python and of the libraries, to install the missing ones and to install the course library `aitbc` with access to the data sets.

The libraries of the stack and their role:
- **NumPy** — arrays; an image is an array H × W × C;
- **OpenCV** (`cv2`) — reading and writing images and video, colour conversion, filters, geometry;
- **Matplotlib** — display of images and plots;
- **pandas** — tables (annotations, results);
- **PyTorch** — tensors on CPU and GPU, neural networks (Topics 4–5);
- **Ultralytics** — YOLO detectors (Topic 5); installed in Colab by `pip install ultralytics`.

**Parameters.**
- `INSTALL_ULTRALYTICS` — install Ultralytics now (about 1 min); it is needed only from Topic 5.
- `CACHE_ON_DRIVE` — keep the downloaded data in Google Drive (folder `MyDrive/AITBC/data_cache`): the next session does not download them again.

**Expected result.** A table with the versions and the status `OK` for every required library; the message of the course library about the connected data sets.

**Control question 2.** Why is the table of versions saved together with the results of every experiment (file `project.json`, task 1.3.5)?
""")

code(r"""
#@title 1.3.2 Python, library versions and the course library { display-mode: "form" }
INSTALL_ULTRALYTICS = False #@param {type:"boolean"}
CACHE_ON_DRIVE = False #@param {type:"boolean"}

REQUIRED = {"numpy": "1.24", "cv2": "4.8", "matplotlib": "3.7", "pandas": "2.0", "torch": "2.0"}


def version_tuple(v):
    return tuple(int("".join(ch for ch in x if ch.isdigit()) or 0) for x in v.split(".")[:2])


if INSTALL_ULTRALYTICS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)
rows = [("Python", platform.python_version(), "3.10", "OK" if sys.version_info >= (3, 10) else "update")]
for mod, need in list(REQUIRED.items()) + [("ultralytics", "8.0")]:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        status = "OK" if version_tuple(v) >= version_tuple(need) else "update"
    except ImportError:
        v, status = "not installed", "install" if mod != "ultralytics" else "optional (Topic 5)"
    rows.append(({"cv2": "opencv (cv2)"}.get(mod, mod), v, need, status))
VERSIONS = pd.DataFrame(rows, columns=["Library", "Installed", "Required at least", "Status"]).set_index("Library")
display(VERSIONS)

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
hit, kust, vis, m3ot, goose = aitbc.HITUAV(), aitbc.Kust4K(), aitbc.VisDrone(), aitbc.M3OT(), aitbc.GOOSE()
print(f"Course library aitbc {aitbc.__version__}: data sets HIT-UAV, Kust4K, VisDrone, M3OT, GOOSE connected "
      f"in {time.time() - t0:.0f} s; cache {aitbc.DATA_ROOT}")
""", form=True)

md(r"""
## Task 1.3.3. Visible frame of the electro-optical system of a UAV

**Purpose.** To load a real frame of a UAV camera, to display it in the notebook and to read its properties: size, data type, range and mean of the brightness, file size.

**Parameters.**
- `SOURCE` — real frame: VisDrone (camera drone over a city, oblique view), M3OT (DJI Mavic 3T, vertical view from 100–120 m), Kust4K (visible camera of a UAV over a road), or the own frame of the participant (upload, file on Google Drive or URL).
- `FRAME_NUMBER` — number of the frame of the selected source.
- `SHOW_ANNOTATION` — boxes of the annotated objects (VisDrone, M3OT).
- `OWN_PATH`, `OWN_URL` — location of the own frame.

**Code (shown).** The frame is read by `cv2.imread` and converted from the BGR order of OpenCV to RGB for Matplotlib.

**Expected result.** The frame, its histogram of brightness per channel and the table of properties. A visible frame of an 8-bit camera has the type `uint8`, the shape (H, W, 3) and values 0…255.

**Control question 3.** How many bytes does one uncompressed RGB frame of 640 × 512 px occupy, and how many megabytes per second does a stream of 30 such frames per second produce? Compare with the size of the PNG file in the table.
""")

code(r"""
#@title 1.3.3 Visible frame of a UAV (code shown) { run: "auto" }
SOURCE = "M3OT (DJI Mavic 3T, vertical view)" #@param ["VisDrone (oblique view over a city)", "M3OT (DJI Mavic 3T, vertical view)", "Kust4K (UAV over a road)", "own frame: upload", "own frame: file on Google Drive", "own frame: URL"]
FRAME_NUMBER = 1 #@param {type:"slider", min:1, max:8, step:1}
SHOW_ANNOTATION = True #@param {type:"boolean"}
OWN_PATH = "/content/drive/MyDrive/AITBC/data/images/frame.jpg" #@param {type:"string"}
OWN_URL = "" #@param {type:"string"}

M3OT_VIEWS = [("1-04", 150), ("1-09", 300), ("1-01", 700), ("1-07", 200), ("1-05", 150), ("1-03", 240), ("1-06", 300), ("1-08", 300)]
objs, path = [], None
if SOURCE.startswith("VisDrone"):
    name = vis.CURATED[FRAME_NUMBER - 1]
    path = vis.z.get(f"VisDrone2019-DET-val/images/{name}.jpg")
    _, objs = vis.load(name)
    caption = f"VisDrone {name}"
elif SOURCE.startswith("M3OT"):
    seq, k = M3OT_VIEWS[FRAME_NUMBER - 1]
    path = m3ot.prefetch(seq, "rgb", start=k, count=1)[0]
    objs = m3ot.objects(m3ot.tracks(seq, "rgb"), k + 1)
    caption = f"M3OT sequence {seq} ({m3ot.SEQUENCES[seq]}), frame {k + 1}"
elif SOURCE.startswith("Kust4K"):
    fid = kust.DAY[FRAME_NUMBER - 1]
    kust.prefetch([fid])
    path = kust._zip("rgb").get(fid + ".png")
    caption = f"Kust4K {fid}"
elif SOURCE.endswith("upload"):
    if not IN_COLAB:
        raise RuntimeError("upload works only in Google Colab")
    from google.colab import files
    up = files.upload()
    path = os.path.join(WORK_DIR, next(iter(up)))
    caption = "own frame"
elif SOURCE.endswith("Google Drive"):
    if IN_COLAB and OWN_PATH.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    path, caption = OWN_PATH, "own frame"
else:
    path = aitbc.fetch(OWN_URL, os.path.join(WORK_DIR, "own_frame" + os.path.splitext(OWN_URL.split("?")[0])[1]))
    caption = "own frame"

bgr = cv2.imread(path, cv2.IMREAD_COLOR)            # OpenCV reads colour images in the order B, G, R
if bgr is None:
    raise IOError(f"OpenCV cannot read {path}")
frame_eo = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)     # Matplotlib expects R, G, B
FRAME_EO, FRAME_EO_NAME = frame_eo, caption
shown = aitbc.draw_boxes(frame_eo, objs, {"vehicle": (255, 235, 59)} if objs and objs[0]["cls"] == "vehicle" else vis.COLORS,
                         labels=False) if (SHOW_ANNOTATION and objs) else frame_eo
fig, ax = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax[0].imshow(shown)
ax[0].set_title(caption + (f" — {len(objs)} annotated objects" if objs else ""), fontsize=10)
ax[0].axis("off")
for c, col in enumerate(("red", "green", "blue")):
    ax[1].hist(frame_eo[..., c].ravel(), bins=64, range=(0, 255), color=col, alpha=0.45, label=col)
ax[1].set_xlabel("brightness, grey levels")
ax[1].set_ylabel("number of pixels")
ax[1].legend()
ax[1].set_title("Histogram of brightness per channel")
plt.tight_layout()
plt.show()
h, w = frame_eo.shape[:2]
display(pd.DataFrame({"Value": [str(frame_eo.shape), str(frame_eo.dtype), f"{frame_eo.min()} … {frame_eo.max()}",
                                ", ".join(f"{v:.1f}" for v in frame_eo.reshape(-1, 3).mean(0)),
                                f"{frame_eo.nbytes / 1024:.0f}", f"{os.path.getsize(path) / 1024:.0f}",
                                f"{frame_eo.nbytes / os.path.getsize(path):.1f}"]},
                     index=["shape (H, W, C)", "data type", "range of values", "mean of R, G, B",
                            "size in memory, KB", "size of the file, KB", "compression ratio of the file"]))
""")

md(r"""
## Task 1.3.4. Thermal frame and its comparison with the visible frame

**Purpose.** To load a thermal frame of the same scene as the visible one, to establish the palette of the thermal channel and to compare the two channels: what each channel shows, their brightness statistics, and how well their structures coincide.

**Parameters.**
- `PAIR` — source of a synchronised pair: M3OT (visible and thermal camera of one DJI Mavic 3T; day, dusk, night) or Kust4K (registered pair of a UAV over a road; night and day pairs alternate).
- `FRAME_NUMBER` — pair of the selected source.
- `VIEW` — side by side, blend of the two channels with a slider, or edges of both channels in one image (red — visible, green — thermal; yellow — coinciding edges).
- `BLEND` — share of the thermal channel in the blend.
- `THERMAL_VIEW` — the thermal frame as recorded in the data set, or converted to the convention "warm = bright" (white-hot), which is used in the following lessons for the detection of warm objects.
- `CHECK_ALL_PAIRS` — the palette check for all eight pairs of the selected source.

**Palette of a thermal frame.** A thermal camera converts the temperature of the scene into brightness through a palette. The two grey palettes are white-hot (warm = bright; HIT-UAV) and black-hot (warm = dark); pseudo-colour palettes map the temperature to colours (Kust4K: from purple, cold, to yellow, warm). When the description of a data set does not state the palette, it is established from the scene. In a road scene the asphalt is, as a rule, warmer than vegetation: by day it absorbs most of the solar radiation, at night it gives off the heat stored during the day, while vegetation is cooled by the evaporation of water from the leaves. The road surface is taken around the annotated vehicles (M3OT) or from the segmentation labels (Kust4K). The vegetation is taken by the green colour of the visible frame, with the excess-green index $ExG = (2G - R - B)/(R + G + B) > 0.12$ (M3OT), or from the labels (Kust4K). A road darker than the vegetation by more than 15 grey levels indicates the black-hot palette, a brighter road — the white-hot convention.

**Expected result.** Panels of the two channels, a table of their properties and the result of the palette check. For every channel the table shows the mean, the standard deviation (contrast) and the **entropy** of the brightness histogram

$$H = -\sum_{k=0}^{255} p_k \log_2 p_k, \tag{2}$$

bit, where $p_k$ is the share of pixels with the brightness $k$. The entropy shows how many brightness levels are effectively used: 8 bit — all 256 levels equally, low values — a dark or flat image. The **sharpness** is the variance of the Laplacian (Lesson 1.1).

**What to observe.** At night the visible frame of M3OT (sequence 1-02) is dark and has a low entropy, while the thermal frame keeps the full range. In the thermal frames of M3OT the road surface is dark, while vegetation, fields and vehicles are bright: the thermal channel of M3OT is recorded in the black-hot palette, therefore the dark asphalt is warmer than the vegetation, and the bright vehicles are cooler than the road. Kust4K shows the opposite order of brightness for the same order of temperatures. The rule "asphalt is warmer than vegetation" has exceptions: in the day pair 00355D of Kust4K (pair 8; low sun, long shadows) the sunlit tree crowns are warmer than the road, therefore the palette is established on several frames (`CHECK_ALL_PAIRS`). In the edge view the edges of roads and buildings coincide only approximately: the two cameras have different lenses, and the residual shift between the channels is measured in task 1.4.4.

**Control question 4.** Name two objects that are visible in the thermal frame but not in the visible frame, and two details that are visible only in the visible frame. Which palette does the thermal channel of M3OT use, and which observation in the frame shows it?
""")

code(r"""
#@title 1.3.4 Thermal frame and comparison with the visible frame [auto] { run: "auto", display-mode: "form" }
PAIR = "M3OT (DJI Mavic 3T: day, dusk, night)" #@param ["M3OT (DJI Mavic 3T: day, dusk, night)", "Kust4K (UAV over a road: day, night)"]
FRAME_NUMBER = 1 #@param {type:"slider", min:1, max:8, step:1}
VIEW = "side by side" #@param ["side by side", "blend", "edges of both channels"]
BLEND = 0.5 #@param {type:"slider", min:0, max:1, step:0.05}
THERMAL_VIEW = "as recorded" #@param ["as recorded", "warm = bright (white-hot)"]
CHECK_ALL_PAIRS = False #@param {type:"boolean"}

M3OT_PAIRS = [("1-02", 300), ("1-04", 150), ("1-06", 300), ("1-08", 300), ("1-09", 300), ("1-01", 700), ("1-03", 240), ("1-05", 150)]
KUST_PAIRS = ("02566N", "00031D", "02807N", "00580D", "03135N", "00990D", "03895N", "00355D")


def load_pair(number):
    # visible frame, thermal frame, palette, road and vegetation masks and caption of pair `number`
    if PAIR.startswith("M3OT"):
        seq, k = M3OT_PAIRS[number - 1]
        eo_ = m3ot.frames(seq, "rgb", start=k, count=1)[0]
        ir_ = m3ot.frames(seq, "ir", start=k, count=1)[0]
        boxes = [o["box"] for o in m3ot.objects(m3ot.tracks(seq, "ir"), k + 1)]
        road_ = aitbc.surroundings_mask(ir_.shape, boxes)                  # road surface around the vehicles
        veg_ = aitbc.vegetation_mask(eo_)                                    # green colour of the visible frame
        return eo_, ir_, m3ot.PALETTE, road_, veg_, f"M3OT {seq} ({m3ot.SEQUENCES[seq]}), frame {k + 1}"
    p = kust.pair(KUST_PAIRS[number - 1])
    return (p["rgb"], p["tir"], kust.PALETTE, p["labels"] == 1, p["labels"] == 6,
            f"Kust4K {p['id']} ({'night' if p['night'] else 'day'})")


def entropy(img):
    hist = np.bincount(aitbc.gray(img).ravel(), minlength=256).astype(float)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())                                   # formula (2)


eo, ir, palette, road, veg, caption = load_pair(FRAME_NUMBER)
FRAME_IR = ir                                                               # saved as recorded in task 1.3.5
WARM_BRIGHT = THERMAL_VIEW.startswith("warm")
ir_show = aitbc.to_white_hot(ir, palette) if WARM_BRIGHT else ir
ir_name = "thermal, warm = bright" if WARM_BRIGHT else "thermal, as recorded"

if VIEW == "side by side":
    panels = [(eo, "visible channel"), (ir_show, f"{ir_name}; palette of the data set: {palette}")]
elif VIEW == "blend":
    panels = [(((1 - BLEND) * eo.astype(np.float32) + BLEND * ir_show.astype(np.float32)).astype(np.uint8),
               f"blend: {1 - BLEND:.2f}·visible + {BLEND:.2f}·thermal")]
else:
    e1 = cv2.Canny(cv2.GaussianBlur(aitbc.gray(eo), (5, 5), 0), 60, 140)
    e2 = cv2.Canny(cv2.GaussianBlur(aitbc.gray(ir_show), (5, 5), 0), 60, 140)
    edges = np.zeros_like(eo)
    edges[..., 0], edges[..., 1] = e1, e2
    panels = [(edges, "edges: red — visible, green — thermal, yellow — both")]
fig, ax = plt.subplots(1, len(panels), figsize=(8 * len(panels), 5.6), squeeze=False)
for a, (im_, ti) in zip(ax[0], panels):
    a.imshow(im_)
    a.set_title(ti, fontsize=10)
    a.axis("off")
fig.suptitle(caption, fontsize=11)
plt.tight_layout()
plt.show()

rows = []
for name, im_ in (("visible", eo), (ir_name, ir_show)):
    g = aitbc.gray(im_)
    rows.append(dict(channel=name, mean=round(float(g.mean()), 1), std=round(float(g.std()), 1),
                     entropy_bit=round(entropy(im_), 2), sharpness=round(aitbc.sharpness(im_)), shape=str(im_.shape)))
display(pd.DataFrame(rows).set_index("channel"))

chk = aitbc.palette_check(ir, road, veg)                                    # brightness as recorded
pd.set_option("display.max_colwidth", 100)                                  # show the conclusions in full
display(pd.DataFrame({"Value": [palette,
                                f"{chk['road_median']:.0f} ({chk['road_pixels']} px)",
                                f"{chk['vegetation_median']:.0f} ({chk['vegetation_pixels']} px)",
                                chk["verdict"]]},
                     index=["palette in the course library", "road surface: median brightness as recorded",
                            "vegetation: median brightness as recorded", "palette established from the scene"]))

if CHECK_ALL_PAIRS:
    rows = []
    for n in range(1, 9):
        e_, i_, pal_, r_, v_, cap_ = load_pair(n)
        c_ = aitbc.palette_check(i_, r_, v_)
        rows.append(dict(pair=n, frame=cap_, road=c_["road_median"], vegetation=c_["vegetation_median"],
                         conclusion=c_["verdict"]))
    tab = pd.DataFrame(rows).set_index("pair")
    display(tab)
    print("Conclusions over the eight pairs:")
    print(tab.conclusion.value_counts().to_string())
""", form=True)

md(r"""
## Task 1.3.5. Structure of the project on Google Drive

**Purpose.** To connect Google Drive, to check (and, if necessary, create) the course project `data / frames / models / outputs` and to save in it the frames of tasks 1.3.3 and 1.3.4 together with the versions of the libraries.

**Parameters.**
- `USE_DRIVE` — keep the project on Google Drive (Colab asks for access to the Google account); otherwise it is created on the session disk and lost at the end of the session.
- `PROJECT_PATH` — folder of the project on Google Drive (the same as in Notebook 01).
- `SAVE_FRAMES` — save the visible frame (`data/images`) and the thermal frame (`data/thermal`) of tasks 1.3.3–1.3.4.

**Expected result.** The tree of the project with the new files and the updated `project.json` (versions of the libraries of task 1.3.2, date, list of the lessons done).

**Control question 5.** Why are the frames saved in PNG and not in JPEG? When is JPEG acceptable?
""")

code(r"""
#@title 1.3.5 Project structure on Google Drive and the first saved frames { display-mode: "form" }
USE_DRIVE = True #@param {type:"boolean"}
PROJECT_PATH = "AITBC/AITBC_project" #@param {type:"string"}
SAVE_FRAMES = True #@param {type:"boolean"}

if USE_DRIVE and IN_COLAB:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    PROJECT_ROOT = os.path.join("/content/drive/MyDrive", PROJECT_PATH)
else:
    PROJECT_ROOT = os.path.join(WORK_DIR, PROJECT_PATH)
    if USE_DRIVE:
        print("Google Drive is available only in Google Colab; the project is created in", PROJECT_ROOT)
SUBDIRS = ["data/images", "data/video", "data/thermal", "data/annotations", "frames/visible", "frames/thermal",
           "models", "outputs/figures", "outputs/video", "outputs/reports"]
created = [d for d in SUBDIRS if not os.path.isdir(os.path.join(PROJECT_ROOT, d))]
for d in SUBDIRS:
    os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)
print(f"Project: {PROJECT_ROOT} ({'created now: ' + ', '.join(created) if created else 'structure is complete'})")


def update_project_json(lesson, extra=None):
    path = os.path.join(PROJECT_ROOT, "project.json")
    meta = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"project": os.path.basename(PROJECT_ROOT)}
    meta.setdefault("lessons", [])
    if lesson not in meta["lessons"]:
        meta["lessons"].append(lesson)
    meta["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["versions"] = {k: v for k, v in VERSIONS["Installed"].items()} if "VERSIONS" in globals() else {}
    meta["aitbc"] = aitbc.__version__
    meta.update(extra or {})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return meta


saved = []
if SAVE_FRAMES and "FRAME_EO" in globals():
    stamp = time.strftime("%Y%m%d")
    p1 = os.path.join(PROJECT_ROOT, "data", "images", f"lesson13_eo_{stamp}.png")
    cv2.imwrite(p1, cv2.cvtColor(FRAME_EO, cv2.COLOR_RGB2BGR))
    saved.append(p1)
    if "FRAME_IR" in globals():
        p2 = os.path.join(PROJECT_ROOT, "data", "thermal", f"lesson13_ir_{stamp}.png")
        cv2.imwrite(p2, cv2.cvtColor(FRAME_IR, cv2.COLOR_RGB2BGR))
        saved.append(p2)
meta = update_project_json("1.3", {"lesson_1_3_frames": [os.path.relpath(p, PROJECT_ROOT) for p in saved],
                                   "lesson_1_3_source": globals().get("FRAME_EO_NAME", "")})


def print_tree(path, prefix="", depth=0, max_files=6):
    entries = sorted(os.listdir(path), key=lambda e: (not os.path.isdir(os.path.join(path, e)), e))
    files = [e for e in entries if not os.path.isdir(os.path.join(path, e))]
    shown = [e for e in entries if os.path.isdir(os.path.join(path, e))] + files[:max_files]
    for i, e in enumerate(shown):
        last = i == len(shown) - 1 and len(files) <= max_files
        full = os.path.join(path, e)
        size = "" if os.path.isdir(full) else f"  ({os.path.getsize(full) / 1024:.0f} KB)"
        print(prefix + ("└── " if last else "├── ") + e + ("/" if os.path.isdir(full) else "") + size)
        if os.path.isdir(full):
            print_tree(full, prefix + ("    " if last else "│   "), depth + 1, max_files)
    if len(files) > max_files:
        print(prefix + f"└── … {len(files) - max_files} more files")


print(PROJECT_ROOT + "/")
print_tree(PROJECT_ROOT)
print("project.json: lessons", meta["lessons"], "| updated", meta["updated"])
""", form=True)

md(r"""
### Report of Lesson 1.3

Double-click this cell and write the answers.

1. Runtime of the notebook (CPU or GPU, name of the GPU) and the measured speeds of task 1.3.1:
2. Answer to control question 1:
3. Versions of Python, OpenCV and PyTorch:
4. Answer to control question 2:
5. Source, size and data type of the visible frame; answer to control question 3:
6. Entropy and contrast of the visible and thermal frames by day and at night; palette of the thermal channel of M3OT and of Kust4K established from the scene; answer to control question 4:
7. Path of the project on Google Drive; answer to control question 5:
""")

# =====================================================================================
md(r"""
---
# Lesson 1.4 (Practical class). Processing of video streams of UAVs and ground robotic platforms

**Learning questions**
1. Connection of Google Drive to Colab and loading of a video fragment from the electro-optical system of a UAV.
2. Decomposition of a video stream into separate frames with a given sampling rate.
3. Display of the sequence of frames as a grid for preliminary visual analysis.
4. Processing of a thermal video fragment by the same method.
5. Saving of the formed sets of frames in the corresponding folders of the project.

**Result of the lesson.** Visible and thermal video fragments of a UAV in `data/video`, the frame sets extracted from them with the chosen sampling rate in `frames/visible` and `frames/thermal` with an index table (`frames.csv`), grids of frames in `outputs/figures`, and the measured offset between the visible and the thermal channel.

**Video data of the lesson.**
- **M3OT** — the visible and the thermal camera of a DJI Mavic 3T record the same scene at 10 frames per second. The data set stores the sequences as numbered frames; task 1.4.1 assembles them into a video file, as the stream of a UAV is recorded on board, and the following tasks work with the video file.
- **HIT-UAV** — five thermal videos of a DJI Matrice M210 V2 with a Zenmuse XT2 camera (7.5 frames per second, H.264), recorded as they are.
- **Own video** of the participant from Google Drive (any format that OpenCV reads: MP4, AVI, MOV).
""")

md(r"""
## Task 1.4.1. Google Drive and a video fragment of a UAV

**Purpose.** To obtain a video fragment of a UAV, to save it in `data/video` of the project and to read its properties with `cv2.VideoCapture`.

**Parameters.**
- `SOURCE` — M3OT visible video (assembled from the frames of a sequence), HIT-UAV thermal video, or own video from Google Drive (`OWN_VIDEO`).
- `SEQUENCE`, `START_FRAME`, `N_FRAMES` — sequence of M3OT and the fragment of it (10 frames = 1 s). Every frame is a PNG file of about 0.6 MB: at the first run 200 visible frames (task 1.4.1) and 200 thermal frames (task 1.4.4) mean about 230 MB of download; later runs read the frames from the cache (`CACHE_ON_DRIVE`, task 1.3.2, keeps it between sessions).
- `CODEC` — codec of the written file: `mp4v` (MPEG-4 Part 2, file .mp4) or `MJPG` (Motion JPEG, file .avi, every frame is a JPEG image: larger files, simple random access).

**Code (shown).** The video is written with `cv2.VideoWriter` and read back with `cv2.VideoCapture`; the properties are read with `cap.get(cv2.CAP_PROP_...)`.

**Expected result.** A table of the properties of the video file (frame rate, number of frames, duration, size, codec, data rate) and three frames of it. The data rate of a video file is

$$R = \frac{8 S}{T}, \tag{3}$$

bit/s, where $S$ is the size of the file in bytes and $T$ the duration in seconds. It is compared with the rate of the uncompressed stream $R_0 = W \cdot H \cdot 3 \cdot 8 \cdot f$ for frames W × H px at f frames per second; the ratio $R_0 / R$ is the compression ratio of the codec.

**Control question 6.** Why does a video file take tens of times less space than the same number of PNG frames? What is lost during this compression?
""")

code(r"""
#@title 1.4.1 Video fragment of a UAV in data/video (code shown) { display-mode: "form" }
SOURCE = "M3OT visible video (DJI Mavic 3T)" #@param ["M3OT visible video (DJI Mavic 3T)", "HIT-UAV thermal video (Zenmuse XT2)", "own video from Google Drive"]
SEQUENCE = "1-04" #@param ["1-01", "1-02", "1-03", "1-04", "1-05", "1-06", "1-07", "1-08", "1-09"]
START_FRAME = 0 #@param {type:"integer"}
N_FRAMES = 200 #@param {type:"slider", min:20, max:600, step:10}
HIT_VIDEO = "60m-30_1.mov" #@param ["60m-30_1.mov", "60m-40_3.mov", "70m-90_1.mov", "120m-30_3.mov", "130m-60_3.mov"]
OWN_VIDEO = "/content/drive/MyDrive/AITBC/data/video/my_video.mp4" #@param {type:"string"}
CODEC = "mp4v" #@param ["mp4v", "MJPG"]

video_dir = os.path.join(PROJECT_ROOT, "data", "video")
ext = ".mp4" if CODEC == "mp4v" else ".avi"
if SOURCE.startswith("M3OT"):
    n_total = m3ot.n_frames(SEQUENCE, "rgb")
    start = int(np.clip(START_FRAME, 0, n_total - 2))
    count = int(min(N_FRAMES, n_total - start))
    frames = m3ot.frames(SEQUENCE, "rgb", start=start, count=count)
    VIDEO_EO = os.path.join(video_dir, f"m3ot_{SEQUENCE}_eo_{start:04d}_{count}{ext}")
    aitbc.write_video(VIDEO_EO, frames, fps=m3ot.CAMERA["fps"], codec=CODEC)     # cv2.VideoWriter inside
    png_size = sum(os.path.getsize(p) for p in m3ot.prefetch(SEQUENCE, "rgb", start=start, count=count))
    VIDEO_EO_SOURCE = dict(dataset="M3OT", sequence=SEQUENCE, channel="visible", start_frame=start + 1, frames=count)
    del frames
elif SOURCE.startswith("HIT-UAV"):
    src = hit.video(HIT_VIDEO)
    VIDEO_EO = os.path.join(video_dir, "hituav_" + HIT_VIDEO)
    shutil.copy2(src, VIDEO_EO)
    png_size = None
    VIDEO_EO_SOURCE = dict(dataset="HIT-UAV", video=HIT_VIDEO, channel="thermal")
else:
    if IN_COLAB and OWN_VIDEO.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    if not os.path.isfile(OWN_VIDEO):
        raise FileNotFoundError(f"{OWN_VIDEO} not found: copy the video to Google Drive and give its path in OWN_VIDEO")
    VIDEO_EO, png_size = OWN_VIDEO, None
    VIDEO_EO_SOURCE = dict(dataset="own video", path=OWN_VIDEO)

cap = cv2.VideoCapture(VIDEO_EO)
if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) <= 0:
    raise IOError(f"OpenCV cannot read the video {VIDEO_EO}: convert it to MP4 (H.264) or AVI (MJPG)")
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1", "replace")
samples = []
for k in (0, n // 2, max(n - 1, 0)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, f = cap.read()
    if ok:
        samples.append((k, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
cap.release()
S = os.path.getsize(VIDEO_EO)
T = n / fps if fps else float("nan")
R = 8 * S / T                                                           # formula (3)
R0 = w * h * 3 * 8 * fps
rows = [("file", os.path.relpath(VIDEO_EO, PROJECT_ROOT) if VIDEO_EO.startswith(PROJECT_ROOT) else VIDEO_EO),
        ("codec (FourCC)", codec), ("frame size W x H, px", f"{w} x {h}"), ("frame rate f, frames/s", f"{fps:.2f}"),
        ("number of frames", n), ("duration T, s", f"{T:.1f}"), ("file size S, MB", f"{S / 1e6:.2f}"),
        ("data rate R, Mbit/s (3)", f"{R / 1e6:.2f}"), ("uncompressed rate R0, Mbit/s", f"{R0 / 1e6:.1f}"),
        ("compression ratio R0 / R", f"{R0 / R:.0f}")]
if png_size:
    rows.append(("same frames as PNG files, MB", f"{png_size / 1e6:.1f}"))
display(pd.DataFrame(rows, columns=["Property", "Value"]).set_index("Property"))
fig, ax = plt.subplots(1, len(samples), figsize=(16, 4.4))
for a, (k, f) in zip(np.atleast_1d(ax), samples):
    a.imshow(f)
    a.set_title(f"frame {k} (t = {k / fps:.1f} s)", fontsize=10)
    a.axis("off")
plt.tight_layout()
plt.show()
""", form=True)

md(r"""
## Task 1.4.2. Decomposition of the video stream into frames with a given sampling rate

**Purpose.** To extract frames from the video file with a chosen sampling rate and to estimate how the choice of the rate changes the number of frames, their similarity and the motion between them.

With the frame rate of the video $f$ and the required sampling rate $f_s \le f$, every $N$-th frame is taken:

$$N = \operatorname{round}\left(\frac{f}{f_s}\right), \qquad f_s' = \frac{f}{N}, \qquad \Delta t = \frac{N}{f}, \tag{4}$$

where $f_s'$ is the actual sampling rate and $\Delta t$ is the time between the extracted frames. For a video of $n$ frames the set contains $\lceil n/N \rceil$ frames.

**Parameters.**
- `SAMPLING_RATE_HZ` — required sampling rate $f_s$, frames per second.
- `MAX_FRAMES` — maximum number of extracted frames.
- `RESIZE_WIDTH` — width of the extracted frames, px (0 = original size); a smaller width reduces the volume of the set.

**Code (shown).** The video is read frame by frame with a generator (`aitbc.iter_video`), therefore a long video does not fill the memory.

**Output.** The table of the extraction: $N$, $f_s'$, $\Delta t$ (4), number and volume of the frames; two plots for the extracted frames:
1. the mean absolute difference of successive frames (similarity: small values mean almost repeated frames);
2. the displacement of the image between successive extracted frames, estimated by phase correlation (motion of the camera).

**What to observe.** With a high sampling rate successive frames are almost identical: the set is large but carries little new information. With a low rate the displacement between frames grows; when it reaches a large part of the frame width, neighbouring frames no longer overlap, and stabilisation, mosaicking or tracking (Topics 3 and 5) become impossible. The sampling rate is chosen as a compromise between the volume of the set and the overlap of the frames. This is how the data sets of the course were made: Kust4K and HIT-UAV took every 15th frame of their videos.

**Control question 7.** A video of 30 frames per second is decomposed with $f_s$ = 4 Hz. What are $N$, $f_s'$ and $\Delta t$? How many frames does a 2-minute video give?
""")

code(r"""
#@title 1.4.2 Frames with a given sampling rate (code shown) { run: "auto" }
SAMPLING_RATE_HZ = 2.0 #@param {type:"slider", min:0.2, max:30, step:0.2}
MAX_FRAMES = 60 #@param {type:"slider", min:6, max:300, step:6}
RESIZE_WIDTH = 0 #@param {type:"slider", min:0, max:1280, step:160}

info = aitbc.video_info(VIDEO_EO)
f = info["fps"]
N = max(1, int(round(f / SAMPLING_RATE_HZ)))                     # formula (4)
fs_actual, dt = f / N, N / f
EXTRACTED_EO = []                                                # list of (frame number, time s, frame)
for k, t, frame in aitbc.iter_video(VIDEO_EO, every_n=N, max_frames=MAX_FRAMES):
    if RESIZE_WIDTH:
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(round(frame.shape[0] * RESIZE_WIDTH / frame.shape[1]))),
                           interpolation=cv2.INTER_AREA)
    EXTRACTED_EO.append((k, t, frame))
EXTRACTION_EO = dict(video=VIDEO_EO, fps=f, N=N, fs=fs_actual, dt=dt, resize_width=RESIZE_WIDTH)
diffs = [aitbc.frame_difference(EXTRACTED_EO[i - 1][2], EXTRACTED_EO[i][2]) for i in range(1, len(EXTRACTED_EO))]
shifts = [np.hypot(*aitbc.global_shift(aitbc.gray(EXTRACTED_EO[i - 1][2]), aitbc.gray(EXTRACTED_EO[i][2])))
          for i in range(1, len(EXTRACTED_EO))]
vol = sum(fr.nbytes for _, _, fr in EXTRACTED_EO)
display(pd.DataFrame({"Value": [f"{f:.2f}", f"{SAMPLING_RATE_HZ:.2f}", N, f"{fs_actual:.3f}", f"{dt:.3f}",
                                f"{info['frames']}", len(EXTRACTED_EO), f"{math.ceil(info['frames'] / N)}",
                                str(EXTRACTED_EO[0][2].shape), f"{vol / 1e6:.1f}"]},
                     index=["frame rate of the video f, frames/s", "required sampling rate f_s, Hz", "step N, frames (4)",
                            "actual sampling rate f_s', Hz (4)", "time between extracted frames Δt, s (4)",
                            "frames in the video", "frames extracted", "frames of the whole video at this step",
                            "shape of an extracted frame", "volume of the set in memory, MB"]))
if len(EXTRACTED_EO) > 1:
    tt = [t for _, t, _ in EXTRACTED_EO[1:]]
    fig, ax = plt.subplots(1, 2, figsize=(15, 3.6))
    ax[0].plot(tt, diffs, "o-", ms=3)
    ax[0].set_xlabel("time, s")
    ax[0].set_ylabel("grey levels")
    ax[0].set_title("Mean absolute difference of successive extracted frames")
    ax[1].plot(tt, shifts, "s-", ms=3, color="C1")
    ax[1].axhline(0.25 * EXTRACTED_EO[0][2].shape[1], color="C3", ls="--", lw=1, label="25 % of the frame width")
    ax[1].set_xlabel("time, s")
    ax[1].set_ylabel("px")
    ax[1].set_title("Displacement of the image between successive extracted frames")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Median difference {np.median(diffs):.2f} grey levels; median displacement {np.median(shifts):.1f} px, "
          f"maximum {np.max(shifts):.1f} px per step of {dt:.2f} s")
""")

md(r"""
## Task 1.4.3. Grid of frames for preliminary visual analysis

**Purpose.** To display the extracted frames as a grid (*contact sheet*) and to use it for a quick review of the fragment: changes of the scene, motion of the camera, presence of objects, defective frames.

**Parameters.**
- `COLUMNS` — number of columns of the grid; `FRAMES_IN_GRID` — number of frames (taken uniformly from the extracted set);
- `SHOW_TIME` — label of every frame with its number and time;
- `SHOW_TRACKS` — boxes of the annotated vehicles (M3OT only).

**Output.** The grid of frames; a table of the frames with the mean brightness, the sharpness (variance of the Laplacian) and the number of annotated vehicles; frames whose sharpness is below half of the median are marked as possibly blurred.

**Control question 8.** Which changes in the scene and in the motion of the camera can be found in the grid? Which frames would be removed from a training set, and why?
""")

code(r"""
#@title 1.4.3 Grid of frames for visual analysis [auto] { run: "auto", display-mode: "form" }
COLUMNS = 6 #@param {type:"slider", min:2, max:10, step:1}
FRAMES_IN_GRID = 24 #@param {type:"slider", min:4, max:60, step:2}
SHOW_TIME = True #@param {type:"boolean"}
SHOW_TRACKS = True #@param {type:"boolean"}

sel = np.unique(np.linspace(0, len(EXTRACTED_EO) - 1, min(FRAMES_IN_GRID, len(EXTRACTED_EO))).round().astype(int))
tab = None
if SHOW_TRACKS and VIDEO_EO_SOURCE.get("dataset") == "M3OT":
    tab = m3ot.tracks(VIDEO_EO_SOURCE["sequence"], "rgb")
grid_frames, labels, rows = [], [], []
for i in sel:
    k, t, fr = EXTRACTED_EO[i]
    n_obj = None
    if tab is not None:
        frame_id = VIDEO_EO_SOURCE["start_frame"] + k
        objs = m3ot.objects(tab, frame_id)
        if fr.shape[1] != 640:
            s = fr.shape[1] / 640
            objs = [dict(o, box=[v * s for v in o["box"]]) for o in objs]
        n_obj = len(objs)
        fr = aitbc.draw_boxes(fr, objs, color=(255, 235, 59), labels=False, thickness=2)
    grid_frames.append(fr)
    labels.append(f"#{k}  t={t:.1f} s")
    rows.append(dict(frame=k, time_s=round(t, 2), mean=round(float(aitbc.gray(EXTRACTED_EO[i][2]).mean()), 1),
                     sharpness=round(aitbc.sharpness(EXTRACTED_EO[i][2])), vehicles=n_obj))
sheet = aitbc.contact_sheet(grid_frames, cols=COLUMNS, width=1800, labels=labels if SHOW_TIME else None)
GRID_EO = sheet
fig, ax = plt.subplots(figsize=(18, 18 * sheet.shape[0] / sheet.shape[1]))
ax.imshow(sheet)
ax.set_title(f"{len(sel)} frames of {os.path.basename(VIDEO_EO)}, step N = {EXTRACTION_EO['N']} "
             f"(Δt = {EXTRACTION_EO['dt']:.2f} s)", fontsize=11)
ax.axis("off")
plt.tight_layout()
plt.show()
t_ = pd.DataFrame(rows)
med = t_.sharpness.median()
t_["note"] = np.where(t_.sharpness < 0.5 * med, "possibly blurred", "")
display(t_.set_index("frame").T if len(t_) <= 12 else t_.set_index("frame"))
""", form=True)

md(r"""
## Task 1.4.4. Thermal video fragment by the same method

**Purpose.** To apply the method of tasks 1.4.1–1.4.3 to the thermal video of the same flight and to compare the two channels in time and in space.

For M3OT the visible and the thermal camera of one drone record the same scene, and the annotated vehicles have the same identifiers in both channels. The authors aligned the channels with an affine transform, but small residual errors remain. Two of them are measured here, with the annotations of the vehicles:

- the **spatial offset** $(\Delta x, \Delta y)$ — the median difference of the positions of the same vehicle in the thermal and in the visible frame;
- the **time offset** $\tau$ — the number of frames by which the thermal stream lags behind the visible one or leads it.

The time offset is the shift $\tau$ that minimises the spread of the offsets around their median. If the streams are synchronised, the spread is minimal at $\tau = 0$; with a time offset, a vehicle that moves with the speed $v$ px/frame is seen at different places in the two channels, and the spread is minimal at the true lag. Only moving vehicles carry information about the lag; in sequences with few moving vehicles the curve is flat and the estimate is uncertain. Multiplied by the speed of a vehicle (1–7 px per frame in M3OT), a lag of 3 frames (0.3 s) gives a position error of 3–20 px, comparable with the size of the vehicle (about 10 px). The time alignment of streams of different sensors is a task of the data-fusion modules of Topics 4–5.

**Parameters.**
- `SAMPLING_RATE_HZ` — sampling rate of the thermal frames (by default the same as in task 1.4.2);
- `COLUMNS`, `FRAMES_IN_GRID` — grid of the thermal frames;
- `SEARCH_LAG` — maximum lag searched, frames. The offsets are estimated from the annotations of the whole sequence (all frames, not only the fragment), because a short fragment contains too few moving vehicles.
- `OWN_VIDEO_IR` — path of an own thermal video on Google Drive; empty — the thermal video of the data set: the same flight for M3OT, otherwise a thermal video of HIT-UAV.

**Output.** The thermal video file in `data/video` and its properties; the grid of the thermal frames; a pair of frames visible / thermal of the same moment; for M3OT the curve of the spread of the offsets against the lag $\tau$, with the found time offset and spatial offset. For HIT-UAV thermal video in task 1.4.1 there is no visible pair: only the grid is produced. The thermal frames of M3OT are shown as recorded, in the black-hot palette (task 1.3.4): the bright vehicles on the dark road are cooler than the asphalt.

**Control question 9.** What error in metres does a time offset of 3 frames at 10 frames per second cause for a vehicle that drives at 15 m/s? How does it affect the fusion of detections of the two channels?
""")

code(r"""
#@title 1.4.4 Thermal video fragment and comparison of the channels [auto] { run: "auto", display-mode: "form" }
SAMPLING_RATE_HZ_IR = 2.0 #@param {type:"slider", min:0.2, max:30, step:0.2}
COLUMNS_IR = 6 #@param {type:"slider", min:2, max:10, step:1}
FRAMES_IN_GRID_IR = 12 #@param {type:"slider", min:4, max:48, step:2}
SEARCH_LAG = 8 #@param {type:"slider", min:1, max:15, step:1}
OWN_VIDEO_IR = "" #@param {type:"string"}

is_m3ot = VIDEO_EO_SOURCE.get("dataset") == "M3OT" and not OWN_VIDEO_IR
if OWN_VIDEO_IR:
    if IN_COLAB and OWN_VIDEO_IR.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    if not os.path.isfile(OWN_VIDEO_IR):
        raise FileNotFoundError(f"{OWN_VIDEO_IR} not found: copy the thermal video to Google Drive or leave OWN_VIDEO_IR empty")
    VIDEO_IR = OWN_VIDEO_IR
    VIDEO_IR_SOURCE = dict(dataset="own video", path=OWN_VIDEO_IR, channel="thermal")
elif is_m3ot:
    seq_ = VIDEO_EO_SOURCE["sequence"]
    start, count = VIDEO_EO_SOURCE["start_frame"] - 1, VIDEO_EO_SOURCE["frames"]
    VIDEO_IR = os.path.join(PROJECT_ROOT, "data", "video", f"m3ot_{seq_}_ir_{start:04d}_{count}" + os.path.splitext(VIDEO_EO)[1])
    if not os.path.exists(VIDEO_IR):
        fr_ir = m3ot.frames(seq_, "ir", start=start, count=count)
        aitbc.write_video(VIDEO_IR, fr_ir, fps=m3ot.CAMERA["fps"], codec="mp4v" if VIDEO_IR.endswith(".mp4") else "MJPG")
        del fr_ir
    VIDEO_IR_SOURCE = dict(dataset="M3OT", sequence=seq_, channel="thermal", start_frame=start + 1, frames=count)
else:
    hv = VIDEO_EO_SOURCE.get("video", "70m-90_1.mov")
    VIDEO_IR = os.path.join(PROJECT_ROOT, "data", "video", "hituav_" + hv)
    if not os.path.exists(VIDEO_IR):
        shutil.copy2(hit.video(hv), VIDEO_IR)
    VIDEO_IR_SOURCE = dict(dataset="HIT-UAV", video=hv, channel="thermal")
info_ir = aitbc.video_info(VIDEO_IR)
N_ir = max(1, int(round(info_ir["fps"] / SAMPLING_RATE_HZ_IR)))                          # formula (4)
EXTRACTED_IR = [(k, t, fr) for k, t, fr in aitbc.iter_video(VIDEO_IR, every_n=N_ir, max_frames=300)]
EXTRACTION_IR = dict(video=VIDEO_IR, fps=info_ir["fps"], N=N_ir, fs=info_ir["fps"] / N_ir, dt=N_ir / info_ir["fps"], resize_width=0)
ch_141 = "visible" if VIDEO_EO_SOURCE.get("channel") != "thermal" else "thermal"
display(pd.DataFrame({f"task 1.4.1 ({ch_141})": [os.path.basename(VIDEO_EO), f"{aitbc.video_info(VIDEO_EO)['fps']:.2f}", EXTRACTION_EO["N"], len(EXTRACTED_EO)],
                      "task 1.4.4 (thermal)": [os.path.basename(VIDEO_IR), f"{info_ir['fps']:.2f}", N_ir, len(EXTRACTED_IR)]},
                     index=["video file", "frame rate, frames/s", "step N (4)", "frames extracted"]))
sel = np.unique(np.linspace(0, len(EXTRACTED_IR) - 1, min(FRAMES_IN_GRID_IR, len(EXTRACTED_IR))).round().astype(int))
GRID_IR = aitbc.contact_sheet([EXTRACTED_IR[i][2] for i in sel], cols=COLUMNS_IR, width=1800,
                              labels=[f"#{EXTRACTED_IR[i][0]}  t={EXTRACTED_IR[i][1]:.1f} s" for i in sel])
fig, ax = plt.subplots(figsize=(18, 18 * GRID_IR.shape[0] / GRID_IR.shape[1]))
ax.imshow(GRID_IR)
ax.set_title(f"Thermal frames of {os.path.basename(VIDEO_IR)}, step N = {N_ir}", fontsize=11)
ax.axis("off")
plt.tight_layout()
plt.show()

if is_m3ot:
    ta, tb = m3ot.tracks(seq_, "rgb"), m3ot.tracks(seq_, "ir")     # annotations of the whole sequence
    curve = []
    for tau in range(-SEARCH_LAG, SEARCH_LAG + 1):
        dx, dy = aitbc.track_offsets(ta, tb, lag=tau)
        if len(dx) < 20:
            continue
        cx, cy = np.median(dx), np.median(dy)
        curve.append((tau, float(np.median(np.hypot(dx - cx, dy - cy))), cx, cy, len(dx)))
    curve = pd.DataFrame(curve, columns=["tau", "spread", "dx", "dy", "pairs"])
    best = curve.loc[curve.spread.idxmin()]
    LAG_RESULT = dict(sequence=seq_, lag_frames=int(best.tau), lag_s=float(best.tau) / m3ot.CAMERA["fps"],
                      offset_dx_px=round(float(best.dx), 1), offset_dy_px=round(float(best.dy), 1),
                      spread_px=round(float(best.spread), 2), spread_at_zero_px=round(float(curve.set_index("tau").spread.get(0, np.nan)), 2))
    k_mid = len(EXTRACTED_EO) // 2
    k0 = EXTRACTED_EO[k_mid][0]
    eo_f = EXTRACTED_EO[k_mid][2]
    ir_f = m3ot.frames(seq_, "ir", start=start + k0, count=1)[0]
    if eo_f.shape[:2] != ir_f.shape[:2]:
        eo_f = cv2.resize(eo_f, (ir_f.shape[1], ir_f.shape[0]))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    ax[0].imshow(eo_f)
    ax[0].set_title(f"visible, frame {k0}", fontsize=10)
    ax[1].imshow(ir_f)
    ax[1].set_title(f"thermal, frame {k0}", fontsize=10)
    for a in ax[:2]:
        a.axis("off")
    ax[2].plot(curve.tau, curve.spread, "o-")
    ax[2].axvline(best.tau, color="C3", ls="--", lw=1)
    ax[2].set_xlabel("lag τ, frames (thermal frame t compared with visible frame t + τ)")
    ax[2].set_ylabel("median spread of the offsets, px")
    ax[2].set_title(f"time offset τ = {int(best.tau)} frames ({best.tau / m3ot.CAMERA['fps']:.1f} s); "
                    f"spatial offset ({best.dx:.1f}, {best.dy:.1f}) px", fontsize=10)
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    display(pd.DataFrame({"Value": list(LAG_RESULT.values())}, index=list(LAG_RESULT.keys())))
else:
    LAG_RESULT = None
    print("This thermal video has no synchronised and annotated visible stream: the comparison of the channels is skipped.")
""", form=True)

md(r"""
## Task 1.4.5. Saving of the frame sets in the folders of the project

**Purpose.** To save the extracted visible and thermal frames in `frames/visible` and `frames/thermal`, to record for every frame its origin (video, frame number, time) in an index table, and to save the grids of frames in `outputs/figures`.

**Parameters.**
- `FORMAT` — PNG (lossless) or JPEG (smaller files, compression artefacts); `JPEG_QUALITY` — quality of JPEG.
- `SET_NAME` — name of the set: a subfolder `frames/<channel>/<SET_NAME>` is created.
- `OVERWRITE` — replace an existing set with the same name.

**File names.** `<set>_<channel>_<frame number>.<format>`, for example `m3ot_1-04_eo_000040.png`. The index table `frames.csv` of every set has the columns: file, source video, frame number in the video, time, frame number in the data set, channel, width, height, file size.

**Output.** The number and volume of the saved files, the first lines of `frames.csv`, the record of the lesson in `project.json` and the tree of `frames/`.

**Control question 10.** Why is the frame number in the source video stored together with each saved frame? Which task of the following lessons needs it?
""")

code(r"""
#@title 1.4.5 Saving of the frame sets { display-mode: "form" }
FORMAT = "png" #@param ["png", "jpg"]
JPEG_QUALITY = 95 #@param {type:"slider", min:50, max:100, step:5}
SET_NAME = "auto" #@param {type:"string"}
OVERWRITE = True #@param {type:"boolean"}


def save_set(extracted, extraction, source, channel):
    name = SET_NAME if SET_NAME and SET_NAME != "auto" else (
        f"m3ot_{source['sequence']}" if source.get("dataset") == "M3OT" else os.path.splitext(os.path.basename(extraction["video"]))[0])
    tag = "eo" if channel == "visible" else "ir"
    folder = os.path.join(PROJECT_ROOT, "frames", channel, name)
    if os.path.isdir(folder) and OVERWRITE:
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    rows = []
    params = [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)] if FORMAT == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    for k, t, fr in extracted:
        fn = f"{name}_{tag}_{k:06d}.{FORMAT}"
        img = fr if fr.ndim == 2 else cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folder, fn), img, params)
        rows.append(dict(file=fn, video=os.path.basename(extraction["video"]), frame_in_video=k, time_s=round(t, 3),
                         frame_in_dataset=(source.get("start_frame", 1) + k) if source.get("dataset") == "M3OT" else None,
                         channel=channel, width=fr.shape[1], height=fr.shape[0],
                         file_kb=round(os.path.getsize(os.path.join(folder, fn)) / 1024, 1)))
    index = pd.DataFrame(rows)
    index.to_csv(os.path.join(folder, "frames.csv"), index=False)
    return folder, index


results = {}
CH_EO = "visible" if VIDEO_EO_SOURCE.get("channel") != "thermal" else "thermal"     # channel of the video of task 1.4.1
for extracted, extraction, source, channel in ((EXTRACTED_EO, EXTRACTION_EO, VIDEO_EO_SOURCE, CH_EO),
                                               (EXTRACTED_IR, EXTRACTION_IR, VIDEO_IR_SOURCE, "thermal")):
    if extraction["video"] == VIDEO_EO and channel == "thermal" and VIDEO_IR == VIDEO_EO and "thermal" in results:
        continue
    folder, index = save_set(extracted, extraction, source, channel)
    results[channel if channel not in results else channel + "_2"] = (folder, index)
for name, grid in (("grid_visible" if CH_EO == "visible" else "grid_thermal_1_4_3", GRID_EO), ("grid_thermal", GRID_IR)):
    cv2.imwrite(os.path.join(PROJECT_ROOT, "outputs", "figures", f"lesson14_{name}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
summary = []
for ch, (folder, index) in results.items():
    summary.append(dict(channel=ch, folder=os.path.relpath(folder, PROJECT_ROOT), files=len(index),
                        volume_MB=round(index.file_kb.sum() / 1024, 1), first=index.file.iloc[0], last=index.file.iloc[-1]))
display(pd.DataFrame(summary).set_index("channel"))
first_index = next(iter(results.values()))[1]
display(first_index.head(5))
meta = update_project_json("1.4", {"lesson_1_4": {
    "video_visible": (os.path.relpath(VIDEO_EO, PROJECT_ROOT) if VIDEO_EO.startswith(PROJECT_ROOT) else VIDEO_EO) if CH_EO == "visible" else None,
    "video_thermal": os.path.relpath(VIDEO_IR, PROJECT_ROOT) if VIDEO_IR.startswith(PROJECT_ROOT) else VIDEO_IR,
    "sampling": {"visible": {k: EXTRACTION_EO[k] for k in ("fps", "N", "fs", "dt")},
                 "thermal": {k: EXTRACTION_IR[k] for k in ("fps", "N", "fs", "dt")}},
    "sets": {ch: os.path.relpath(folder, PROJECT_ROOT) for ch, (folder, _) in results.items()},
    "channel_offset": LAG_RESULT, "source": [VIDEO_EO_SOURCE, VIDEO_IR_SOURCE]}})
print("project.json: lessons", meta["lessons"])
print_tree(os.path.join(PROJECT_ROOT, "frames"), max_files=3)
""", form=True)

md(r"""
## Task 1.4.6. Additional task: video stream of a ground robotic platform

**Purpose.** To apply the same method to the front camera of a ground vehicle (GOOSE data set). The camera records a sequence with time stamps of a recording session (the data set stores selected frames of the sessions). Compared with a UAV, the view is oblique and low, the scene changes quickly near the vehicle and slowly far away, and the traversable part of the frame (road, track, grass) is the main information for the platform.

**Parameters.** `SESSION` — recording session of GOOSE; `N_FRAMES` — number of frames; `SHOW_TRAVERSABLE` — traversable (green), conditionally traversable (yellow) and obstacle (red) areas from the pixel labels; `SAVE` — save the frames in `frames/visible/goose_<session>` with an index table.

**Output.** The grid of the frames with their times from the time stamps, the interval between successive frames, and the share of the traversable area of every frame.

**Control question 11.** Why do the intervals between the frames of the GOOSE sessions differ, while the frames of a video file have a constant interval? What does this mean for a tracking algorithm?
""")

code(r"""
#@title 1.4.6 Video stream of a ground robotic platform (GOOSE) { display-mode: "form" }
SESSION = "2022-09-21_garching_uebungsplatz_2" #@param ["2022-07-22_flight", "2022-08-30_siegertsbrunn_feldwege", "2022-09-21_garching_uebungsplatz_2", "2022-12-07_aying_hills", "2023-03-03_garching_2", "2023-05-17_neubiberg_sunny"]
N_FRAMES = 12 #@param {type:"slider", min:4, max:24, step:2}
SHOW_TRAVERSABLE = True #@param {type:"boolean"}
SAVE = False #@param {type:"boolean"}

frs = [n for n in goose.frames() if n.split("/")[2] == SESSION]
frs = sorted(frs, key=lambda n: int(n.split("__")[1].split("_")[0]))
first = len(frs) // 3
picked = frs[first:first + N_FRAMES]
goose.prefetch(picked)
stamps = np.array([int(n.split("__")[1].split("_")[1]) for n in picked]) / 1e9
imgs, shares = [], []
for lm in picked:
    img, lab = goose.load(lm, scale=0.5)
    tv = goose.traversability(lab)
    valid = tv > 0
    shares.append((tv == 1).sum() / max(valid.sum(), 1))
    if SHOW_TRAVERSABLE:
        col = np.zeros_like(img)
        col[tv == 1], col[tv == 2], col[tv == 3] = (0, 200, 0), (255, 200, 0), (230, 30, 30)
        img = aitbc.overlay(img, col, 0.4, tv > 0)
    imgs.append(img)
t_rel = stamps - stamps[0]
sheet = aitbc.contact_sheet(imgs, cols=4, width=1800, labels=[f"t = {t:.1f} s" for t in t_rel])
fig, ax = plt.subplots(figsize=(18, 18 * sheet.shape[0] / sheet.shape[1]))
ax.imshow(sheet)
ax.set_title(f"GOOSE session {SESSION}: {len(picked)} frames of the front camera of the ground vehicle", fontsize=11)
ax.axis("off")
plt.tight_layout()
plt.show()
display(pd.DataFrame({"time, s": np.round(t_rel, 2), "interval to the previous frame, s": np.r_[np.nan, np.round(np.diff(t_rel), 2)],
                      "traversable share": np.round(shares, 2)}).T)
if SAVE:
    folder = os.path.join(PROJECT_ROOT, "frames", "visible", "goose_" + SESSION)
    os.makedirs(folder, exist_ok=True)
    rows = []
    for k, (lm, t) in enumerate(zip(picked, t_rel)):
        img, _ = goose.load(lm, scale=1.0)
        fn = f"goose_{SESSION}_eo_{k:06d}.png"
        cv2.imwrite(os.path.join(folder, fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append(dict(file=fn, source=lm.split("/")[-1].replace("_labelids.png", ""), time_s=round(float(t), 3)))
    pd.DataFrame(rows).to_csv(os.path.join(folder, "frames.csv"), index=False)
    print(f"Saved {len(rows)} frames in {os.path.relpath(folder, PROJECT_ROOT)}")
""", form=True)

md(r"""
### Report of Lesson 1.4

Double-click this cell and write the answers.

1. Video fragment of task 1.4.1: source, frame rate, duration, data rate (3), compression ratio; answer to control question 6:
2. Chosen sampling rate, step $N$, actual rate and $\Delta t$ (4); median displacement between extracted frames; answer to control question 7:
3. Observations in the grid of frames; answer to control question 8:
4. Time offset and spatial offset between the thermal and the visible channel (task 1.4.4); answer to control question 9:
5. Folders and number of the saved frames; answer to control question 10:
6. (Additional) Answer to control question 11:

---
## Final self-check of Topic 1

1. Which properties of a video file are read with `cv2.VideoCapture.get`, and how is the duration computed?
2. Why is a video decoded frame by frame with a generator instead of loading all frames into a list?
3. Calculate the number of frames of a 5-minute thermal video at 9 frames per second decomposed with $f_s$ = 1.5 Hz. *(N = 6, f_s' = 1.5 Hz, 450 frames.)*
4. Why are the frames of a visible and a thermal camera of the same UAV not aligned automatically, and which two kinds of misalignment were measured in task 1.4.4?
5. Describe the structure of the course project and the purpose of every folder.

## Data sources and licences

The course library reads the files directly from the repositories of the authors; the data are not redistributed with the notebook. When images or results are published, the data sets are cited.

1. **M3OT.** Nie Z., Xue L., Fang Z., Ren J., Wei Y., Zheng J. M3OT: A Multi-Drone Multi-Modality dataset for Multi-Object Tracking. *Scientific Data*, 12, 1927 (2025). https://doi.org/10.1038/s41597-025-06204-0 · Data: https://doi.org/10.6084/m9.figshare.28308887 · Licence CC BY 4.0.
2. **HIT-UAV.** Suo J., Wang T., Zhang X., Chen H., Zhou W., Shi W. HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection. *Scientific Data*, 10, 227 (2023). https://doi.org/10.1038/s41597-023-02066-6 · Data: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset · Licence CC BY 4.0.
3. **Kust4K.** Ouyang J., Wang Q., Shang Y., Jin P., Zhong H., Zhou L., Shen T. An RGB-TIR dataset from UAV platform for robust urban traffic scenes semantic segmentation. *Scientific Data*, 12, 1701 (2025). https://doi.org/10.1038/s41597-025-05994-7 · Data: https://doi.org/10.6084/m9.figshare.29476610 · Licence CC BY 4.0.
4. **VisDrone.** Zhu P., Wen L., Du D., Bian X., Fan H., Hu Q., Ling H. Detection and tracking meet drones challenge. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(11), 7380–7399 (2022). Data: https://github.com/VisDrone/VisDrone-Dataset · Terms of use: see the data set page.
5. **GOOSE.** Mortimer P., Hagmanns R., Granero M., Luettel T., Petereit J., Wuensche H.-J. The GOOSE dataset for perception in unstructured environments. *IEEE ICRA*, 2024. https://arxiv.org/abs/2310.16788 · Data: https://goose-dataset.de · Licence CC BY-SA 4.0.

**Next notebook.** Notebook 03 — Topic 2, Lessons 2.1 and 2.2: the digital image and the video stream of electro-optical surveillance devices; stages of preprocessing of video frames.
""")


# =====================================================================================
def build(path_out, lib_path="aitbc.py"):
    lib = open(lib_path, encoding="utf-8").read()
    assert "\'\'\'" not in lib, "library source must not contain triple single quotes"
    nb = new_notebook()
    nb.metadata = {"colab": {"provenance": [], "toc_visible": True, "gpuType": "T4", "name": os.path.basename(path_out)},
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
    out = sys.argv[1] if len(sys.argv) > 1 else "AITBC_NB02_Colab_frames_and_video.ipynb"
    nb = build(out)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    print(f"written {out}: {len(nb.cells)} cells ({n_md} text, {len(nb.cells) - n_md} code)")
