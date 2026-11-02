# Builder of the English practical notebook of the course "AI Technologies (Basic Course)" (AIT-BC).
# Five topics in one notebook; all demonstrations use real open UAV data sets (course library aitbc).
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n"), False))


def code(text, form=True):
    CELLS.append(("code", text.strip("\n"), form))


# =====================================================================================
md(r"""
# AI Technologies (Basic Course) · Practical notebook
## Computer vision for UAV and UGV tasks in Google Colab: from a video frame to detection and tracking

**Kruty Heroes Military Institute of Telecommunications and Information Technology · Department of Special Information Systems and Robotic Complexes**

The notebook accompanies the five topics of the 40-hour programme. Step by step you take a **real** video recorded from a UAV, decompose it into frames, study the image as a data array, degrade and restore it, extract classical features, train a classifier of thermal objects, detect warm objects and track a moving vehicle with a Kalman filter.

**Result of the work:** a reproducible project with code, figures, tables of metrics and a short final report.

### How to work with the notebook
1. Open the file in [Google Colab](https://colab.research.google.com/) (*File → Upload notebook*) and save your own copy (*File → Save a copy in Drive*).
2. The base route runs on a CPU. A GPU (*Runtime → Change runtime type → T4 GPU*) speeds up only the optional blocks MobileNet, YOLO and ByteTrack.
3. Run the cells from top to bottom. Every block starts with a text cell: **Purpose**, **Parameters**, **What to look at**. The parameters are set in the forms on the right side of the code cells; after a change, run the cell again.
4. After a restart of the runtime run the cells again from cell 0.1.
5. Every figure and table is also saved in the folder `outputs` of the project.

### Data
All demonstrations use real open data sets with manual annotation (sources and licences are at the end). Only the files that a block needs are downloaded, the cache is kept between the blocks.

| Data set | What it contains | Where it is used |
|---|---|---|
| **M3OT** | visible and thermal image sequences of a DJI Mavic 3T UAV, 100–120 m, 10 frames/s, annotated vehicle tracks | Topic 1 (video), 3.3 (camera motion), 5.2 and 5.4 (tracking) |
| **VisDrone** | visible images of camera drones over cities with boxes of pedestrians and vehicles | Topics 2 and 3, 5.3 (YOLO) |
| **HIT-UAV** | thermal (LWIR) images of a DJI Matrice M210 V2, 60–130 m, day and night, boxes of persons, cars and bicycles | Topic 4 (classification), 5.1 (warm objects) |

### Map of the course

| Topic | Practical work in this notebook | Control result |
|---|---|---|
| **1. Google Colab and video data** | project folders, a real UAV video fragment, decomposition into frames with a given sampling rate | video file, table of time stamps, motion-activity plot |
| **2. Digital image** | colour channels, histograms, degradations of the video link, noise filters, CLAHE, letterbox | PSNR/SSIM tables and curves, object size after scaling |
| **3. Classical features** | Sobel, Canny, contours, ORB, homography, compensation of camera motion | edge-density curves, robustness of ORB to rotation and scale |
| **4. Machine learning** | data set of thermal patches, HOG + SVM, hyperparameter map, learning curve, MobileNet (optional) | verified classifier, confusion matrix, comparison of features |
| **5. Detection and tracking** | mask and boxes of warm objects, precision–recall curves day/night, Kalman filter, YOLO11 and ByteTrack (optional) | detection metrics, tracking error and tuning curves |

Schedule of the programme: 10 hours of lectures, 8 hours of group classes and 22 hours of practical classes (training plan AIT-BC, 2026).
""")

# =====================================================================================
md(r"""
---
## 0. Preparation of the runtime

### 0.1. Libraries, random seed and project folders

**Purpose.** Check the runtime, fix the random seed and create the project structure:

| Folder | Content |
|---|---|
| `data` | video files and other source data of the project |
| `frames` | frames extracted from the video |
| `models` | trained models |
| `outputs` | figures, tables (CSV) and the final report (JSON) |

**Parameters.**
- `SEED` — seed of the random number generator. The same seed gives the same noise, the same data split and the same results after a restart.
- `USE_GOOGLE_DRIVE` — keep the project on Google Drive (it survives the end of the Colab session) instead of the temporary disk `/content`.

**What to look at.** The table of versions: it is saved in `outputs/environment.json`, because results of computer vision depend on library versions.
""")

code(r"""
#@title 0.1 Runtime, random seed and project folders { display-mode: "form" }
SEED = 42 #@param {type:"integer"}
USE_GOOGLE_DRIVE = False #@param {type:"boolean"}

import sys, os, time, json, math, platform, shutil, subprocess, importlib, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

for module, package in {"sklearn": "scikit-learn", "skimage": "scikit-image"}.items():
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
import sklearn, skimage

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

random_state = np.random.default_rng(SEED)
np.random.seed(SEED)

if USE_GOOGLE_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT = "/content/drive/MyDrive/AITBC/practical_course"
else:
    PROJECT = os.path.join("/content" if IN_COLAB else os.getcwd(), "aitbc_practical_course")
DIRS = {name: os.path.join(PROJECT, name) for name in ["data", "frames", "models", "outputs"]}
for path in DIRS.values():
    os.makedirs(path, exist_ok=True)
OUT = DIRS["outputs"]

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]


def save_figure(fig, name):
    # save a figure in outputs and show it
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def show_images(images, titles, name, cols=None, cmap="gray", width=4.2):
    # grid of images with titles
    cols = cols or len(images)
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, width * rows * h / w + 0.4 * rows), squeeze=False)
    for ax, im, title in zip(axes.flat, images, titles):
        ax.imshow(im, cmap=cmap if im.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    for ax in list(axes.flat)[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, name)


ENVIRONMENT = {"Python": platform.python_version(), "NumPy": np.__version__, "OpenCV": cv2.__version__,
               "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scikit-image": skimage.__version__,
               "Matplotlib": matplotlib.__version__, "Google Colab": "yes" if IN_COLAB else "no"}
with open(os.path.join(OUT, "environment.json"), "w") as fh:
    json.dump(ENVIRONMENT, fh, indent=2)
display(pd.DataFrame(ENVIRONMENT.items(), columns=["Component", "Version"]).set_index("Component"))
print("Project folder:", PROJECT)
""")

md(r"""
### 0.2. Course library and real data sets

**Purpose.** Write the course library `aitbc` (version 2.1.0, its source is embedded in the cell) and connect the data sets HIT-UAV, VisDrone and M3OT. At this step only the annotation files and the directories of the archives are read (about 10 MB, 15–40 s). Images are downloaded later, when a block needs them.

**Parameters.** `CACHE_ON_DRIVE` — keep the download cache on Google Drive (`MyDrive/AITBC/data_cache`), so that a new session does not download the files again. The same cache is used by the other notebooks of the course.
""")

code(r"""
#@title 0.2 Course library aitbc and data sets { display-mode: "form" }
CACHE_ON_DRIVE = False #@param {type:"boolean"}

LIB_SOURCE = r'''%%LIB%%'''
WORK_DIR = "/content" if IN_COLAB else os.getcwd()
with open(os.path.join(WORK_DIR, "aitbc.py"), "w", encoding="utf-8") as fh:
    fh.write(LIB_SOURCE)
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)
import aitbc
aitbc = importlib.reload(aitbc)

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
hit, vis, m3ot = aitbc.HITUAV(), aitbc.VisDrone(), aitbc.M3OT()
ENVIRONMENT["aitbc"] = aitbc.__version__
n_files, size_mb = aitbc.cache_size()
print(f"aitbc {aitbc.__version__}: HIT-UAV ({len(hit.images)} thermal images, {len(hit.objects)} objects), "
      f"VisDrone ({len(vis.names())} images), M3OT ({len(m3ot.sequences())} sequences) connected in {time.time() - t0:.0f} s")
print(f"Cache {aitbc.DATA_ROOT}: {n_files} files, {size_mb:.1f} MB")
""")

# =====================================================================================
md(r"""
---
## Topic 1. Google Colab and video data

A UAV camera produces a video stream: a sequence of frames with a fixed frame rate $f_v$ (frames per second). For analysis the stream is usually **decomposed into frames** with a lower sampling rate $f_s \le f_v$: neighbouring frames of a 10–30 frames/s video are almost identical, and processing every frame costs time without new information.

### 1.1. Real UAV video fragment

**Purpose.** Build a short video file from a real image sequence of the M3OT data set (DJI Mavic 3T, 100–120 m, 10 frames/s) and check its properties with OpenCV.

**Parameters.**
- `SEQUENCE` — scene of the flight (day, dusk, night; road, crossroads, highway).
- `CHANNEL` — visible camera or thermal camera of the same UAV.
- `START_FRAME`, `N_FRAMES` — first frame and length of the fragment. One frame is about 0.6 MB, a fragment of 60 frames (6 s) is about 35 MB.
- `CODEC` — `mp4v` (.mp4, compact) or `MJPG` (.avi, every frame is a JPEG image).

**What to look at.** The table of video properties (frame rate, number of frames, duration, size); the compression ratio of the video against the source PNG frames; the frames with annotated vehicles: the boxes move between the frames, the scene stays in place.
""")

code(r"""
#@title 1.1 Real UAV video fragment (M3OT) { display-mode: "form" }
SEQUENCE = "1-04 · day, suburban highway" #@param ["1-01 · day, suburban road", "1-02 · night, city crossroads", "1-03 · day, city road", "1-04 · day, suburban highway", "1-05 · day, crossroads", "1-06 · dusk, highway", "1-07 · day, crossroads", "1-08 · dusk, crossroads", "1-09 · day, city street with parked cars"]
CHANNEL = "visible" #@param ["visible", "thermal"]
START_FRAME = 1 #@param {type:"slider", min:1, max:240, step:1}
N_FRAMES = 60 #@param {type:"slider", min:20, max:150, step:10}
CODEC = "mp4v (.mp4)" #@param ["mp4v (.mp4)", "MJPG (.avi)"]

SEQ = SEQUENCE.split(" ")[0]
CH = "rgb" if CHANNEL == "visible" else "ir"
FPS = m3ot.CAMERA["fps"]
n_total = m3ot.n_frames(SEQ, CH)
start = int(min(START_FRAME - 1, max(n_total - N_FRAMES, 0)))
count = int(min(N_FRAMES, n_total - start))

t0 = time.time()
png_paths = m3ot.prefetch(SEQ, CH, start=start, count=count)
FRAMES = m3ot.frames(SEQ, CH, start=start, count=count)
t_download = time.time() - t0
fourcc, ext = ("mp4v", ".mp4") if CODEC.startswith("mp4v") else ("MJPG", ".avi")
VIDEO = os.path.join(DIRS["data"], f"m3ot_{SEQ}_{CH}_{start + 1:04d}_{count}{ext}")
aitbc.write_video(VIDEO, FRAMES, fps=FPS, codec=fourcc)
info = aitbc.video_info(VIDEO)
png_mb = sum(os.path.getsize(p) for p in png_paths) / 1e6
TRACKS = m3ot.tracks(SEQ, CH)                               # annotation of the whole sequence

table = pd.DataFrame([
    ("Sequence", f"{SEQ} ({m3ot.SEQUENCES[SEQ]}), {CHANNEL} channel"),
    ("Frames of the fragment", f"{start + 1}–{start + count} of {n_total}"),
    ("Frame size", f"{info['width']} × {info['height']} px"),
    ("Frame rate f_v", f"{info['fps']:.1f} frames/s"),
    ("Duration", f"{info['duration_s']:.1f} s"),
    ("Codec", info["codec"]),
    ("Source PNG frames", f"{png_mb:.1f} MB"),
    ("Video file", f"{info['size_mb']:.2f} MB (compression ratio {png_mb / max(info['size_mb'], 1e-6):.0f} : 1)"),
    ("Annotated vehicle tracks in the sequence", int(TRACKS.id.nunique())),
    ("Time of download and assembly", f"{t_download:.0f} s"),
], columns=["Property", "Value"]).set_index("Property")
display(table)

picks = [0, count // 2, count - 1]
views, titles = [], []
for k in picks:
    objs = m3ot.objects(TRACKS, start + k + 1)
    views.append(aitbc.draw_boxes(FRAMES[k], objs, color=(255, 230, 0), labels=False, thickness=2))
    titles.append(f"frame {start + k + 1}, t = {k / FPS:.1f} s, {len(objs)} vehicles")
show_images(views, titles, "1_1_video_frames.png", cols=3, width=4.6)
""")

md(r"""
### 1.2. Decomposition of the video into frames

**Purpose.** Write a function that reads the video and saves frames with a given sampling rate $f_s$, and build the table of time stamps. A frame with number $n$ has the time stamp $t_n = n / f_v$; a frame is saved when $t_n$ reaches the next sampling instant $0,\ 1/f_s,\ 2/f_s, \dots$ The time stamp of the last frame is $(N_v - 1)/f_v$, where $N_v$ is the number of frames in the video, therefore the number of saved frames is

$$N_s = \min\left(\left\lfloor \frac{(N_v - 1)\, f_s}{f_v} \right\rfloor + 1,\; N_v\right).$$

When $f_s \ge f_v$ every frame is saved and a further increase of $f_s$ gives nothing.

**Parameters.**
- `SAMPLING_FPS` — sampling rate $f_s$, frames/s.
- `MAX_FRAMES` — upper limit of the number of saved frames.

**What to look at.**
1. The **motion-activity plot**: the mean absolute difference of neighbouring frames shows when the scene changes (moving vehicles, camera motion). The saved frames are marked on it: with a low $f_s$ short events fall between the samples.
2. The plot $N_s(f_s)$ with the saturation at $f_s = f_v$.
3. The contact sheet: a quick visual review of the saved frames.
""")

code(r"""
#@title 1.2 Decomposition of the video into frames { display-mode: "form" }
SAMPLING_FPS = 2.0 #@param {type:"slider", min:0.5, max:12, step:0.5}
MAX_FRAMES = 100 #@param {type:"slider", min:5, max:300, step:5}


def extract_frames(video_path, destination, sampling_fps=2.0, max_frames=100):
    # save frames of a video with the sampling rate sampling_fps; returns the table of time stamps
    if sampling_fps <= 0 or max_frames < 1:
        raise ValueError("The sampling rate must be positive and the number of frames at least one.")
    os.makedirs(destination, exist_ok=True)
    for old in os.listdir(destination):
        if old.startswith("frame_") and old.endswith(".jpg"):
            os.remove(os.path.join(destination, old))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open the video file {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("The video file has no valid frame rate.")
    records, index, next_sample = [], 0, 0.0
    try:
        while len(records) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index / fps
            if timestamp + 1e-9 >= next_sample:
                name = f"frame_{len(records):04d}.jpg"
                if not cv2.imwrite(os.path.join(destination, name), frame):
                    raise OSError(f"Cannot save the frame {name}")
                records.append({"frame": name, "time_s": round(timestamp, 3), "source_index": index})
                next_sample += 1.0 / sampling_fps
            index += 1
    finally:
        cap.release()
    return pd.DataFrame(records)


frames_table = extract_frames(VIDEO, DIRS["frames"], SAMPLING_FPS, MAX_FRAMES)
frames_table.to_csv(os.path.join(OUT, "frame_times.csv"), index=False)
interval = frames_table.time_s.diff().dropna()
print(f"Saved {len(frames_table)} frames of {info['frames']}; interval between samples "
      f"{interval.mean():.2f} s (requested {1 / SAMPLING_FPS:.2f} s)")
display(frames_table.head(8))

# motion activity: mean absolute difference of neighbouring frames of the whole fragment
activity = np.array([aitbc.frame_difference(FRAMES[i - 1], FRAMES[i]) for i in range(1, count)])
t_act = np.arange(1, count) / FPS
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.7, 1]})
axes[0].plot(t_act, activity, color=COLORS[0], lw=1.4, label="difference of neighbouring frames")
for t in frames_table.time_s:
    axes[0].axvline(t, color=COLORS[1], lw=0.8, alpha=0.6)
axes[0].plot([], [], color=COLORS[1], lw=0.8, label=f"saved frames, f_s = {SAMPLING_FPS} frames/s")
axes[0].set(xlabel="time, s", ylabel="mean |ΔI|, grey levels", title="Motion activity of the fragment")
axes[0].legend(loc="upper right")
fs_grid = np.arange(0.5, 15.01, 0.5)
n_theory = np.minimum(np.floor((info["frames"] - 1) * fs_grid / info["fps"] + 1e-9) + 1, info["frames"])
axes[1].plot(fs_grid, n_theory, color=COLORS[2], lw=2)
axes[1].axvline(info["fps"], color="grey", ls="--", lw=1)
axes[1].text(info["fps"], n_theory.max() * 0.5, "  f_s = f_v", color="grey")
axes[1].scatter([SAMPLING_FPS], [len(frames_table)], color=COLORS[1], zorder=3, label="this run")
axes[1].set(xlabel="sampling rate f_s, frames/s", ylabel="saved frames N_s", title="Number of frames against f_s")
axes[1].legend(loc="lower right")
fig.tight_layout()
save_figure(fig, "1_2_sampling.png")

saved = [cv2.cvtColor(cv2.imread(os.path.join(DIRS["frames"], f)), cv2.COLOR_BGR2RGB) for f in frames_table.frame[:18]]
sheet = aitbc.contact_sheet(saved, cols=6, width=1500, labels=[f"t = {t:.1f} s" for t in frames_table.time_s[:18]])
fig, ax = plt.subplots(figsize=(15, 15 * sheet.shape[0] / sheet.shape[1]))
ax.imshow(sheet); ax.axis("off"); ax.set_title("Saved frames (first 18)")
save_figure(fig, "1_2_contact_sheet.png")
""")

md(r"""
**Task 1.**
1. Change `SAMPLING_FPS` to 0.5, 2, 5 and 10. Compare the number of files and the interval between samples with the formula for $N_s$.
2. Find on the motion-activity plot the moments of the largest change of the scene. At which $f_s$ do they still fall between the saved frames?
3. Repeat the decomposition for the thermal channel of the same sequence (1.1, `CHANNEL = thermal`). Is the motion activity of the thermal video higher or lower, and why?
4. To process your own video, upload it to Google Drive and pass its path to `extract_frames`.
""")
