# Builder of Notebook 04 of the course "AI Technologies (Basic Course)" (AIT-BC).
# Topic 2, Lessons 2.3 and 2.4 (practical). All exercises use real open data sets.
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n"), False))


def code(text, form=False):
    CELLS.append(("code", text.strip("\n"), form))


SOURCES = ('["project set: visible (Lesson 1.4)", "project set: thermal (Lesson 1.4)", '
           '"VisDrone: visible, oblique view over a city", "M3OT: visible, dusk and night", "Kust4K: visible", '
           '"Kust4K: thermal (colour palette)", "HIT-UAV: thermal (white-hot)", "GOOSE: ground vehicle, front camera", '
           '"own frame (block 0.3)"]')

# =====================================================================================
md(r"""
# AI Technologies (Basic Course) · Notebook 04
## Topic 2. Digital image and preprocessing of frames in electro-optical systems

| Lesson | Type, duration | Title |
|---|---|---|
| **2.3** | Practical class, 2 h | Channel decomposition and geometric normalisation of a video frame |
| **2.4** | Practical class, 2 h | Filtering methods for the increase of the visual informativeness of video frames |

Course: *AI Technologies (Basic Course)*, AIT-BC · Content module 1 · Kruty Heroes Military Institute of Telecommunications and Information Technology, Kyiv, 2026.

**How to work with this notebook**

1. Run the cells from top to bottom: *Runtime → Run all*, or `Shift+Enter` cell by cell. Section 0 is always run first; after a restart of the runtime it is run again.
2. Every task begins with a text cell: the purpose, the parameters of the form, the formulas, the expected result and the control question. The answers are written in the text cell **Report** at the end of each lesson.
3. Cells marked **(code shown)** are the practical skills of the lesson: read the code, change it and run it again. Cells marked **[auto]** run again after every change of a parameter.
4. The results are saved in the course project on Google Drive (`frames`, `outputs`, `project.json`), created in Notebook 02. If Notebook 02 was not run, block 0.2 creates the project and the frame sets of Lesson 1.4 from the same frames of the data set.

**Data.** The frames of the course project (Lesson 1.4: visible and thermal frames of a DJI Mavic 3T, M3OT data set) and real frames of open data sets with manual annotation: VisDrone (camera drones over cities), Kust4K (visible and thermal pairs of a UAV over roads, pixel labels), HIT-UAV (thermal frames of a reconnaissance UAV, boxes of persons and vehicles) and GOOSE (front camera of a ground vehicle, pixel labels). Noise of a known kind and level is added to real frames in Lesson 2.4, so that every filter is compared with the clean frame; the added part is stated in each task.

**Contents**

- **0. Preparation** — 0.1 runtime, course library and data sets · 0.2 course project and the frame sets of Lesson 1.4 · 0.3 own frame (optional)
- **Lesson 2.3 (practical class)**
  - 2.3.1 Loading of a colour frame and its decomposition into the channels R, G, B
  - 2.3.2 Histograms of brightness of the colour channels
  - 2.3.3 Conversion of a frame between BGR, RGB, grey and HSV
  - 2.3.4 Region of interest (ROI) of a frame for an applied task
  - 2.3.5 Change of the size of a frame and the letterbox transformation
- **Lesson 2.4 (practical class)**
  - 2.4.1 Real frames and the kinds of noise
  - 2.4.2 Gaussian Blur and Median Blur: comparison of effectiveness
  - 2.4.3 CLAHE for a night frame and the gain of visual informativeness
  - 2.4.4 Thermal frame: suitability of a contrast enhancement method
- **Summary**: self-check of Topic 2, data sources and licences
""")

# =====================================================================================
md(r"""
---
# 0. Preparation

### Block 0.1. Runtime, course library and data sets

**Purpose.** To prepare the runtime: the libraries, the course library `aitbc` (version 2.3.0 adds the noise models and the measures of Lesson 2.4) and the open data sets of the course.

**Parameters.**
- `CACHE_ON_DRIVE` — keep the downloaded frames on Google Drive (they remain after the end of the session); otherwise they are kept on the disk of the session.
- `PREFETCH_ALL` — download all frames of the notebook now (about 60 MB); otherwise every task downloads its frames on first use.

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
hit, vis, kust, m3ot, goose = aitbc.HITUAV(), aitbc.VisDrone(), aitbc.Kust4K(), aitbc.M3OT(), aitbc.GOOSE()
M3OT_LOW_LIGHT = [("1-02", 679), ("1-06", 300), ("1-08", 300), ("1-02", 100), ("1-06", 100), ("1-08", 100), ("1-02", 500), ("1-08", 500)]
KUST_PAIRS = ("02566N", "00031D", "02807N", "00580D", "03135N", "00990D", "03895N", "00355D")
GOOSE_FRAMES = [goose.CURATED[i] for i in (1, 3, 4, 0, 2, 7, 8, 11)]          # summer, March, December, rain
HIT_NIGHT = (hit.images[hit.images.night & (hit.images.Person >= 3) & (hit.images.split != "train")]
             .sample(20, random_state=7).file.tolist())                       # night frames with 3 or more persons
if PREFETCH_ALL:
    vis.prefetch(vis.CURATED)
    kust.prefetch(list(KUST_PAIRS))
    hit.prefetch(list(hit.EXAMPLES) + HIT_NIGHT[:12])
    goose.prefetch(GOOSE_FRAMES)
    for seq, f in M3OT_LOW_LIGHT:
        for ch in ("rgb", "ir"):
            m3ot.prefetch(seq, ch, start=f, count=1)
VERSIONS = {"Python": platform.python_version(), "NumPy": np.__version__, "OpenCV": cv2.__version__,
            "pandas": pd.__version__, "Matplotlib": matplotlib.__version__, "aitbc": aitbc.__version__}
n_files, size_mb = aitbc.cache_size()
display(pd.DataFrame({"Value": list(VERSIONS.values()) + ["Google Colab" if IN_COLAB else "local Jupyter",
                                                           f"{aitbc.DATA_ROOT}: {n_files} files, {size_mb:.0f} MB",
                                                           f"{time.time() - t0:.0f} s"]},
                     index=list(VERSIONS) + ["environment", "cache of the data sets", "preparation time"]))
""", form=True)

md(r"""
### Block 0.2. Course project and the frame sets of Lesson 1.4

**Purpose.** To connect the course project created in Notebook 02 and to find the frame sets saved in task 1.4.5 (`frames/visible/<set>` and `frames/thermal/<set>` with the index table `frames.csv`). The tasks of Lesson 2.3 read these frames from the files, as frames of an own video would be read.

**Parameters.**
- `USE_DRIVE` — the project is on Google Drive (Colab asks for access to the Google account); otherwise it is on the disk of the session.
- `PROJECT_PATH` — folder of the project; the same as in Notebook 02.

**What the code does.** Creates the missing folders of the project. If there is no visible frame set, it creates the set `m3ot_1-04` of task 1.4.5 from the same frames of the M3OT data set (sequence 1-04, every 5th frame, 40 frames of each channel, about 45 MB). It defines the functions `load_frame(source, number)` — a frame of a set or of a data set with its annotation and the path of its file — and `update_project_json(lesson, extra)`.

**Output.** The table of the frame sets of the project and the tree of the folder `frames`.
""")

code(r"""
#@title 0.2 Course project and the frame sets of Lesson 1.4 { display-mode: "form" }
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
SUBDIRS = ["data/images", "data/video", "data/thermal", "data/annotations", "frames/visible", "frames/thermal",
           "models", "outputs/figures", "outputs/video", "outputs/reports"]
for d in SUBDIRS:
    os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")


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


def print_tree(path, prefix="", max_files=3):
    entries = sorted(os.listdir(path), key=lambda e: (not os.path.isdir(os.path.join(path, e)), e))
    files = [e for e in entries if not os.path.isdir(os.path.join(path, e))]
    shown = [e for e in entries if os.path.isdir(os.path.join(path, e))] + files[:max_files]
    for i, e in enumerate(shown):
        last = i == len(shown) - 1 and len(files) <= max_files
        full = os.path.join(path, e)
        size = "" if os.path.isdir(full) else f"  ({os.path.getsize(full) / 1024:.0f} KB)"
        print(prefix + ("└── " if last else "├── ") + e + ("/" if os.path.isdir(full) else "") + size)
        if os.path.isdir(full):
            print_tree(full, prefix + ("    " if last else "│   "), max_files)
    if len(files) > max_files:
        print(prefix + f"└── … {len(files) - max_files} more files")


def find_sets():
    # frame sets of the project: {channel: {set name: index table frames.csv}}
    sets = {}
    for ch in ("visible", "thermal"):
        base = os.path.join(PROJECT_ROOT, "frames", ch)
        for name in sorted(os.listdir(base)):
            idx = os.path.join(base, name, "frames.csv")
            if os.path.isfile(idx) and "_lb" not in name:
                sets.setdefault(ch, {})[name] = pd.read_csv(idx)
    return sets


def build_m3ot_set(seq="1-04", step=5, count=40):
    # the frame set of task 1.4.5, made from the frames of the data set (used when Notebook 02 was not run)
    for ch, tag, mch in (("visible", "eo", "rgb"), ("thermal", "ir", "ir")):
        folder = os.path.join(PROJECT_ROOT, "frames", ch, f"m3ot_{seq}")
        os.makedirs(folder, exist_ok=True)
        rows = []
        for i, fr in enumerate(m3ot.frames(seq, mch, start=0, count=count, step=step)):
            k = i * step
            fn = f"m3ot_{seq}_{tag}_{k:06d}.png"
            cv2.imwrite(os.path.join(folder, fn), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            rows.append(dict(file=fn, video="frames of the data set (Notebook 02 was not run)", frame_in_video=k,
                             time_s=k / m3ot.CAMERA["fps"], frame_in_dataset=k + 1, channel=ch,
                             width=fr.shape[1], height=fr.shape[0],
                             file_kb=round(os.path.getsize(os.path.join(folder, fn)) / 1024, 1)))
        pd.DataFrame(rows).to_csv(os.path.join(folder, "frames.csv"), index=False)


SETS = find_sets()
if "visible" not in SETS:
    print("No visible frame set in the project: the set m3ot_1-04 of task 1.4.5 is created from the M3OT data set ...")
    build_m3ot_set()
    SETS = find_sets()


def set_frame(channel, number=1, name=None):
    # frame `number` (1...8, evenly spaced over the set) of a frame set of the project
    if channel not in SETS:
        raise ValueError(f"there is no {channel} frame set in the project")
    name = name or next(iter(SETS[channel]))
    index = SETS[channel][name]
    i = int(np.round(np.linspace(0, len(index) - 1, 8))[(int(number) - 1) % 8])
    row = index.iloc[i]
    path = os.path.join(PROJECT_ROOT, "frames", channel, name, row.file)
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    objs = []
    if name.startswith("m3ot_") and pd.notna(row.get("frame_in_dataset", np.nan)):
        seq = name.split("_")[1]
        objs = m3ot.objects(m3ot.tracks(seq, "rgb" if channel == "visible" else "ir"), int(row.frame_in_dataset))
        s = img.shape[1] / 640.0
        if abs(s - 1) > 1e-6:
            objs = [dict(o, box=[v * s for v in o["box"]]) for o in objs]
    return dict(img=img, path=path, objects=objs, labels=None, channel=channel,
                palette=(m3ot.PALETTE if name.startswith("m3ot_") else None) if channel == "thermal" else None,
                caption=f"project set {channel}/{name}, {row.file} (t = {row.time_s:.1f} s)")


def _to_file(img, name):
    folder = os.path.join(WORK_DIR, "aitbc_frames")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def load_frame(source, number=1):
    # Real frame as an RGB array (H, W, 3) with the path of its file and its annotation:
    # objects = list of dict(cls, box=[x1, y1, x2, y2]); labels = pixel labels or None.
    k = (int(number) - 1) % 8
    if source.startswith("project set"):
        return set_frame("thermal" if "thermal" in source else "visible", number)
    if source.startswith("VisDrone"):
        name = vis.CURATED[k]
        img, objs = vis.load(name)
        return dict(img=img, path=vis.z.get(f"VisDrone2019-DET-val/images/{name}.jpg"), objects=objs, labels=None,
                    channel="visible", palette=None, caption=f"VisDrone {name}")
    if source.startswith("M3OT"):
        seq, f = M3OT_LOW_LIGHT[k]
        img = m3ot.frames(seq, "rgb", start=f, count=1)[0]
        return dict(img=img, path=m3ot.prefetch(seq, "rgb", start=f, count=1)[0],
                    objects=m3ot.objects(m3ot.tracks(seq, "rgb"), f + 1), labels=None, channel="visible", palette=None,
                    caption=f"M3OT {seq} ({m3ot.SEQUENCES[seq]}), visible, frame {f + 1}")
    if source.startswith("Kust4K"):
        thermal = "thermal" in source
        p = kust.pair(KUST_PAIRS[k])
        objs = aitbc.boxes_from_labels(p["labels"], (4, 7), kust.CLASSES, min_area=30)
        img = p["tir"] if thermal else p["rgb"]
        return dict(img=img, path=kust._zip("tir" if thermal else "rgb").get(p["id"] + ".png"), objects=objs,
                    labels=p["labels"], channel="thermal" if thermal else "visible", palette=kust.PALETTE if thermal else None,
                    caption=f"Kust4K {p['id']} ({'night' if p['night'] else 'day'}), {'thermal' if thermal else 'visible'}")
    if source.startswith("HIT-UAV"):
        name = hit.EXAMPLES[k]
        img, objs = hit.load(name)
        r = hit.images.set_index("file").loc[name]
        return dict(img=aitbc.to_rgb(img), path=hit.path(name), objects=objs, labels=None, channel="thermal",
                    palette=hit.PALETTE, caption=f"HIT-UAV {name}: {r.altitude} m, camera angle {r.angle}°, {'night' if r.night else 'day'}")
    if source.startswith("GOOSE"):
        name = GOOSE_FRAMES[k]
        img, lab = goose.load(name, scale=0.5)
        return dict(img=img, path=_to_file(img, "goose_" + os.path.basename(name).replace("_labelids.png", "")), objects=[],
                    labels=lab, channel="visible", palette=None, caption=f"GOOSE {name.split('/')[2]} (front camera of the ground vehicle)")
    own = globals().get("OWN_FRAME")
    if own is None:
        print("There is no own frame: load it in block 0.3. The frame of the project set is used instead.")
        return set_frame("visible", number)
    return dict(img=own, path=_to_file(own, "own_frame"), objects=[], labels=None, channel="visible", palette=None,
                caption="own frame (block 0.3)")


rows = []
for ch, sets in SETS.items():
    for name, index in sets.items():
        rows.append(dict(channel=ch, set=name, frames=len(index), size=f"{index.width.iloc[0]} × {index.height.iloc[0]}",
                         duration_s=float(index.time_s.max() - index.time_s.min()), source=str(index.video.iloc[0])))
display(pd.DataFrame(rows).set_index(["channel", "set"]))
print(os.path.join(PROJECT_ROOT, "frames") + "/")
print_tree(os.path.join(PROJECT_ROOT, "frames"))
""", form=True)

md(r"""
### Block 0.3. Own frame (optional)

**Purpose.** The tasks of Lesson 2.3 and tasks 2.4.3 can process a frame of the participant's own (for example, a frame of a course video) instead of the frames of the project: choose `own frame (block 0.3)` as the source.

**Parameters.**
- `SOURCE` — `none`, upload from this computer, a file path (Google Drive or `/content`), or an Internet address (URL).
- `FILE_PATH`, `URL` — location of the frame; a path that starts with `/content/drive` mounts Google Drive automatically.
- `MAX_WIDTH` — larger frames are reduced to this width.

**Note.** Restricted or classified material must not be uploaded to a public cloud service.
""")

code(r"""
#@title 0.3 Own frame (optional) { display-mode: "form" }
SOURCE = "none" #@param ["none", "upload from this computer", "file path (Google Drive or /content)", "URL"]
FILE_PATH = "/content/drive/MyDrive/AITBC/data/images/frame.jpg" #@param {type:"string"}
URL = "" #@param {type:"string"}
MAX_WIDTH = 1280 #@param {type:"integer"}


def decode_image(data):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("the data are not an image that OpenCV can read")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


OWN_FRAME = None
try:
    if SOURCE.startswith("upload"):
        if not IN_COLAB:
            raise RuntimeError("upload works only in Google Colab; use the file path option")
        from google.colab import files
        uploaded = files.upload()
        OWN_FRAME = decode_image(next(iter(uploaded.values())))
    elif SOURCE.startswith("file"):
        if IN_COLAB and FILE_PATH.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
            from google.colab import drive
            drive.mount("/content/drive")
        with open(FILE_PATH, "rb") as fh:
            OWN_FRAME = decode_image(fh.read())
    elif SOURCE == "URL":
        import urllib.request
        with urllib.request.urlopen(URL, timeout=30) as resp:
            OWN_FRAME = decode_image(resp.read())
except Exception as e:
    print("The frame was not loaded:", e)

if OWN_FRAME is not None:
    h0, w0 = OWN_FRAME.shape[:2]
    if w0 > MAX_WIDTH:
        OWN_FRAME = cv2.resize(OWN_FRAME, (MAX_WIDTH, int(h0 * MAX_WIDTH / w0)), interpolation=cv2.INTER_AREA)
    print("Own frame loaded:", OWN_FRAME.shape)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(OWN_FRAME)
    ax.set_title("Own frame")
    ax.axis("off")
    plt.show()
else:
    print("Own frame is not used: the tasks work with the frames of the project and of the open data sets.")
""", form=True)


# =====================================================================================
md(r"""
---
# Lesson 2.3 (Practical class). Channel decomposition and geometric normalisation of a video frame

**Learning questions**
1. Loading of a colour frame and its decomposition into the separate channels R, G, B.
2. Construction of brightness histograms for the separate colour channels.
3. Conversion of a frame between the colour spaces BGR, RGB, Grayscale and HSV.
4. Selection of a region of interest (ROI) in a video frame according to the applied task.
5. Change of the geometric size of a video frame and application of the letterbox transformation.

**Result of the lesson.** Figures of the channels and histograms and a table of the statistics of the channels in `outputs`; the region of interest of an applied task in `project.json`; a frame set converted to the input of a neural network (`frames/visible/<set>_lb<size>`) with labels of the vehicles in the format of YOLO detectors, which is used in Topic 5.
""")

md(r"""
## Task 2.3.1. Loading of a colour frame and its decomposition into the channels R, G, B

**Purpose.** To read a colour frame from its file with OpenCV, to decompose it into three matrices of brightness and to compare what each channel shows.

**Parameters.**
- `SOURCE` — frame set of the project (Lesson 1.4) or a frame of a data set; `FRAME_NUMBER` — frame 1…8 of the source.
- `SHOW_CHANNELS` — every channel as grey levels (the values of the matrix) or in the colour of its filter.

**Code (shown).** `cv2.imread` returns an array H × W × 3 in the order **B, G, R**; `cv2.split` gives three matrices H × W of the type `uint8`; `cv2.merge([r, g, b])` builds the frame in the order R, G, B, which Matplotlib expects.

**Expected result.** The frame and its three channels; a table of the statistics of every channel; the matrix of correlation of the channels; for frames with annotated objects — the signal-to-clutter ratio (SCR, Notebook 01) of the objects in every channel.

**What to observe.** In a natural scene the three channels are strongly correlated (in the default M3OT frame R–G 0.98, G–B 0.90): most of the information is in the brightness, the colour adds a smaller part. A channel can nevertheless separate objects better than the others: in the default frame the annotated vehicles have the median SCR of about 0.65 in the channel G against about 0.4 in R and 0.48 in grey. A thermal frame stored as RGB has three equal channels: the decomposition gives no new information.

**Control question 1.** Why are the channels of a natural scene strongly correlated? Why can the channel that separates the objects best differ between scenes?
""")

code(r"""
#@title 2.3.1 Colour frame and its channels R, G, B (code shown) { run: "auto" }
SOURCE = "project set: visible (Lesson 1.4)" #@param %%SOURCES%%
FRAME_NUMBER = 1 #@param {type:"slider", min:1, max:8, step:1}
SHOW_CHANNELS = "in grey levels" #@param ["in grey levels", "in the colour of the channel"]

fr = load_frame(SOURCE, FRAME_NUMBER)
bgr = cv2.imread(fr["path"], cv2.IMREAD_COLOR)      # array H x W x 3, order of the channels B, G, R
if bgr is None:
    raise IOError(f"OpenCV cannot read {fr['path']}")
b, g, r = cv2.split(bgr)                             # three matrices H x W of the type uint8
rgb = cv2.merge([r, g, b])                           # the same frame in the order R, G, B
FRAME_23 = dict(fr, img=rgb)                         # used in tasks 2.3.2-2.3.4

fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
ax[0].imshow(rgb)
ax[0].set_title(fr["caption"], fontsize=9)
for i, (a, name, ch) in enumerate(zip(ax[1:], "RGB", (r, g, b))):
    if SHOW_CHANNELS.startswith("in grey"):
        a.imshow(ch, cmap="gray", vmin=0, vmax=255)
    else:
        only = np.zeros_like(rgb)
        only[..., i] = ch                                # the other two channels are zero
        a.imshow(only)
    a.set_title(f"channel {name}: mean {ch.mean():.1f}, std {ch.std():.1f}", fontsize=10)
for a in ax:
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson23_channels.png"), dpi=110)
plt.show()

stats = pd.DataFrame([dict(channel=n, mean=float(c.mean()), std=float(c.std()), min=int(c.min()), max=int(c.max()),
                           share_0=float((c == 0).mean()), share_255=float((c == 255).mean())) for n, c in zip("RGB", (r, g, b))]).set_index("channel")
display(stats.round(3))
equal = np.array_equal(r, g) and np.array_equal(g, b)
if equal:
    print("The three channels are equal (grey or thermal frame stored as RGB): the frame has one informative channel.")
else:
    corr = np.corrcoef(np.stack([r.ravel(), g.ravel(), b.ravel()]).astype(np.float64))
    display(pd.DataFrame(corr, index=list("RGB"), columns=list("RGB")).style.format(precision=2).set_caption("Correlation of the channels"))
if fr["objects"] and not equal:
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    masks = [aitbc.box_mask(rgb.shape, o["box"]) for o in fr["objects"]]
    row = {}
    for name, plane in (("R", r), ("G", g), ("B", b), ("grey", grey)):
        v = [aitbc.signal_to_clutter(plane, m) for m in masks]
        v = [x for x in v if np.isfinite(x)]
        row[name] = float(np.median(v)) if v else np.nan
    display(pd.DataFrame([row], index=[f"median SCR of {len(masks)} annotated objects"]).round(2))
""")

md(r"""
## Task 2.3.2. Histograms of brightness of the colour channels

**Purpose.** To construct the histogram of every channel and to read from it the brightness, the contrast, the used part of the range and the clipped pixels; to compare a day and a night frame.

**Histogram.** With B bins of the width 256 / B the histogram of a channel counts the pixels in every bin, and its normalised form is the share of pixels:

$$h_j = \#\left\{(x, y) : \left\lfloor I(x, y)\,B / 256 \right\rfloor = j\right\}, \qquad p_j = \frac{h_j}{H W}, \qquad F_j = \sum_{i \le j} p_i, \tag{1}$$

where $F_j$ is the cumulative histogram. The used part of the range of a channel is measured between its 1st and 99th percentiles, so that single extreme pixels do not determine it:

$$D = P_{99} - P_{1}. \tag{2}$$

**Parameters.**
- `COMPARE_WITH`, `COMPARE_NUMBER` — second frame for the comparison (by default a dusk or night frame of M3OT); `none` — only the frame of task 2.3.1.
- `BINS` — number of bins B; `LOG_SCALE` — logarithmic axis of the number of pixels; `CUMULATIVE` — cumulative histograms $F_j$.

**Code (shown).** `cv2.calcHist([bgr], [channel], None, [B], [0, 256])` for the channels 0, 1, 2 of a BGR array (B, G, R).

**Expected result.** Histograms of the channels of the two frames; a table with the mean, the median, $P_1$, $P_{99}$, the range $D$ (2), the number of used levels, the entropy (Notebook 02, formula (2)) and the share of clipped pixels (0 or 255); the table is saved in `outputs/reports/lesson23_histograms.csv`.

**What to observe.** At night the histograms are shifted to the dark part: in the frame M3OT 1-02 the median is 36–51 levels against about 110–130 in the day frame, and the peak of the dark background lies at about 30 levels. The range $D$ between $P_1$ and $P_{99}$ is nevertheless not narrower (channel R: 195 levels at night, 165 by day), because street lights and headlights fill the bright end of the histogram: a night frame is described by the shape of its histogram, not by its range. Pixels clipped at 255 lose their information; in this frame they are 0.1 % of the channel R. In the day frame the channel B has a wider range than R (about 195 against 165 levels).

**Control question 2.** Which properties of a frame can be read from the histogram of a channel, and which cannot (for example, where in the frame the dark pixels are)?
""")

code(r"""
#@title 2.3.2 Histograms of the channels (code shown) { run: "auto" }
COMPARE_WITH = "M3OT: visible, dusk and night" #@param ["none", "project set: visible (Lesson 1.4)", "VisDrone: visible, oblique view over a city", "M3OT: visible, dusk and night", "Kust4K: visible", "GOOSE: ground vehicle, front camera", "own frame (block 0.3)"]
COMPARE_NUMBER = 1 #@param {type:"slider", min:1, max:8, step:1}
BINS = 64 #@param {type:"slider", min:8, max:256, step:8}
LOG_SCALE = False #@param {type:"boolean"}
CUMULATIVE = False #@param {type:"boolean"}


def channel_histograms(rgb, bins):
    bgr_ = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return {name: cv2.calcHist([bgr_], [i], None, [bins], [0, 256]).ravel() / (rgb.shape[0] * rgb.shape[1])
            for i, name in enumerate("BGR")}                                   # formula (1), p_j


def channel_table(rgb, label):
    rows = []
    for i, name in enumerate("RGB"):
        c = rgb[..., i]
        p = np.bincount(c.ravel(), minlength=256) / c.size
        p1, p99 = np.percentile(c, [1, 99])
        rows.append(dict(frame=label, channel=name, mean=float(c.mean()), median=float(np.median(c)), P1=p1, P99=p99,
                         range_D=p99 - p1, levels_used=int((p > 0).sum()),                       # formula (2)
                         entropy_bit=float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
                         clipped_share=float(((c == 0) | (c == 255)).mean())))
    return rows


frames = [(FRAME_23["caption"], FRAME_23["img"])]
if COMPARE_WITH != "none":
    fr2 = load_frame(COMPARE_WITH, COMPARE_NUMBER)
    frames.append((fr2["caption"], fr2["img"]))
fig, ax = plt.subplots(len(frames), 2, figsize=(16, 4.3 * len(frames)), squeeze=False, gridspec_kw=dict(width_ratios=[1, 1.5]))
rows = []
centres = (np.arange(BINS) + 0.5) * 256 / BINS
for k, (label, img) in enumerate(frames):
    ax[k, 0].imshow(img)
    ax[k, 0].set_title(label, fontsize=9)
    ax[k, 0].axis("off")
    hist = channel_histograms(img, BINS)
    for name, col in (("R", "red"), ("G", "green"), ("B", "blue")):
        y = np.cumsum(hist[name]) if CUMULATIVE else hist[name]
        ax[k, 1].plot(centres, y, color=col, lw=1.3, label=f"channel {name}")
    ax[k, 1].set_xlim(0, 255)
    if LOG_SCALE and not CUMULATIVE:
        ax[k, 1].set_yscale("log")
    ax[k, 1].set_xlabel("brightness, grey levels")
    ax[k, 1].set_ylabel("cumulative share F" if CUMULATIVE else f"share of pixels p (bin of {256 // BINS} levels)")
    ax[k, 1].grid(alpha=0.3)
    ax[k, 1].legend(fontsize=8)
    rows += channel_table(img, label[:40])
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson23_histograms.png"), dpi=110)
plt.show()
tab = pd.DataFrame(rows).set_index(["frame", "channel"])
tab.to_csv(os.path.join(REPORT_DIR, "lesson23_histograms.csv"))
display(tab.round(3))
""")

md(r"""
## Task 2.3.3. Conversion of a frame between BGR, RGB, grey and HSV

**Purpose.** To convert the frame of task 2.3.1 between the colour spaces used by OpenCV, Matplotlib and neural networks, to check the conversions by the formulas for one pixel and to see the typical errors.

**Formulas.** The grey level (luma) of a pixel is

$$Y = 0.299\,R + 0.587\,G + 0.114\,B. \tag{3}$$

HSV: value $V = \max(R, G, B)$, saturation $S = (V - m)/V$ with $m = \min(R, G, B)$ ($S = 0$ for $V = 0$), hue

$$H = \begin{cases} 60^\circ\,(G - B)/(V - m), & V = R,\\ 120^\circ + 60^\circ\,(B - R)/(V - m), & V = G,\\ 240^\circ + 60^\circ\,(R - G)/(V - m), & V = B,\end{cases} \tag{4}$$

with 360° added to a negative H. OpenCV stores the 8-bit HSV as H/2 (0…179), S and V in 0…255; the conversion `HSV_FULL` stores H·256/360 (0…255); for float32 images H is in degrees (0…360) and S, V in 0…1.

**Parameters.** `PIXEL_X`, `PIXEL_Y` — position of the pixel for the check (share of the width and of the height); `SHOW_ERRORS` — the typical errors.

**Code (shown).** `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`, `COLOR_BGR2GRAY`, `COLOR_BGR2HSV`, `COLOR_BGR2HSV_FULL` and the conversion of a float32 array.

**Expected result.** Panels of the representations; a table of the values of the pixel in every space, computed by OpenCV and by formulas (3)–(4); a table of shapes, types and ranges; with `SHOW_ERRORS` — the measured effect of two errors.

**What to observe.** A BGR array shown by Matplotlib has exchanged red and blue. A conversion code for the wrong order of channels does not raise an error: `COLOR_BGR2GRAY` applied to an RGB array weights blue with 0.299 and red with 0.114; in the default frame this changes the grey level by about 3 levels on average and by more than 30 levels in some pixels. The 8-bit HSV loses precision: the conversion RGB → HSV → RGB changes some pixels by 2–3 levels, because H is stored with a step of 2°.

**Control question 3.** A red vehicle appears blue in the figure of a notebook. Name the cause and the line of code that corrects it. Why does a threshold on the hue written for degrees (for example, 120° for green) fail on an 8-bit HSV array of OpenCV?
""")

code(r"""
#@title 2.3.3 BGR, RGB, grey and HSV (code shown) { run: "auto" }
PIXEL_X = 0.5 #@param {type:"slider", min:0, max:1, step:0.01}
PIXEL_Y = 0.5 #@param {type:"slider", min:0, max:1, step:0.01}
SHOW_ERRORS = True #@param {type:"boolean"}

rgb = FRAME_23["img"]
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)                    # the order of OpenCV
grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)                  # formula (3)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)                    # H = degrees / 2 (0...179), S, V in 0...255
hsv_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL)          # H = degrees * 256 / 360 (0...255)
hsv_float = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)   # H in degrees, S, V in 0...1

fig, ax = plt.subplots(2, 3, figsize=(16, 8.6))
panels = [(bgr, "BGR array shown as RGB (wrong order)", None), (rgb, "RGB (correct order)", None),
          (grey, "grey Y (3)", "gray"), (hsv[..., 0], "H, 8 bit (0...179 = degrees / 2)", "hsv"),
          (hsv[..., 1], "S (0...255)", "gray"), (hsv[..., 2], "V (0...255)", "gray")]
H_, W_ = grey.shape
x, y = int(PIXEL_X * (W_ - 1)), int(PIXEL_Y * (H_ - 1))
for a, (im_, t_, cm) in zip(ax.ravel(), panels):
    a.imshow(im_, cmap=cm, vmin=0 if cm else None, vmax=(179 if cm == "hsv" else 255) if cm else None)
    a.plot(x, y, "+", color="yellow", ms=14, mew=2)
    a.set_title(t_, fontsize=10)
    a.axis("off")
plt.tight_layout()
plt.show()

R, G, B = (float(v) for v in rgb[y, x])
V = max(R, G, B)
m_ = min(R, G, B)
S = 0.0 if V == 0 else (V - m_) / V
if V == m_:
    Hdeg = 0.0
elif V == R:
    Hdeg = 60 * (G - B) / (V - m_)
elif V == G:
    Hdeg = 120 + 60 * (B - R) / (V - m_)
else:
    Hdeg = 240 + 60 * (R - G) / (V - m_)
Hdeg = Hdeg + 360 if Hdeg < 0 else Hdeg                      # formula (4)
rows = [("B, G, R (OpenCV array)", str(tuple(int(v) for v in bgr[y, x])), ""),
        ("R, G, B", str(tuple(int(v) for v in rgb[y, x])), ""),
        ("grey Y", int(grey[y, x]), f"{0.299 * R + 0.587 * G + 0.114 * B:.1f}"),
        ("H, S, V (8 bit)", str(tuple(int(v) for v in hsv[y, x])), f"H/2 = {Hdeg / 2:.1f}, S·255 = {S * 255:.1f}, V = {V:.0f}"),
        ("H (HSV_FULL)", int(hsv_full[y, x, 0]), f"H·256/360 = {Hdeg * 256 / 360:.1f}"),
        ("H, S, V (float32)", f"({hsv_float[y, x, 0]:.1f}°, {hsv_float[y, x, 1]:.3f}, {hsv_float[y, x, 2]:.3f})",
         f"({Hdeg:.1f}°, {S:.3f}, {V / 255:.3f})")]
display(pd.DataFrame(rows, columns=["representation", "OpenCV", "formulas (3)-(4)"]).set_index("representation")
        .style.set_caption(f"Pixel x = {x}, y = {y}"))
spaces = [("BGR", bgr), ("RGB", rgb), ("grey", grey), ("HSV 8 bit", hsv), ("HSV_FULL", hsv_full), ("HSV float32", hsv_float)]
display(pd.DataFrame([dict(space=n, shape=str(a_.shape), dtype=str(a_.dtype), min=float(a_.min()), max=float(a_.max()))
                      for n, a_ in spaces]).set_index("space").round(3))
if SHOW_ERRORS:
    wrong = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)              # error: BGR code applied to an RGB array
    d = np.abs(wrong.astype(np.int16) - grey.astype(np.int16))
    back = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), cv2.COLOR_HSV2RGB)
    e = np.abs(back.astype(np.int16) - rgb.astype(np.int16))
    display(pd.DataFrame({"mean |difference|": [d.mean(), e.mean()], "max |difference|": [d.max(), e.max()],
                          "share of pixels changed by more than 2 levels": [(d > 2).mean(), (e > 2).mean()]},
                         index=["grey of an RGB array with COLOR_BGR2GRAY", "RGB → HSV (8 bit) → RGB"]).round(3))
""")

md(r"""
## Task 2.3.4. Region of interest (ROI) of a frame for an applied task

**Purpose.** To select the part of the frame that is needed for an applied task, to measure how much computation it saves and to check what it loses.

**Variants of the task.**
- *Manual rectangle* on the frame of task 2.3.1 (sliders).
- *Traffic area of an M3OT sequence.* The annotated vehicles of the first part of the sequence (`LEARN_SHARE`) define the rectangle in which vehicles move; the second part of the sequence checks how many vehicle boxes the rectangle keeps. A region of interest obtained from earlier observations is checked on later observations, otherwise its quality is not known.
- *Path area of the ground vehicle (GOOSE).* The lower part of the frame of the front camera (`LOWER_SHARE` of the height); the pixel labels show what share of the traversable surface it contains.
- *Chip around an annotated object* — a small region with a margin around one object, the input of a classifier (Topic 4).

**Code (shown).** A region is a slice of the array: `roi = frame[y0:y1, x0:x1]`. The slice is a *view*: it shares the memory with the frame, and a change of the ROI changes the frame; `roi.copy()` makes an independent array.

**Expected result.** The frame with the region, the region itself, a table: share of the pixels, time of a processing step (Gaussian filter and Canny edges) for the whole frame and for the region, objects or traversable pixels kept. The definition of the region is saved in `project.json`.

**What to observe.** For the M3OT sequences the traffic area occupies from about a fifth of the frame (1-03, a road across the frame) to the whole frame (crossroads 1-05 and 1-08). In the GOOSE frames the lower half contains 78–99 % of the traversable pixels. The region saves computation in proportion to its share of pixels, but objects that appear outside it are lost.

**Control question 4.** Why is a region of interest defined from one part of a sequence checked on another part? Give an example of a task for which a region of interest is dangerous.
""")

code(r"""
#@title 2.3.4 Region of interest (code shown) { run: "auto" }
TASK = "traffic area of an M3OT sequence (from the annotation)" #@param ["manual rectangle on the frame of task 2.3.1", "traffic area of an M3OT sequence (from the annotation)", "path area of the ground vehicle (GOOSE)", "chip around an annotated object"]
SEQUENCE = "1-03" #@param ["1-01", "1-02", "1-03", "1-04", "1-05", "1-06", "1-07", "1-08", "1-09"]
LEARN_SHARE = 0.5 #@param {type:"slider", min:0.1, max:0.9, step:0.1}
X0 = 0.25 #@param {type:"slider", min:0, max:0.9, step:0.05}
Y0 = 0.25 #@param {type:"slider", min:0, max:0.9, step:0.05}
ROI_W = 0.5 #@param {type:"slider", min:0.1, max:1, step:0.05}
ROI_H = 0.5 #@param {type:"slider", min:0.1, max:1, step:0.05}
LOWER_SHARE = 0.5 #@param {type:"slider", min:0.2, max:1, step:0.05}
GOOSE_FRAME = 1 #@param {type:"slider", min:1, max:8, step:1}
OBJECT_NUMBER = 1 #@param {type:"slider", min:1, max:30, step:1}
MARGIN_PX = 16 #@param {type:"slider", min:0, max:64, step:4}


def processing(img):
    return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0), 60, 140)


rows, roi_def = [], {}
if TASK.startswith("manual"):
    frame = FRAME_23["img"]
    H_, W_ = frame.shape[:2]
    x0, y0 = int(X0 * W_), int(Y0 * H_)
    x1, y1 = min(W_, x0 + int(ROI_W * W_)), min(H_, y0 + int(ROI_H * H_))
    roi = frame[y0:y1, x0:x1]                                   # a view of the frame
    rows.append(("the ROI shares the memory with the frame (view)", np.shares_memory(roi, frame)))
    rows.append(("roi.copy() shares the memory", np.shares_memory(roi.copy(), frame)))
    objs = FRAME_23["objects"]
    inside = [o for o in objs if x0 <= (o["box"][0] + o["box"][2]) / 2 < x1 and y0 <= (o["box"][1] + o["box"][3]) / 2 < y1]
    rows.append(("annotated objects in the frame / in the ROI", f"{len(objs)} / {len(inside)}"))
    title = FRAME_23["caption"]
    roi_def = dict(task="manual", box=[x0, y0, x1, y1])
elif TASK.startswith("traffic"):
    tr = m3ot.tracks(SEQUENCE, "rgb")
    frames_ = np.sort(tr.frame.unique())
    split = frames_[int(LEARN_SHARE * (len(frames_) - 1))]
    learn, test = tr[tr.frame <= split], tr[tr.frame > split]
    x0, y0 = int(max(0, learn.x.min())), int(max(0, learn.y.min()))
    x1, y1 = int(min(640, (learn.x + learn.w).max())), int(min(512, (learn.y + learn.h).max()))
    frame = m3ot.frames(SEQUENCE, "rgb", start=int(split) - 1, count=1)[0]
    H_, W_ = frame.shape[:2]
    roi = frame[y0:y1, x0:x1]
    cx, cy = test.x + test.w / 2, test.y + test.h / 2
    kept = float(((cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)).mean()) if len(test) else float("nan")
    rows.append(("frames used to define the ROI / to check it", f"{(frames_ <= split).sum()} / {(frames_ > split).sum()}"))
    rows.append(("vehicle boxes of the check part kept in the ROI", f"{kept:.1%}"))
    title = f"M3OT {SEQUENCE} ({m3ot.SEQUENCES[SEQUENCE]}), frame {int(split)}: traffic area of frames 1...{int(split)}"
    roi_def = dict(task="traffic area", sequence=SEQUENCE, box=[x0, y0, x1, y1], kept_on_check=kept)
elif TASK.startswith("path"):
    img, lab = goose.load(GOOSE_FRAMES[GOOSE_FRAME - 1], scale=0.5)
    frame = img
    H_, W_ = frame.shape[:2]
    x0, x1, y1 = 0, W_, H_
    y0 = int(H_ * (1 - LOWER_SHARE))
    roi = frame[y0:y1, x0:x1]
    shares = []
    for name in GOOSE_FRAMES:
        _, lab_ = goose.load(name, scale=0.5)
        trav = goose.traversability(lab_) == 1
        shares.append(trav[int(lab_.shape[0] * (1 - LOWER_SHARE)):].sum() / max(trav.sum(), 1))
    trav = goose.traversability(lab) == 1
    rows.append(("traversable pixels of this frame inside the ROI", f"{trav[y0:].sum() / max(trav.sum(), 1):.1%}"))
    rows.append(("the same for the 8 GOOSE frames: min ... max", f"{min(shares):.1%} ... {max(shares):.1%}"))
    title = f"GOOSE {GOOSE_FRAMES[GOOSE_FRAME - 1].split('/')[2]}: lower {LOWER_SHARE:.0%} of the frame"
    roi_def = dict(task="path area", lower_share=LOWER_SHARE)
else:
    src = FRAME_23 if FRAME_23["objects"] else load_frame("project set: visible (Lesson 1.4)", 1)
    frame = src["img"]
    H_, W_ = frame.shape[:2]
    o = src["objects"][(OBJECT_NUMBER - 1) % len(src["objects"])]
    bx = [int(round(v)) for v in o["box"]]
    x0, y0 = max(0, bx[0] - MARGIN_PX), max(0, bx[1] - MARGIN_PX)
    x1, y1 = min(W_, bx[2] + MARGIN_PX), min(H_, bx[3] + MARGIN_PX)
    roi = frame[y0:y1, x0:x1].copy()
    rows.append(("object", f"{o['cls']}, box {bx[2] - bx[0]} × {bx[3] - bx[1]} px"))
    rows.append(("chip with the margin", f"{x1 - x0} × {y1 - y0} px"))
    title = src["caption"]
    roi_def = dict(task="chip", box=[x0, y0, x1, y1])

t_full = aitbc.time_call(lambda: processing(frame), repeats=5)
t_roi = aitbc.time_call(lambda: processing(np.ascontiguousarray(roi)), repeats=5)
rows = [("ROI x0, y0, x1, y1, px", f"{x0}, {y0}, {x1}, {y1}"),
        ("share of the pixels of the frame", f"{(x1 - x0) * (y1 - y0) / (W_ * H_):.1%}"),
        ("Gaussian 5×5 + Canny: whole frame / ROI, ms", f"{t_full:.2f} / {t_roi:.2f}")] + rows
shown = frame.copy()
cv2.rectangle(shown, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 3)
fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
ax[0].imshow(shown)
ax[0].set_title(title, fontsize=9)
ax[1].imshow(roi, interpolation="nearest")
ax[1].set_title(f"region of interest {roi.shape[1]} × {roi.shape[0]} px", fontsize=10)
for a in ax:
    a.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson23_roi.png"), dpi=110)
plt.show()
display(pd.DataFrame(rows, columns=["Quantity", "Value"]).set_index("Quantity"))
update_project_json("2.3", {"roi": roi_def})
""")

md(r"""
## Task 2.3.5. Change of the size of a frame and the letterbox transformation

**Purpose.** To implement the letterbox transformation with two OpenCV functions, to check it against the function of the course library, to convert the whole visible frame set of the project to the input of a detector and to write the boxes of the vehicles as labels of a YOLO detector.

**Formulas.** A frame W × H is scaled by one factor for both axes and padded to the input S × S (Notebook 03, block 2.2.5-A):

$$r = \min\!\left(\frac{S}{W}, \frac{S}{H}\right), \qquad W' = \operatorname{round}(rW), \quad H' = \operatorname{round}(rH), \tag{5}$$

$$d_x = \left\lfloor \frac{S - W'}{2} \right\rfloor, \qquad d_y = \left\lfloor \frac{S - H'}{2} \right\rfloor, \tag{6}$$

and a point of the frame goes to $x' = r x + d_x$, $y' = r y + d_y$. With minimal padding $S - W'$ and $S - H'$ are replaced by their remainders after division by the stride 32. A box $[x'_1, y'_1, x'_2, y'_2]$ of the input $W_{in} \times H_{in}$ is written in the label file of a YOLO detector as a line `class cx cy w h` of normalised values:

$$c_x = \frac{x'_1 + x'_2}{2 W_{in}}, \quad c_y = \frac{y'_1 + y'_2}{2 H_{in}}, \quad w = \frac{x'_2 - x'_1}{W_{in}}, \quad h = \frac{y'_2 - y'_1}{H_{in}}. \tag{7}$$

**Parameters.**
- `INPUT_SIZE` — S; `MINIMAL_PADDING` — padding to a multiple of 32 only; `INTERPOLATION` — method of the resampling; `PAD_VALUE` — grey value of the padding.
- `SAVE_SET` — write the converted set in `frames/visible/<set>_lb<S>` with `frames.csv`; `WRITE_YOLO_LABELS` — write the label files of the vehicles in `labels/` of this folder.

**Code (shown).** The function `letterbox_own` uses `cv2.resize` and `cv2.copyMakeBorder`; its result is compared with `aitbc.letterbox` pixel by pixel.

**Expected result.** The comparison of the two implementations, a grid of converted frames with the boxes, a table of the set (scale, padding, number of labels, time per frame), the largest error of the inverse mapping of the boxes, the record in `project.json`.

**What to observe.** For the frames 640 × 512 of the project and S = 640 the scale is r = 1, and the padding is 64 rows above and below; with minimal padding there is no padding at all, because 512 is a multiple of 32. For S = 320 the vehicles of about 10 px become about 5 px. The label files of this task are used for the training of a detector in Topic 5.

**Control question 5.** Calculate r, $d_x$, $d_y$ for a frame 1920 × 1080 and S = 640, and the YOLO line of a box [960, 540, 1000, 600] of the class 0. *(r = 1/3; $d_x$ = 0, $d_y$ = 140; `0 0.5104 0.5156 0.0208 0.0313`.)*
""")

code(r"""
#@title 2.3.5 Size of the frame and letterbox (code shown) { display-mode: "form" }
INPUT_SIZE = "640" #@param ["320", "416", "512", "640", "960"]
MINIMAL_PADDING = False #@param {type:"boolean"}
INTERPOLATION = "area" #@param ["area", "bilinear", "nearest"]
PAD_VALUE = 114 #@param {type:"slider", min:0, max:255, step:1}
SAVE_SET = True #@param {type:"boolean"}
WRITE_YOLO_LABELS = True #@param {type:"boolean"}

INTERP = {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}[INTERPOLATION]
S = int(INPUT_SIZE)


def letterbox_own(img, size, pad_value=114, minimal=False, stride=32, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    r = min(size / h, size / w)                                         # formula (5)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = size - nw, size - nh
    if minimal:
        dw, dh = dw % stride, dh % stride
    left, top = dw // 2, dh // 2                                        # formula (6)
    resized = cv2.resize(img, (nw, nh), interpolation=interp) if (nw, nh) != (w, h) else img
    out = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return out, r, (left, top)


# 1) check of the own implementation against the course library (bilinear interpolation in both)
test = FRAME_23["img"]
mine, r_, pad_ = letterbox_own(test, S, PAD_VALUE, MINIMAL_PADDING, interp=cv2.INTER_LINEAR)
ref, r_ref, pad_ref = aitbc.letterbox(test, S, auto=MINIMAL_PADDING, color=PAD_VALUE)
print(f"Own letterbox: {mine.shape}, r = {r_:.4f}, padding {pad_}; library: {ref.shape}, r = {r_ref:.4f}, padding {pad_ref}; "
      f"largest difference of the pixels: {int(np.abs(mine.astype(int) - ref.astype(int)).max()) if mine.shape == ref.shape else 'shapes differ'}")

# 2) conversion of the visible frame set of the project
set_name = next(iter(SETS["visible"]))
index = SETS["visible"][set_name]
src_dir = os.path.join(PROJECT_ROOT, "frames", "visible", set_name)
out_dir = os.path.join(PROJECT_ROOT, "frames", "visible", f"{set_name}_lb{S}{'m' if MINIMAL_PADDING else ''}")
if SAVE_SET:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
seq = set_name.split("_")[1] if set_name.startswith("m3ot_") else None
tracks = m3ot.tracks(seq, "rgb") if seq else None
rows, grid, errors, times = [], [], [], []
for i, row in enumerate(index.itertuples()):
    img = cv2.cvtColor(cv2.imread(os.path.join(src_dir, row.file)), cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    inp, r, (dx, dy) = letterbox_own(img, S, PAD_VALUE, MINIMAL_PADDING, interp=INTERP)
    times.append((time.perf_counter() - t0) * 1000)
    Hin, Win = inp.shape[:2]
    boxes = []
    if tracks is not None and pd.notna(getattr(row, "frame_in_dataset", np.nan)):
        s0 = img.shape[1] / 640.0
        boxes = [[v * s0 for v in o["box"]] for o in m3ot.objects(tracks, int(row.frame_in_dataset))]
    mapped = [[b[0] * r + dx, b[1] * r + dy, b[2] * r + dx, b[3] * r + dy] for b in boxes]     # x' = r x + d_x
    back = [[(m[0] - dx) / r, (m[1] - dy) / r, (m[2] - dx) / r, (m[3] - dy) / r] for m in mapped]
    if boxes:
        errors.append(float(np.abs(np.array(back) - np.array(boxes)).max()))
    lines = []
    for x1, y1, x2, y2 in mapped:
        x1, x2 = np.clip([x1, x2], 0, Win)
        y1, y2 = np.clip([y1, y2], 0, Hin)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            lines.append(f"0 {(x1 + x2) / (2 * Win):.6f} {(y1 + y2) / (2 * Hin):.6f} {(x2 - x1) / Win:.6f} {(y2 - y1) / Hin:.6f}")   # (7)
    stem = os.path.splitext(row.file)[0]
    if SAVE_SET:
        cv2.imwrite(os.path.join(out_dir, stem + ".png"), cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
        if WRITE_YOLO_LABELS:
            with open(os.path.join(out_dir, "labels", stem + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
    rows.append(dict(file=stem + ".png", source=row.file, input_w=Win, input_h=Hin, r=round(r, 5), pad_x=dx, pad_y=dy,
                     labels=len(lines), short_side_min_px=round(min([min(m[2] - m[0], m[3] - m[1]) for m in mapped]), 1) if mapped else None))
    if i in np.round(np.linspace(0, len(index) - 1, 6)).astype(int):
        grid.append(aitbc.draw_boxes(inp, [dict(cls="vehicle", box=m) for m in mapped], color=(255, 235, 59), labels=False, thickness=1))
tab = pd.DataFrame(rows)
if SAVE_SET:
    tab.to_csv(os.path.join(out_dir, "frames.csv"), index=False)
    with open(os.path.join(out_dir, "classes.txt"), "w") as fh:
        fh.write("vehicle\n")
sheet = aitbc.contact_sheet(grid, cols=3, width=1500)
fig, ax = plt.subplots(figsize=(15, 15 * sheet.shape[0] / sheet.shape[1]))
ax.imshow(sheet)
ax.set_title(f"Input of the detector {tab.input_w.iloc[0]} × {tab.input_h.iloc[0]} ({INTERPOLATION}), vehicles of the annotation", fontsize=10)
ax.axis("off")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"lesson23_letterbox_{S}.png"), dpi=100)
plt.show()
display(pd.DataFrame({"Value": [set_name, len(tab), f"{tab.input_w.iloc[0]} × {tab.input_h.iloc[0]}", tab.r.iloc[0],
                                f"{tab.pad_x.iloc[0]} / {tab.pad_y.iloc[0]}", int(tab.labels.sum()),
                                tab.short_side_min_px.min() if tab.short_side_min_px.notna().any() else "—",
                                f"{max(errors):.1e}" if errors else "—", f"{np.median(times):.2f}",
                                os.path.relpath(out_dir, PROJECT_ROOT) if SAVE_SET else "not saved"]},
                     index=["source set", "frames", "input W_in × H_in", "scale r (5)", "padding d_x / d_y (6)",
                            "labels of vehicles (7)", "smallest short side of a box in the input, px",
                            "largest error of the inverse mapping, px", "time of letterbox per frame, ms", "folder"]))
if SAVE_SET:
    update_project_json("2.3", {"letterbox_set": dict(folder=os.path.relpath(out_dir, PROJECT_ROOT), input_size=S,
                                                      minimal_padding=MINIMAL_PADDING, interpolation=INTERPOLATION,
                                                      frames=len(tab), labels=int(tab.labels.sum()))})
    print_tree(out_dir, max_files=2)
""", form=True)

md(r"""
### Report of Lesson 2.3

Double-click this cell and write the answers.

1. Source of the frame; correlation of the channels; the channel that separates the annotated objects best; answer to control question 1:
2. Range $D$ (2) and number of used levels of the channels of the day and of the night frame; answer to control question 2:
3. Values of the checked pixel in RGB, grey and HSV; effect of the wrong conversion code; answer to control question 3:
4. Chosen applied task, size and share of the region of interest, what it keeps and what it loses; answer to control question 4:
5. Input size, scale r, padding, number of saved frames and labels; answer to control question 5:
""")


# =====================================================================================
md(r"""
---
# Lesson 2.4 (Practical class). Filtering methods for the increase of the visual informativeness of video frames

**Learning questions**
1. Loading of real video frames with regard to the influence of different kinds of noise.
2. Application of the Gaussian Blur and Median Blur filters with a subsequent comparison of their effectiveness.
3. Use of the CLAHE method for a night frame and evaluation of the gain of visual informativeness.
4. Processing of a frame of a thermal camera with the determination of the suitability of a contrast enhancement method.

**Method of the lesson.** The real frames of the data sets contain little noise, because the cameras reduce it before recording (Notebook 03, block 2.2.1-A). Noise of a known kind and level is therefore added to a real frame, and every filter is compared with the clean frame. Contrast enhancement is evaluated on real night and thermal frames without added noise, with the annotated objects of the data sets.

**Result of the lesson.** Tables of the effectiveness of the filters and of the contrast enhancement methods in `outputs/reports`, the chosen parameters of CLAHE and the verdicts of suitability for the thermal frames in `project.json`.
""")

md(r"""
## Task 2.4.1. Real frames and the kinds of noise

**Purpose.** To add noise of the five kinds that occur in the video of UAVs and UGVs to a real frame, to see how each kind looks in the frame, in the difference image and in the profiles, and to check the estimate of the noise level.

**Models of noise.** The clean frame is $I$, the noisy frame $I'$:
- *Gaussian* (noise of the sensor and of the amplifier): $I' = I + n$, $n \sim \mathcal{N}(0, \sigma^2)$ independently in every pixel; (8)
- *Poisson* (photon noise): $I' = \mathrm{Poisson}(k I)/k$ with k photons per grey level, the variance $I/k$ grows with the brightness; (9)
- *impulse* (sparkles of an analog link, defective pixels): the share p of the pixels becomes 0 or 255; (10)
- *column stripes* (fixed-pattern noise of an uncooled thermal sensor between two non-uniformity corrections): $I'(x, y) = I(x, y) + o_x$, $o_x \sim \mathcal{N}(0, \sigma_c^2)$ for every column x; (11)
- *JPEG compression* of a digital link: blocks 8 × 8 and the loss of fine details at a low quality.

**Estimate of the noise level** (Notebook 03, formula (13), Immerkær): $\hat\sigma = \sqrt{\pi/2}\,\frac{1}{6|\Omega|}\sum_{\Omega} |(I' * M)|$ with the mask $M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]$. (12)

**Parameters.** `FRAME`; `NOISE`; the level of every kind: `SIGMA`, `PHOTONS_PER_LEVEL`, `IMPULSE_SHARE`, `STRIPE_SIGMA`, `JPEG_QUALITY`; `SEED` — seed of the random numbers.

**Expected result.** The clean and the noisy frame (central part enlarged), the difference image, the profiles of a row and of the column means, the histogram of the difference; a table with the true standard deviation of the added noise (from the difference with the clean frame) and the estimate (12); a table of the five kinds.

**What to observe.** The estimate (12) is close to the true level for Gaussian and Poisson noise (default frame, σ = 15: estimate 15.2–15.3). It underestimates impulse noise (3 % of the pixels: true level 23, estimate 13) and does not see column stripes and JPEG artefacts. For stripes with the true level 7.7 the estimate (about 2) equals the estimate of the clean frame, because the mask M suppresses changes that are constant along a column. For JPEG of quality 10 (true level about 11) the estimate (0.8) is lower than for the clean frame, because the compression removes the fine noise of the camera together with the fine details. The kind of noise is therefore identified first — from the difference image, the column profile and the histogram — and only then its level is estimated and a filter is chosen. Poisson noise is stronger in bright regions than in shadows: in the night frame M3OT 1-02 with k = 0.5 its standard deviation is 8.4 in the dark pixels (grey < 64) and 16.9 in the bright pixels (grey ≥ 128), while Gaussian noise does not depend on the brightness (default frame: 14.9 in both groups of pixels). A thermal frame has one channel, and the noise is added to its grey levels.

**Control question 6.** Which kind of noise is recognised by a column profile, which by the histogram of the difference, and which by the dependence on the brightness?
""")

code(r"""
#@title 2.4.1 Kinds of noise on real frames (code shown) { run: "auto" }
FRAME = "project set: visible (Lesson 1.4)" #@param ["project set: visible (Lesson 1.4)", "project set: thermal (Lesson 1.4)", "HIT-UAV: night thermal, 15 persons", "M3OT 1-02: night, visible"]
NOISE = "Gaussian (sensor)" #@param ["Gaussian (sensor)", "Poisson (photons)", "impulse (analog link)", "column stripes (thermal sensor)", "JPEG compression (digital link)"]
SIGMA = 15 #@param {type:"slider", min:1, max:50, step:1}
PHOTONS_PER_LEVEL = 0.5 #@param {type:"slider", min:0.05, max:5, step:0.05}
IMPULSE_SHARE = 0.03 #@param {type:"slider", min:0.005, max:0.2, step:0.005}
STRIPE_SIGMA = 8 #@param {type:"slider", min:1, max:30, step:1}
JPEG_QUALITY = 10 #@param {type:"slider", min:2, max:95, step:1}
SEED = 1 #@param {type:"integer"}


def clean_frame_24(name):
    # a thermal frame has one channel: the noise is added to its grey levels, not to three copies of them
    if name.startswith("project set"):
        fr = set_frame("thermal" if "thermal" in name else "visible", 1)
        return dict(fr, img=aitbc.gray(fr["img"])) if fr["channel"] == "thermal" else fr
    if name.startswith("HIT-UAV"):
        img, objs = hit.load(HIT_NIGHT[1])
        return dict(img=img, objects=objs, caption=f"HIT-UAV {HIT_NIGHT[1]} (night, thermal)")
    img = m3ot.frames("1-02", "rgb", start=679, count=1)[0]
    return dict(img=img, objects=m3ot.objects(m3ot.tracks("1-02", "rgb"), 680), caption="M3OT 1-02 (night), visible, frame 680")


NOISE_MAKERS = {
    "Gaussian (sensor)": lambda x: aitbc.add_gaussian_noise(x, SIGMA, seed=SEED),                      # (8)
    "Poisson (photons)": lambda x: aitbc.add_poisson_noise(x, PHOTONS_PER_LEVEL, seed=SEED),           # (9)
    "impulse (analog link)": lambda x: aitbc.add_impulse_noise(x, IMPULSE_SHARE, seed=SEED),           # (10)
    "column stripes (thermal sensor)": lambda x: aitbc.add_column_noise(x, STRIPE_SIGMA, seed=SEED),   # (11)
    "JPEG compression (digital link)": lambda x: aitbc.jpeg(x, JPEG_QUALITY)}
CLEAN_24 = clean_frame_24(FRAME)
clean = CLEAN_24["img"]
noisy = NOISE_MAKERS[NOISE](clean)
diff = noisy.astype(np.float32) - clean.astype(np.float32)
H_, W_ = clean.shape[:2]
ys, xs = slice(H_ // 2 - H_ // 6, H_ // 2 + H_ // 6), slice(W_ // 2 - W_ // 6, W_ // 2 + W_ // 6)

fig = plt.figure(figsize=(18, 8.6))
gs = fig.add_gridspec(2, 3)
d_show = np.clip((diff if diff.ndim == 2 else diff.mean(axis=2)) * 3 + 128, 0, 255).astype(np.uint8)
for k, (im_, t_) in enumerate(((clean, "clean frame (real), central part"), (noisy, f"with noise: {NOISE}"),
                                (d_show, "difference I' − I (×3, grey = 0)"))):
    a = fig.add_subplot(gs[0, k])
    a.imshow(im_[ys, xs], cmap="gray" if im_.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
    a.set_title(t_, fontsize=10)
    a.axis("off")
g0, g1 = aitbc.gray(clean).astype(np.float32), aitbc.gray(noisy).astype(np.float32)
a = fig.add_subplot(gs[1, 0])
a.plot(g1[H_ // 2], lw=0.7, color="C3", label="noisy")
a.plot(g0[H_ // 2], lw=0.9, color="k", label="clean")
a.set_title("Profile of the middle row", fontsize=10)
a.set_xlabel("column x")
a.set_ylabel("grey level")
a.legend(fontsize=8)
a = fig.add_subplot(gs[1, 1])
a.plot((g1 - g0).mean(axis=0), lw=0.8, color="C0")
a.set_title("Mean of the difference over every column", fontsize=10)
a.set_xlabel("column x")
a.set_ylabel("grey levels")
a = fig.add_subplot(gs[1, 2])
a.hist((g1 - g0).ravel(), bins=np.arange(-130, 131, 2), color="C2", log=True)
a.set_title("Histogram of the difference (logarithmic)", fontsize=10)
a.set_xlabel("I' − I, grey levels")
for a in fig.axes[3:]:
    a.grid(alpha=0.3)
fig.suptitle(CLEAN_24["caption"], fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson24_noise.png"), dpi=100)
plt.show()

rows = []
for kind, make in NOISE_MAKERS.items():
    z = make(clean)
    d = z.astype(np.float32) - clean.astype(np.float32)
    rows.append(dict(noise=kind, true_sigma=float(d.std()), estimate_12=aitbc.noise_sigma(z),
                     estimate_12_clean=aitbc.noise_sigma(clean), PSNR_dB=aitbc.psnr(clean, z),
                     changed_pixels=float((np.abs(d) > 0.5).any(axis=-1).mean() if d.ndim == 3 else (np.abs(d) > 0.5).mean())))
display(pd.DataFrame(rows).set_index("noise").round(2))
gd = aitbc.gray(clean)
dark, bright = gd < 64, gd >= 128
parts = [f"{t_} ({m_.mean():.0%} of the pixels): {diff[m_].std():.1f}" for t_, m_ in (("grey < 64", dark), ("grey ≥ 128", bright)) if m_.sum() > 100]
print(f"{NOISE}: standard deviation of I' − I in the dark and in the bright pixels of the frame — " + "; ".join(parts))
""")

md(r"""
## Task 2.4.2. Gaussian Blur and Median Blur: comparison of effectiveness

**Purpose.** To choose a filter for every kind of noise of task 2.4.1, to check the choice by measurements against the clean frame and to compare the effectiveness of the Gaussian and the median filter.

**Measures of effectiveness.**
- PSNR (Notebook 01, formula (4)) and SSIM — closeness of the filtered frame to the clean frame;
- agreement of the edges: the Canny edges of the filtered frame are compared with the edges of the clean frame within 1 px, with the precision P (share of the found edges that are true), the recall R (share of the true edges found) and

$$F_1 = \frac{2 P R}{P + R}; \tag{13}$$

- the time of the filter, ms.

**Filters.** Gaussian and median filters with the kernels 3 × 3, 5 × 5 and 7 × 7 (Notebook 03, formulas (14)–(15)), the correction of column stripes by the column means (`aitbc.column_correction`: the offset of every column is the difference of its mean from the smoothed profile of the column means) and no filter.

**Parameters.** `CHOICE_...` — your choice of a filter for every kind of noise; `CRITERION` — measure by which the choice is checked. The frame and the levels of noise are those of task 2.4.1.

**Expected result.** Heat maps of PSNR and of the edge measure (13) for all pairs noise × filter; the noisy frames and the best filters; a table that compares your choice with the measured best filter. The table is saved in `outputs/reports/lesson24_filters.csv`.

**What to observe.** In the default day frame the Gaussian filter 3 × 3 is best for Gaussian noise with σ = 15 (PSNR about 29.2 dB, F₁ 0.89), and the median filter 3 × 3 is best for impulse noise of 3 % (about 30 dB, F₁ 0.86–0.87). For column stripes the correction of the columns is better than any blur, because it removes the offsets of the columns without smoothing the scene (34.4 dB and F₁ 0.97 against about 28.3 dB and 0.77 of the Gaussian 5 × 5). After JPEG compression of quality 10 no filter restores the details: the Gaussian 3 × 3 adds 0.3 dB of PSNR, and the agreement of the edges is not higher than without a filter, because block artefacts are not independent noise of the pixels. The best kernel depends on the frame. In the thermal frame of HIT-UAV, which contains few fine details, the Gaussian 7 × 7 gives the highest PSNR for Gaussian noise (36.3 dB) and the Gaussian 5 × 5 the best agreement of the edges (0.83 against 0.69 of the 3 × 3); the correction of the columns gives 42.8 dB and F₁ 0.98 against 35.6 dB of the Gaussian 5 × 5. In a visible frame larger kernels destroy the details; the measure of the edges shows it earlier than PSNR.

**Control question 7.** Why does the median filter remove impulses, while the Gaussian filter spreads them into grey spots? Why is neither of them suitable against column stripes?
""")

code(r"""
#@title 2.4.2 Gaussian Blur and Median Blur: your choice and its check (code shown) { run: "auto" }
CHOICE_GAUSSIAN = "Gaussian 3×3" #@param ["no filter", "Gaussian 3×3", "Gaussian 5×5", "Gaussian 7×7", "median 3×3", "median 5×5", "median 7×7", "column correction"]
CHOICE_POISSON = "Gaussian 3×3" #@param ["no filter", "Gaussian 3×3", "Gaussian 5×5", "Gaussian 7×7", "median 3×3", "median 5×5", "median 7×7", "column correction"]
CHOICE_IMPULSE = "median 3×3" #@param ["no filter", "Gaussian 3×3", "Gaussian 5×5", "Gaussian 7×7", "median 3×3", "median 5×5", "median 7×7", "column correction"]
CHOICE_STRIPES = "Gaussian 5×5" #@param ["no filter", "Gaussian 3×3", "Gaussian 5×5", "Gaussian 7×7", "median 3×3", "median 5×5", "median 7×7", "column correction"]
CHOICE_JPEG = "Gaussian 3×3" #@param ["no filter", "Gaussian 3×3", "Gaussian 5×5", "Gaussian 7×7", "median 3×3", "median 5×5", "median 7×7", "column correction"]
CRITERION = "PSNR" #@param ["PSNR", "SSIM", "edge F1 (13)"]

FILTERS = {"no filter": lambda x: x,
           "Gaussian 3×3": lambda x: cv2.GaussianBlur(x, (3, 3), 0), "Gaussian 5×5": lambda x: cv2.GaussianBlur(x, (5, 5), 0),
           "Gaussian 7×7": lambda x: cv2.GaussianBlur(x, (7, 7), 0), "median 3×3": lambda x: cv2.medianBlur(x, 3),
           "median 5×5": lambda x: cv2.medianBlur(x, 5), "median 7×7": lambda x: cv2.medianBlur(x, 7),
           "column correction": lambda x: aitbc.column_correction(x)}
CHOICES = dict(zip(NOISE_MAKERS, (CHOICE_GAUSSIAN, CHOICE_POISSON, CHOICE_IMPULSE, CHOICE_STRIPES, CHOICE_JPEG)))
KEY = {"PSNR": "PSNR_dB", "SSIM": "SSIM", "edge F1 (13)": "edge_F1"}[CRITERION]
clean = CLEAN_24["img"]
rows, best_frames = [], []
for kind, make in NOISE_MAKERS.items():
    noisy = make(clean)
    for name, f in FILTERS.items():
        out = f(noisy)
        rows.append(dict(noise=kind, filter=name, PSNR_dB=aitbc.psnr(clean, out), SSIM=aitbc.ssim(clean, out),
                         edge_F1=aitbc.edge_f1(clean, out)["f1"], time_ms=aitbc.time_call(lambda: f(noisy), repeats=3)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(REPORT_DIR, "lesson24_filters.csv"), index=False)

fig, ax = plt.subplots(1, 2, figsize=(18, 4.4))
for a, (col, ttl, fmt) in zip(ax, (("PSNR_dB", "PSNR to the clean frame, dB", "{:.1f}"), ("edge_F1", "Agreement of the edges F₁ (13)", "{:.2f}"))):
    m = res.pivot(index="noise", columns="filter", values=col).loc[list(NOISE_MAKERS), list(FILTERS)]
    a.imshow(m.values, cmap="viridis", aspect="auto")
    a.set_xticks(range(len(FILTERS)))
    a.set_xticklabels(list(FILTERS), rotation=30, ha="right", fontsize=8)
    a.set_yticks(range(len(NOISE_MAKERS)))
    a.set_yticklabels(list(NOISE_MAKERS), fontsize=8)
    for i in range(m.shape[0]):
        j_best = int(np.nanargmax(m.values[i]))
        for j in range(m.shape[1]):
            a.text(j, i, fmt.format(m.values[i, j]), ha="center", va="center", fontsize=7,
                   color="black" if j == j_best else "white", fontweight="bold" if j == j_best else "normal",
                   bbox=dict(facecolor="yellow", lw=0, pad=1) if j == j_best else None)
    a.set_title(ttl + " (yellow — best for the kind of noise)", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson24_filters_heatmap.png"), dpi=100)
plt.show()

check = []
for kind in NOISE_MAKERS:
    sub = res[res.noise == kind].set_index("filter")
    best = sub[KEY].idxmax()
    mine = CHOICES[kind]
    gap = sub.loc[best, KEY] - sub.loc[mine, KEY]
    tol = 0.3 if KEY == "PSNR_dB" else 0.005
    check.append(dict(noise=kind, your_choice=mine, your_value=sub.loc[mine, KEY], best_filter=best, best_value=sub.loc[best, KEY],
                      verdict="✓ best" if mine == best else ("✓ close to the best" if gap <= tol else "✗ differs from the best")))
    noisy_k = NOISE_MAKERS[kind](clean)
    best_frames.append((kind, noisy_k, FILTERS[best](noisy_k), best))
display(pd.DataFrame(check).set_index("noise").round(3).style.set_caption(f"Your choice, checked by {CRITERION}"))

H_, W_ = clean.shape[:2]
ys, xs = slice(H_ // 2 - H_ // 7, H_ // 2 + H_ // 7), slice(W_ // 2 - W_ // 7, W_ // 2 + W_ // 7)
fig, ax = plt.subplots(2, 5, figsize=(18, 7.4))
for j, (kind, nz, out, best) in enumerate(best_frames):
    ax[0, j].imshow(nz[ys, xs], cmap="gray" if nz.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
    ax[0, j].set_title(kind, fontsize=9)
    ax[1, j].imshow(out[ys, xs], cmap="gray" if out.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
    ax[1, j].set_title(f"best by {CRITERION}: {best}", fontsize=9)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.show()
""")

md(r"""
## Task 2.4.3. CLAHE for a night frame and the gain of visual informativeness

**Purpose.** To apply CLAHE to a real night frame, to measure the gain of visual informativeness and its cost, and to choose the parameters of CLAHE by measurements.

**Measures** (input frame $I$, result $J$):

$$\Delta H = H(J) - H(I), \qquad G_{AG} = \frac{AG(J)}{AG(I)}, \qquad G_n = \frac{\hat\sigma(J)}{\hat\sigma(I)}, \qquad \mathrm{AMBE} = \left|\bar J - \bar I\right|, \tag{14}$$

where H is the entropy, AG the average gradient (Notebook 03), $\hat\sigma$ the estimate (12) and AMBE the absolute change of the mean brightness. In addition: the change of the hue $\Delta h$ of the saturated pixels, deg; the rank correlation ρ (Spearman) of the grey levels of $I$ and $J$ — 1 when the order of brightness of all pixels is kept (any global monotonic mapping), less than 1 when the mapping is local; the median SCR of the annotated objects.

**Methods for comparison.** Linear stretch between the percentiles 0.1 % and 99.9 %, gamma correction $J = 255\,(I/255)^{0.6}$, global histogram equalisation (HE) and CLAHE with the chosen parameters.

**Parameters.** `FRAME`; `CLIP_LIMIT`, `TILES` — parameters of CLAHE (Notebook 03, formula (17)); `DENOISE_FIRST` — filter before CLAHE; `COLOUR` — CLAHE applied to the lightness L* of CIE L\*a\*b\* or to every channel R, G, B separately; `MAX_NOISE_GAIN` — the largest acceptable $G_n$ for the automatic choice of the parameters.

**Expected result.** The frame and the results of the methods; a table of the measures (14); heat maps of $G_{AG}$ and $G_n$ over the clip limit and the number of tiles with the chosen point and with the best point under the condition $G_n \le$ `MAX_NOISE_GAIN`. The result and the parameters are saved in the project.

**What to observe.** In the night frame M3OT 1-02 CLAHE with c = 2 and 8 × 8 tiles gives $\Delta H$ = +0.31 bit and $G_{AG}$ = 1.67 at the cost $G_n$ = 1.99 and ρ = 0.91, and the mean brightness changes by 7 levels. Global HE brightens the whole frame (AMBE 55) with the same noise gain; gamma correction brightens it (AMBE 41) but reduces the average gradient. The gain of informativeness is a gain for the operator: the SCR of the annotated vehicles does not grow. CLAHE on every channel separately changes the hues (mean change of the hue of the saturated pixels 28° against 1.5° for CLAHE on L*).

**Control question 8.** Which parameter of CLAHE limits the increase of the noise? Why is CLAHE applied to L* and not to the channels R, G, B?
""")

code(r"""
#@title 2.4.3 CLAHE for a night frame (code shown) { run: "auto" }
FRAME = "M3OT 1-02: night, visible, 8 vehicles" #@param ["M3OT 1-02: night, visible, 8 vehicles", "M3OT 1-08: dusk, visible, 16 vehicles", "Kust4K 02807N: night, visible, street lights", "VisDrone: night street", "own frame (block 0.3)"]
CLIP_LIMIT = 2.0 #@param {type:"slider", min:1, max:10, step:0.5}
TILES = 8 #@param {type:"slider", min:2, max:16, step:1}
DENOISE_FIRST = "none" #@param ["none", "median 3×3", "Gaussian 3×3"]
COLOUR = "lightness L* (colours kept)" #@param ["lightness L* (colours kept)", "every channel R, G, B separately"]
MAX_NOISE_GAIN = 2.0 #@param {type:"slider", min:1.2, max:4, step:0.1}

if FRAME.startswith("M3OT"):
    seq, f0 = ("1-02", 679) if "1-02" in FRAME else ("1-08", 300)
    img = m3ot.frames(seq, "rgb", start=f0, count=1)[0]
    objs = m3ot.objects(m3ot.tracks(seq, "rgb"), f0 + 1)
    caption = f"M3OT {seq} ({m3ot.SEQUENCES[seq]}), frame {f0 + 1}"
elif FRAME.startswith("Kust4K"):
    p = kust.pair("02807N")
    img, objs, caption = p["rgb"], aitbc.boxes_from_labels(p["labels"], (4, 7), kust.CLASSES, min_area=30), "Kust4K 02807N (night)"
elif FRAME.startswith("VisDrone"):
    img, objs = vis.load(vis.CURATED[2])
    caption = f"VisDrone {vis.CURATED[2]} (night street)"
else:
    fr = load_frame("own", 1)
    img, objs, caption = fr["img"], [], fr["caption"]
masks = [aitbc.box_mask(img.shape, o["box"]) for o in objs]


def clahe(x, c, t):
    if COLOUR.startswith("lightness"):
        return aitbc.equalize(x, "clahe", c, t)                     # L* of CIE L*a*b*
    eq = cv2.createCLAHE(clipLimit=float(c), tileGridSize=(int(t), int(t)))
    return np.dstack([eq.apply(np.ascontiguousarray(x[..., k])) for k in range(3)])


def denoise(x):
    return x if DENOISE_FIRST == "none" else (cv2.medianBlur(x, 3) if DENOISE_FIRST.startswith("median") else cv2.GaussianBlur(x, (3, 3), 0))


def measures(I, J):
    v = [aitbc.signal_to_clutter(J, m) for m in masks]
    v = [s for s in v if np.isfinite(s)]
    return dict(delta_H_bit=aitbc.entropy(J) - aitbc.entropy(I), G_AG=aitbc.average_gradient(J) / aitbc.average_gradient(I),
                G_n=aitbc.noise_sigma(J) / max(aitbc.noise_sigma(I), 1e-6), AMBE=aitbc.ambe(I, J),          # formula (14)
                hue_shift_deg=aitbc.hue_shift(I, J), rho=aitbc.rank_correlation(I, J),
                SCR_objects=float(np.median(v)) if v else np.nan)


methods = {"original": img, "linear stretch 0.1–99.9 %": aitbc.linear_stretch(img, 0.1, 99.9),
           "gamma 0.6": aitbc.gamma_correction(img, 0.6), "global HE": aitbc.equalize(img, "he"),
           f"CLAHE c = {CLIP_LIMIT:g}, {TILES}×{TILES}" + ("" if DENOISE_FIRST == "none" else f", after {DENOISE_FIRST}")
           + ("" if COLOUR.startswith("lightness") else ", R, G, B separately"): clahe(denoise(img), CLIP_LIMIT, TILES)}
tab = pd.DataFrame([dict(method=k, **measures(img, v)) for k, v in methods.items()]).set_index("method")

fig, ax = plt.subplots(1, len(methods), figsize=(19, 3.9))
for a, (k, v) in zip(ax, methods.items()):
    a.imshow(v)
    a.set_title(k, fontsize=9)
    a.axis("off")
fig.suptitle(caption, fontsize=11)
plt.tight_layout()
plt.show()
display(tab.round(3))

CLIPS, TILE_SET = [1, 1.5, 2, 3, 4, 6, 8], [2, 4, 8, 12, 16]
pre = denoise(img)
gag = np.zeros((len(CLIPS), len(TILE_SET)))
gn = np.zeros_like(gag)
for i, c in enumerate(CLIPS):
    for j, t in enumerate(TILE_SET):
        J = clahe(pre, c, t)
        gag[i, j] = aitbc.average_gradient(J) / aitbc.average_gradient(img)
        gn[i, j] = aitbc.noise_sigma(J) / max(aitbc.noise_sigma(img), 1e-6)
ok = gn <= MAX_NOISE_GAIN
bi, bj = np.unravel_index(np.argmax(np.where(ok, gag, -np.inf)), gag.shape) if ok.any() else (0, 0)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
for a, (m, ttl) in zip(ax, ((gag, "gain of the average gradient G_AG"), (gn, "gain of the noise G_n"))):
    im = a.imshow(m, cmap="magma", aspect="auto", origin="lower")
    a.set_xticks(range(len(TILE_SET)))
    a.set_xticklabels([f"{t}×{t}" for t in TILE_SET])
    a.set_yticks(range(len(CLIPS)))
    a.set_yticklabels([f"{c:g}" for c in CLIPS])
    a.set_xlabel("tiles")
    a.set_ylabel("clip limit c")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            a.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if m[i, j] < m.max() * 0.7 else "black")
    a.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="cyan", lw=2.5))
    if CLIP_LIMIT in CLIPS and TILES in TILE_SET:
        a.add_patch(plt.Rectangle((TILE_SET.index(TILES) - 0.5, CLIPS.index(CLIP_LIMIT) - 0.5), 1, 1, fill=False, ec="lime", lw=2, ls="--"))
    a.set_title(ttl, fontsize=10)
    plt.colorbar(im, ax=a, fraction=0.046)
plt.suptitle(f"green dashed — your parameters; cyan — largest G_AG with G_n ≤ {MAX_NOISE_GAIN}: c = {CLIPS[bi]:g}, {TILE_SET[bj]}×{TILE_SET[bj]}",
             fontsize=10)
plt.tight_layout()
plt.show()
chosen = list(methods.values())[-1]
cv2.imwrite(os.path.join(FIG_DIR, "lesson24_clahe_result.png"), cv2.cvtColor(chosen, cv2.COLOR_RGB2BGR))
tab.to_csv(os.path.join(REPORT_DIR, "lesson24_clahe_measures.csv"))
update_project_json("2.4", {"clahe": dict(frame=caption, clip_limit=CLIP_LIMIT, tiles=TILES, denoise_first=DENOISE_FIRST, colour=COLOUR,
                                          measures={k: round(float(v), 3) for k, v in tab.iloc[-1].items() if pd.notna(v)},
                                          best_under_noise_gain=dict(clip_limit=CLIPS[bi], tiles=TILE_SET[bj], max_noise_gain=MAX_NOISE_GAIN))})
""")

md(r"""
## Task 2.4.4. Thermal frame: suitability of a contrast enhancement method

**Purpose.** To determine for a thermal frame whether a contrast enhancement method is suitable for the display to the operator and for automatic processing, by criteria that can be checked by measurements.

**Why a separate check is needed.** The brightness of a thermal frame is a relative temperature: a warm object is found because it is brighter (white-hot) or darker (black-hot) than its background, and a detector compares the brightness of a spot with its surroundings. A method that improves the picture for the eye can therefore make the automatic detection worse.

**Criteria of suitability.**
1. *Order of brightness:* ρ ≥ `MIN_RANK` — a warmer surface remains brighter than a cooler one in the whole frame (formula (14), rank correlation).
2. *Contrast of the objects:* median SCR of the annotated objects after the method ≥ `MIN_SCR_SHARE` × SCR before it.
3. *Detection* (with `EVALUATE_DETECTOR`, applied to a frame of HIT-UAV; for the other frames the recall is shown for comparison): the recall of the detector of warm objects of the course (Notebook 01) on 12 annotated night frames of HIT-UAV (153 persons) decreases by not more than `MAX_RECALL_LOSS`.
4. *Visual informativeness:* $\Delta H > 0$ or $G_{AG} > 1.1$.

A method is **suitable for automatic processing** when criteria 1–3 hold, **suitable for the display to the operator** when criterion 4 holds, otherwise **not suitable**. The gain of the noise $G_n$ is shown, but it is not a criterion: a linear stretch multiplies the noise and the signal by the same factor.

**Parameters.** `FRAME`; `STRETCH_LOW`, `STRETCH_HIGH` — percentiles of the linear stretch (the automatic gain control of a thermal camera works in the same way); `GAMMA`; `CLIP_LIMIT`, `TILES` — CLAHE; `EVALUATE_DETECTOR`; the thresholds of the criteria.

**Expected result.** The frame after every method with the annotated objects; a table of the measures and verdicts; with `EVALUATE_DETECTOR` — recall and precision of the detector for every method. The table is saved in `outputs/reports/lesson24_thermal_suitability.csv`, the verdicts in `project.json`.

**What to observe.** On the 12 night frames of HIT-UAV the detector finds 90 % of the persons in the original frames (recall 0.902). After a linear stretch between 1 % and 99 % it finds 62 %: the persons are among the hottest pixels (in the median frame 32 % of the pixels in their boxes are above the 99th percentile), and the stretch clips them to 255 (in the default frame 43 % of the pixels in the boxes of the persons become 255). With the limits 0.1 % and 99.9 % (change `STRETCH_LOW` and `STRETCH_HIGH`) 6 % of these pixels are clipped, the recall is 0.895, and the stretch is suitable for automatic processing and display. Gamma correction 0.6 gives the recall 0.81, CLAHE (c = 2, 8 × 8) 0.63, global HE 0.37. For the vehicles of M3OT (black-hot) the linear stretch 1–99 % keeps the SCR (1-08: 1.03 of the original, project set 1-04: 1.04), while CLAHE lowers it (0.68 and 0.88). CLAHE is suitable for the display of thermal frames to the operator; for automatic detection the frame is used without local contrast enhancement, or the detector is trained on enhanced frames.

**Control question 9.** Why does a linear stretch between the 1st and the 99th percentile reduce the detection of persons in thermal frames, while the stretch between 0.1 % and 99.9 % does not? Which criterion of suitability does CLAHE fail, and why?
""")

code(r"""
#@title 2.4.4 Thermal frame: suitability of a contrast enhancement method (code shown) { display-mode: "form" }
FRAME = "HIT-UAV: night, 15 persons (white-hot)" #@param ["HIT-UAV: night, 15 persons (white-hot)", "M3OT 1-08: dusk, 16 vehicles (black-hot)", "M3OT 1-02: night, 8 vehicles (black-hot)", "project set: thermal (Lesson 1.4)", "Kust4K 02807N: night (luminance of the palette)"]
STRETCH_LOW = 1.0 #@param {type:"slider", min:0, max:5, step:0.1}
STRETCH_HIGH = 99.0 #@param {type:"slider", min:95, max:100, step:0.1}
GAMMA = 0.6 #@param {type:"slider", min:0.3, max:2, step:0.05}
CLIP_LIMIT = 2.0 #@param {type:"slider", min:1, max:10, step:0.5}
TILES = 8 #@param {type:"slider", min:2, max:16, step:1}
EVALUATE_DETECTOR = True #@param {type:"boolean"}
MIN_RANK = 0.95 #@param {type:"slider", min:0.8, max:1, step:0.01}
MIN_SCR_SHARE = 0.9 #@param {type:"slider", min:0.5, max:1, step:0.05}
MAX_RECALL_LOSS = 0.05 #@param {type:"slider", min:0, max:0.3, step:0.01}
import time

if FRAME.startswith("HIT-UAV"):
    im0, objs = hit.load(HIT_NIGHT[1])
    img, caption = aitbc.to_rgb(im0), f"HIT-UAV {HIT_NIGHT[1]} (night, white-hot)"
    objs = [o for o in objs if o["cls"] == "Person"]
elif FRAME.startswith("M3OT"):
    seq, f0 = ("1-08", 300) if "1-08" in FRAME else ("1-02", 679)
    img = m3ot.frames(seq, "ir", start=f0, count=1)[0]
    objs = m3ot.objects(m3ot.tracks(seq, "ir"), f0 + 1)
    caption = f"M3OT {seq} ({m3ot.SEQUENCES[seq]}), thermal (black-hot), frame {f0 + 1}"
elif FRAME.startswith("project"):
    fr = set_frame("thermal", 1)
    img, objs, caption = fr["img"], fr["objects"], fr["caption"]
else:
    p = kust.pair("02807N")
    img = aitbc.to_rgb(p["tir_gray"])
    objs = aitbc.boxes_from_labels(p["labels"], (4, 7), kust.CLASSES, min_area=30)
    caption = "Kust4K 02807N (night), luminance of the thermal palette"
img = aitbc.gray(img) if img.ndim == 3 else img          # a thermal frame is one channel: the methods work on the grey levels
masks = [aitbc.box_mask(img.shape, o["box"]) for o in objs]
METHODS = {"original": lambda x: x,
           f"linear stretch {STRETCH_LOW:g}–{STRETCH_HIGH:g} %": lambda x: aitbc.linear_stretch(x, STRETCH_LOW, STRETCH_HIGH),
           f"gamma {GAMMA:g}": lambda x: aitbc.gamma_correction(x, GAMMA),
           "global HE": lambda x: aitbc.equalize(x, "he"),
           f"CLAHE c = {CLIP_LIMIT:g}, {TILES}×{TILES}": lambda x: aitbc.equalize(x, "clahe", CLIP_LIMIT, TILES)}


def median_scr(x):
    v = [aitbc.signal_to_clutter(x, m) for m in masks]
    v = [s for s in v if np.isfinite(s)]
    return float(np.median(v)) if v else np.nan


def saturated_share(x):
    # share of the pixels in the boxes of the objects that are at 0 or 255: their differences of temperature are lost
    v = np.concatenate([x[m].ravel() for m in masks]) if masks else np.array([], np.uint8)
    return float(((v == 0) | (v == 255)).mean()) if v.size else np.nan


scr0 = median_scr(img)
rows, outs = [], {}
for name, f in METHODS.items():
    J = f(img)
    outs[name] = J
    rows.append(dict(method=name, rho=aitbc.rank_correlation(img, J), SCR=median_scr(J), SCR_share=median_scr(J) / scr0,
                     delta_H_bit=aitbc.entropy(J) - aitbc.entropy(img), G_AG=aitbc.average_gradient(J) / aitbc.average_gradient(img),
                     G_n=aitbc.noise_sigma(J) / max(aitbc.noise_sigma(img), 1e-6), object_px_saturated=saturated_share(J)))
tab = pd.DataFrame(rows).set_index("method")

if EVALUATE_DETECTOR:
    t0 = time.time()
    counts = {name: [0, 0, 0] for name in METHODS}
    above = []
    for f_ in HIT_NIGHT[:12]:
        im_, ob_ = hit.load(f_)
        pm_ = [aitbc.box_mask(im_.shape, o["box"]) for o in ob_ if o["cls"] == "Person"]
        if pm_:
            above.append(float(np.mean(np.concatenate([im_[m] for m in pm_]) > np.percentile(im_, STRETCH_HIGH))))
        r_ = hit.images.set_index("file").loc[f_]
        gsd = hit.gsd(r_.altitude, r_.angle)
        g_ = aitbc.gray(im_) if im_.ndim == 3 else im_
        for name, f in METHODS.items():
            b_, _ = aitbc.detect_warm_objects(f(g_), gsd, k=5.0)
            m_ = aitbc.match_detections(b_, ob_, ("Person",))
            counts[name][0] += len(m_["tp"])
            counts[name][1] += len(m_["fp"])
            counts[name][2] += len(m_["missed"])
    tab["recall_HIT"] = [counts[n][0] / max(counts[n][0] + counts[n][2], 1) for n in tab.index]
    tab["precision_HIT"] = [counts[n][0] / max(counts[n][0] + counts[n][1], 1) for n in tab.index]
    print(f"Detector of warm objects on 12 night frames of HIT-UAV ({counts['original'][0] + counts['original'][2]} persons): {time.time() - t0:.0f} s")
    print(f"Pixels in the boxes of the persons above the percentile {STRETCH_HIGH:g} % of their frame (clipped to 255 by the linear stretch): "
          f"median {np.median(above):.0%} over the frames, from {min(above):.0%} to {max(above):.0%}")


CRITERION_3 = EVALUATE_DETECTOR and FRAME.startswith("HIT-UAV")
if EVALUATE_DETECTOR and not CRITERION_3:
    print("Criterion 3 is not applied to this frame: the recall is measured on the persons of HIT-UAV and is shown for comparison.")


def verdict(r):
    auto = r.rho >= MIN_RANK and (np.isnan(r.SCR_share) or r.SCR_share >= MIN_SCR_SHARE)
    if CRITERION_3:
        auto = auto and (tab.loc["original", "recall_HIT"] - r.recall_HIT <= MAX_RECALL_LOSS)
    show = r.delta_H_bit > 0 or r.G_AG > 1.1
    return ("automatic processing and display" if auto and show else "automatic processing" if auto
            else "display to the operator" if show else "not suitable")


tab["suitable_for"] = [verdict(r) if n != "original" else "—" for n, r in tab.iterrows()]
fig, ax = plt.subplots(1, len(outs), figsize=(19, 3.9))
for a, (name, J) in zip(ax, outs.items()):
    a.imshow(aitbc.draw_boxes(aitbc.to_rgb(J), objs, color=(0, 220, 255), labels=False, thickness=1))
    a.set_title(f"{name}\n{tab.loc[name, 'suitable_for']}", fontsize=8.5)
    a.axis("off")
fig.suptitle(caption, fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lesson24_thermal_methods.png"), dpi=100)
plt.show()
display(tab.round(3))
tab.to_csv(os.path.join(REPORT_DIR, "lesson24_thermal_suitability.csv"))
update_project_json("2.4", {"thermal_enhancement": dict(frame=caption, verdicts=tab.suitable_for.to_dict(),
                                                        thresholds=dict(min_rank=MIN_RANK, min_scr_share=MIN_SCR_SHARE,
                                                                        max_recall_loss=MAX_RECALL_LOSS))})
""", form=True)

md(r"""
### Report of Lesson 2.4

Double-click this cell and write the answers.

1. Frame, kinds and levels of noise; true level and estimate (12) for every kind; answer to control question 6:
2. Your choice of filters, the measured best filters, the differences; answer to control question 7:
3. Night frame, parameters of CLAHE, measures (14), the point with the largest gain of the average gradient at the acceptable noise; answer to control question 8:
4. Thermal frame, verdicts of suitability of the methods, recall of the detector; answer to control question 9:

---
## Self-check of Topic 2

1. A frame 1280 × 720 of a colour camera is read with `cv2.imread`. Give the shape, the type and the order of the channels of the array, and the shape of the input tensor of a network after the letterbox to 640 × 640.
2. Which preprocessing steps are applied to a night frame of a visible camera, and in which order? Justify the order.
3. Which filter is chosen against the sparkles of an analog video link, against the noise of a sensor at low light, against the stripes of a thermal sensor?
4. Why can a method that increases the visual informativeness of a thermal frame decrease the recall of a detector? How is the suitability of a method checked?
5. Why are the boxes of objects recalculated after the letterbox transformation, and which values are stored in a label file of a YOLO detector?

## Data sources and licences

The course library reads the files directly from the repositories of the authors; the data are not redistributed with the notebook. When images or results are published, the data sets are cited.

1. **M3OT.** Nie Z., Xue L., Fang Z., Ren J., Wei Y., Zheng J. M3OT: A Multi-Drone Multi-Modality dataset for Multi-Object Tracking. *Scientific Data*, 12, 1927 (2025). https://doi.org/10.1038/s41597-025-06204-0 · Data: https://doi.org/10.6084/m9.figshare.28308887 · Licence CC BY 4.0.
2. **HIT-UAV.** Suo J., Wang T., Zhang X., Chen H., Zhou W., Shi W. HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection. *Scientific Data*, 10, 227 (2023). https://doi.org/10.1038/s41597-023-02066-6 · Data: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset · Licence CC BY 4.0.
3. **Kust4K.** Ouyang J., Wang Q., Shang Y., Jin P., Zhong H., Zhou L., Shen T. An RGB-TIR dataset from UAV platform for robust urban traffic scenes semantic segmentation. *Scientific Data*, 12, 1701 (2025). https://doi.org/10.1038/s41597-025-05994-7 · Data: https://doi.org/10.6084/m9.figshare.29476610 · Licence CC BY 4.0.
4. **VisDrone.** Zhu P., Wen L., Du D., Bian X., Fan H., Hu Q., Ling H. Detection and tracking meet drones challenge. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(11), 7380–7399 (2022). Data: https://github.com/VisDrone/VisDrone-Dataset · Terms of use: see the data set page.
5. **GOOSE.** Mortimer P., Hagmanns R., Granero M., Luettel T., Petereit J., Wuensche H.-J. The GOOSE dataset for perception in unstructured environments. *IEEE ICRA*, 2024. https://arxiv.org/abs/2310.16788 · Data: https://goose-dataset.de · Licence CC BY-SA 4.0.

**Next notebook.** Notebook 05 — Topic 3, Lessons 3.1 (lecture), 3.3 and 3.4 (practical): extraction of edges, silhouettes and contours by classical methods; features of silhouettes and thermal signatures in thermal frames; stabilisation of the video stream of an FPV drone.
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
            cell = new_code_cell(text.replace("%%LIB%%", lib).replace("%%SOURCES%%", SOURCES))
            if form:
                cell.metadata["cellView"] = "form"
            nb.cells.append(cell)
    nbformat.validate(nb)
    nbformat.write(nb, path_out)
    return nb


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "AITBC_NB04_Channels_ROI_letterbox_and_filtering.ipynb"
    nb = build(out)
    n_md = sum(c.cell_type == "markdown" for c in nb.cells)
    print(f"written {out}: {len(nb.cells)} cells ({n_md} text, {len(nb.cells) - n_md} code)")
