"""
aitbc - course library of "AI Technologies (Basic Course)" (AIT-BC).

Kruty Heroes Military Institute of Telecommunications and Information Technology,
Kyiv, 2026.

All demonstrations of the course use REAL open data sets recorded by UAVs and
ground vehicles. Only the files that a block needs are downloaded: large archives
are read selectively with HTTP range requests, and every file is cached in
DATA_ROOT (a Google Drive folder keeps the cache between sessions).

Data sets
---------
HIT-UAV   2898 thermal (LWIR) images 640x512 and 5 thermal videos. DJI Matrice
          M210 V2 with a Zenmuse XT2 camera (25 mm lens, field of view 25 x 20 deg),
          flight altitude 60-130 m, camera angle 30-90 deg, day and night. Boxes of
          Person, Car, Bicycle, OtherVehicle, DontCare.
          Suo J. et al. Scientific Data 10, 227 (2023). Licence CC BY 4.0.
Kust4K    4024 pixel-aligned visible (RGB) and thermal (TIR) image pairs 640x512
          from a hexacopter over urban roads, day and night; pixel labels of 8 classes.
          Scientific Data 12, 1701 (2025). Licence CC BY 4.0.
VisDrone  RGB images of drones over cities (VisDrone2019-DET validation part,
          548 images) with boxes of pedestrians, vehicles and other objects.
          Zhu P. et al. IEEE TPAMI 44(11), 7380-7399 (2022).
GOOSE     camera images of a ground vehicle on roads, field tracks, in forests and on
          a training area, with pixel labels of 64 classes (validation part, 962 images).
          IEEE ICRA 2024. Licence CC BY-SA 4.0.
M3OT      synchronised visible and thermal image sequences of two DJI Mavic 3T drones
          (100-120 m, 10 fps, 640x512, day, dusk, night) with vehicle tracks.
          Nie Z. et al. Scientific Data 12, 1927 (2025). Licence CC BY 4.0.

Units: metres, seconds, degrees. Images are NumPy arrays H x W (grey) or H x W x 3 (RGB).
"""

__version__ = "2.1.0"

import csv
import io
import json
import math
import os
import struct
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

DATA_ROOT = os.environ.get("AITBC_DATA") or (
    "/content/aitbc_data" if os.path.isdir("/content") else os.path.abspath("aitbc_data"))


def set_data_root(path):
    """Set the folder of the download cache (for example a folder on Google Drive)."""
    global DATA_ROOT
    DATA_ROOT = os.path.abspath(path)
    os.makedirs(DATA_ROOT, exist_ok=True)
    return DATA_ROOT


def _cache(*parts):
    p = os.path.join(DATA_ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def cache_size():
    """Number of files and size of the download cache, MB."""
    n, s = 0, 0
    for dp, _, fn in os.walk(DATA_ROOT):
        for f in fn:
            n += 1
            s += os.path.getsize(os.path.join(dp, f))
    return n, s / 1e6


# =============================================================================
# 1. Downloading
# =============================================================================
class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects, keep the Range header and upgrade http:// targets to https://."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if newurl.startswith("http://"):
            newurl = "https://" + newurl[len("http://"):]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and req.has_header("Range"):
            new.add_header("Range", req.get_header("Range"))
        return new


_OPENER = urllib.request.build_opener(_HTTPSRedirect)
_OPENER.addheaders = [("User-Agent", "aitbc-course/2.0")]


def _urlopen(url, headers=None, timeout=60, retries=4):
    """Open url with retries; a new Request object is built for every attempt."""
    err = None
    for k in range(retries):
        try:
            return _OPENER.open(urllib.request.Request(url, headers=dict(headers or {})), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404, 416):
                raise
            err = e
        except Exception as e:                      # network error: retry after a pause
            err = e
        time.sleep(1.5 * (k + 1))
    raise err


def fetch(url, path, timeout=120):
    """Download url to path; nothing is done when the file is already in the cache."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with _urlopen(url, timeout=timeout) as r, open(tmp, "wb") as fh:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
    os.replace(tmp, path)
    return path


def fetch_many(jobs, threads=8):
    """jobs: list of (url, path); parallel download, returns the list of paths."""
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(lambda j: fetch(*j), jobs))


class _RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (reads the directory of a remote zip)."""

    def __init__(self, url, chunk=1 << 17, timeout=60):
        self.url, self.chunk, self.timeout = url, chunk, timeout
        r = _urlopen(url, {"Range": "bytes=0-0"}, timeout)
        cr = r.headers.get("Content-Range")
        r.read()
        if r.status != 206 or not cr:
            raise IOError(f"the server of {url} does not support range requests")
        self.size = int(cr.split("/")[-1])
        self.pos, self.buf_start, self.buf = 0, 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (self.buf_start <= self.pos and self.pos + n <= self.buf_start + len(self.buf)):
            start = self.pos
            end = min(self.size, start + max(n, self.chunk)) - 1
            self.buf = _urlopen(self.url, {"Range": f"bytes={start}-{end}"}, self.timeout).read()
            self.buf_start = start
        off = self.pos - self.buf_start
        out = self.buf[off:off + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


class RemoteZip:
    """Selective reading of members of a large remote zip archive: only the central
    directory and the requested members are transferred; members are cached on disk."""

    def __init__(self, url, cache_name):
        import zipfile
        self.url, self.cache_name = url, cache_name
        idx_path = _cache("_index", cache_name + ".json")
        if os.path.exists(idx_path):
            with open(idx_path) as fh:
                self.index = json.load(fh)
        else:
            zf = zipfile.ZipFile(_RangeFile(url))
            self.index = {i.filename: (i.header_offset, i.compress_size, i.file_size, i.compress_type)
                          for i in zf.infolist() if not i.filename.endswith("/")}
            with open(idx_path, "w") as fh:
                json.dump(self.index, fh)

    def namelist(self):
        return list(self.index)

    def _fetch_member(self, name):
        path = _cache(self.cache_name, name)
        if os.path.exists(path):
            return path
        off, csize, usize, ctype = self.index[name]
        raw = _urlopen(self.url, {"Range": f"bytes={off}-{off + 30 + 1024 + csize}"}).read()
        if raw[:4] != b"PK\x03\x04":
            raise IOError(f"bad local header of {name}")
        n, m = struct.unpack("<HH", raw[26:30])
        data = raw[30 + n + m: 30 + n + m + csize]
        if ctype == 8:
            data = zlib.decompressobj(-15).decompress(data)
        elif ctype != 0:
            raise IOError(f"unsupported compression {ctype} in {name}")
        if len(data) != usize:
            raise IOError(f"incomplete member {name}")
        with open(path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(path + ".part", path)
        return path

    def get(self, names, threads=8):
        """Download members (a name or a list of names); returns local path(s)."""
        single = isinstance(names, str)
        names = [names] if single else list(names)
        with ThreadPoolExecutor(threads) as ex:
            paths = list(ex.map(self._fetch_member, names))
        return paths[0] if single else paths


def _rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =============================================================================
# 2. Data sets
# =============================================================================
class HITUAV:
    """HIT-UAV: thermal images and videos of a DJI Matrice M210 V2 with a Zenmuse XT2.

    File name T_HHH_AA_W_NNNNN: T = 0 day / 1 night, HHH = flight altitude, m,
    AA = camera angle, deg (90 = vertical view), W = 0 sunny / 1 rainy.
    Images are 8-bit white-hot (warm = bright) after the automatic gain control
    of the camera: brightness is a relative, not a calibrated, temperature."""

    PALETTE = "white-hot"
    BASE = "https://raw.githubusercontent.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/main/"
    CLASSES = ("Person", "Car", "Bicycle", "OtherVehicle", "DontCare")
    COLORS = {"Person": (0, 229, 255), "Car": (255, 235, 59), "Bicycle": (118, 255, 3),
              "OtherVehicle": (255, 152, 0), "DontCare": (190, 190, 190)}
    CAMERA = dict(model="DJI Zenmuse XT2, uncooled VOx microbolometer", width=640, height=512,
                  focal_mm=25.0, pixel_pitch_um=17.0, hfov_deg=25.0, vfov_deg=20.0, band_um="7.5-13.5")
    EXAMPLES = ('1_60_30_0_00047.jpg', '0_60_60_0_01734.jpg', '1_70_90_0_07776.jpg', '0_100_50_0_08162.jpg', '1_120_90_0_06161.jpg', '0_130_40_0_05603.jpg', '1_90_70_0_04990.jpg', '0_110_80_0_08518.jpg')
    VIDEOS = {"60m-30_1.mov": "60 m, camera angle 30 deg: people on a sports ground",
              "60m-40_3.mov": "60 m, 40 deg: cars on a parking lot",
              "70m-90_1.mov": "70 m, 90 deg (vertical view): people on a square",
              "120m-30_3.mov": "120 m, 30 deg: road along buildings",
              "130m-60_3.mov": "130 m, 60 deg: parking lot and roads"}

    def __init__(self, splits=("train", "val", "test")):
        import pandas as pd
        imgs, objs = [], []
        jobs = [(self.BASE + f"normal_json/annotations/{sp}.json", _cache("hituav", "annotations", sp + ".json"))
                for sp in splits]
        for sp, path in zip(splits, fetch_many(jobs, 3)):
            with open(path) as fh:
                d = json.load(fh)
            cat = {c["id"]: c["name"] for c in d["categories"]}
            id2f = {}
            for im in d["images"]:
                fn = im.get("filename") or im.get("file_name")
                id2f[im["id"]] = fn
                t, h, a, w = fn.split("_")[:4]
                imgs.append(dict(file=fn, split=sp, night=t == "1", altitude=int(h), angle=int(a), rainy=w == "1"))
            for an in d.get("annotation", d.get("annotations", [])):
                x, y, w, h = an["bbox"]
                objs.append(dict(ann_id=an["id"], file=id2f[an["image_id"]], cls=cat[an["category_id"]],
                                 x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h)))
        # oriented boxes [xc, yc, w, h, angle]: true width and length of rotated objects
        rjobs = [(self.BASE + f"rotate_json/annotations/{sp}.json", _cache("hituav", "annotations", "rot_" + sp + ".json"))
                 for sp in splits]
        rot = {}
        for path in fetch_many(rjobs, 3):
            with open(path) as fh:
                d = json.load(fh)
            for an in d.get("annotations", d.get("annotation", [])):
                b = an["bbox"]
                rot[an["id"]] = (min(b[2], b[3]), max(b[2], b[3]), b[4] if len(b) > 4 else 0.0)
        self.images = pd.DataFrame(imgs).sort_values("file").reset_index(drop=True)
        o = pd.DataFrame(objs)
        o["w"], o["h"] = o.x2 - o.x1, o.y2 - o.y1
        o["width_px"] = o.ann_id.map(lambda i: rot.get(i, (np.nan,) * 3)[0])     # oriented box: short side
        o["length_px"] = o.ann_id.map(lambda i: rot.get(i, (np.nan,) * 3)[1])    # oriented box: long side
        self.objects = o.merge(self.images[["file", "night", "altitude", "angle"]], on="file", how="left")
        self._split = dict(zip(self.images.file, self.images.split))
        self._groups = {f: g for f, g in self.objects.groupby("file")}
        counts = self.objects.pivot_table(index="file", columns="cls", values="x1", aggfunc="count", fill_value=0)
        self.images = self.images.merge(counts, left_on="file", right_index=True, how="left").fillna(0)

    def _url(self, f):
        return self.BASE + f"normal_json/{self._split[f]}/{f}"

    def path(self, file):
        return fetch(self._url(file), _cache("hituav", self._split[file], file))

    def prefetch(self, files, threads=8):
        return fetch_many([(self._url(f), _cache("hituav", self._split[f], f)) for f in files], threads)

    def load(self, file):
        """Thermal image (H x W uint8) and the list of annotated objects."""
        img = cv2.imread(self.path(file), cv2.IMREAD_GRAYSCALE)
        g = self._groups.get(file)
        objs = [] if g is None else [dict(cls=r.cls, box=[r.x1, r.y1, r.x2, r.y2]) for r in g.itertuples()]
        return img, objs

    def video(self, name):
        return fetch(self.BASE + "video_sample/" + name, _cache("hituav", "video", name))

    @staticmethod
    def video_geometry(name):
        """Flight altitude, m, and camera angle, deg, from a video name such as '60m-30_1.mov'."""
        alt, rest = name.split("m-")
        return int(alt), int(rest.split("_")[0])

    def gsd(self, altitude, angle=90):
        """Ground sampling distance at the image centre, m/px (formula of Lesson 1.1)."""
        rng = altitude / max(math.sin(math.radians(angle)), 1e-3)
        return 2 * rng * math.tan(math.radians(self.CAMERA["hfov_deg"]) / 2) / self.CAMERA["width"]


class Kust4K:
    """Kust4K: pixel-aligned RGB and thermal (TIR) pairs 640x512 of a hexacopter
    HET-3021C over urban roads, day (id ...D) and night (id ...N), with pixel labels.

    The thermal images are pseudo-colour (palette from purple = cold to yellow = warm);
    tir_gray is their luminance, a relative thermal brightness. The data set contains
    frames with SIMULATED failures (visible channel set to zero, thermal channel
    blurred) made by its authors; they are excluded from the curated lists below."""

    PALETTE = "pseudo-colour (purple = cold, yellow = warm)"
    URL = "https://ndownloader.figshare.com/files/{}"
    FILES = {"rgb": 55979147, "tir": 55979165, "seg": 55979159}
    CLASSES = ("unlabelled", "road", "building", "motorcycle", "car", "truck", "tree", "person", "traffic facilities")
    COLORS = np.array([(0, 0, 0), (150, 150, 150), (229, 57, 53), (255, 152, 0), (255, 235, 59),
                       (236, 64, 122), (46, 160, 67), (0, 229, 255), (30, 136, 229)], np.uint8)
    OBJECTS = (3, 4, 5, 7)
    CAMERA = dict(platform="hexacopter HET-3021C", tir_focal_mm=9.1, tir_pixel_um=12.0,
                  rgb="1920x1080, f = 8 mm, registered to the thermal image and resized to 640x512")
    DAY = ("00720D", "00719D", "00031D", "01944D", "00851D", "00580D", "00990D", "00355D", "02094D", "02293D")
    NIGHT = ("02566N", "02807N", "02921N", "03135N", "03685N", "03895N", "03780N", "04004N", "02867N", "02599N")

    def __init__(self):
        self._z = {}

    def _zip(self, key):
        if key not in self._z:
            self._z[key] = RemoteZip(self.URL.format(self.FILES[key]), "kust4k_" + key)
        return self._z[key]

    def prefetch(self, ids, threads=6):
        names = [i + ".png" for i in ids]
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda k: self._zip(k).get(names, threads), ("rgb", "tir", "seg")))

    def pair(self, fid):
        """dict(id, night, rgb, tir, tir_gray, labels) for a frame id such as '00720D'."""
        self.prefetch([fid])
        name = fid + ".png"
        rgb = _rgb(self._zip("rgb").get(name))
        tir = _rgb(self._zip("tir").get(name))
        lab = cv2.imread(self._zip("seg").get(name), cv2.IMREAD_UNCHANGED)
        return dict(id=fid, night=fid.endswith("N"), rgb=rgb, tir=tir,
                    tir_gray=cv2.cvtColor(tir, cv2.COLOR_RGB2GRAY), labels=lab)


class VisDrone:
    """VisDrone2019-DET validation part: RGB images of drones with object boxes.
    Annotation line: x, y, w, h, score, category, truncation, occlusion."""

    URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
    CLASSES = ("ignored region", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
               "awning-tricycle", "bus", "motor", "others")
    COLORS = {"pedestrian": (0, 229, 255), "people": (0, 176, 255), "bicycle": (118, 255, 3), "car": (255, 235, 59),
              "van": (255, 193, 7), "truck": (255, 152, 0), "tricycle": (186, 104, 200),
              "awning-tricycle": (156, 39, 176), "bus": (244, 67, 54), "motor": (236, 64, 122)}

    CURATED = ('0000213_03920_d_0000243', '0000291_03801_d_0000887', '0000117_00112_d_0000087', '0000001_08414_d_0000013', '0000155_00401_d_0000001', '0000193_01705_d_0000112', '0000295_02400_d_0000033', '0000289_04601_d_0000834', '0000215_00000_d_0000256', '0000327_04001_d_0000731')

    def __init__(self):
        self.z = RemoteZip(self.URL, "visdrone_val")

    def names(self):
        return sorted(n.split("/")[-1][:-4] for n in self.z.namelist() if n.endswith(".jpg"))

    def _members(self, name):
        return [f"VisDrone2019-DET-val/images/{name}.jpg", f"VisDrone2019-DET-val/annotations/{name}.txt"]

    def prefetch(self, names, threads=8):
        self.z.get([m for n in names for m in self._members(n)], threads)

    def load(self, name):
        """RGB image and the list of objects (ignored regions and 'others' are skipped)."""
        pi, pa = self.z.get(self._members(name))
        objs = []
        with open(pa) as fh:
            for line in fh:
                v = line.strip().strip(",").split(",")
                if len(v) < 8:
                    continue
                x, y, w, h, _, c, trunc, occ = (int(float(s)) for s in v[:8])
                if c in (0, 11):
                    continue
                objs.append(dict(cls=self.CLASSES[c], box=[x, y, x + w, y + h], truncation=trunc, occlusion=occ))
        return _rgb(pi), objs


class M3OT:
    """M3OT: synchronised visible (RGB) and thermal (IR) image sequences of two DJI Mavic 3T
    drones flying at 100-120 m (10 frames per second, 640 x 512 px), day, dusk and night, with
    multi-object tracking annotation of vehicles (MOT format: frame, id, x, y, w, h, flag, class,
    confidence). The track identifiers are the same in the RGB and IR sequences of one drone.
    The two channels were aligned by the authors with an affine transform; a residual spatial
    offset of several pixels and a time offset of up to several frames remain (Lesson 1.4).

    The IR frames are 8-bit grey images in the BLACK-HOT palette (warm = dark). The data
    descriptor does not state the palette; it is established from the scene: in all nine
    sequences of drone 1 the road surface is darker than vegetation by day, at dusk and at night,
    while asphalt is, as a rule, the warmer surface (Lesson 1.3, task 1.3.4). Vehicles are
    bright, i.e. cooler than the road. to_white_hot() gives the white-hot convention of HIT-UAV."""

    PALETTE = "black-hot"
    URL = "https://ndownloader.figshare.com/files/52023875"
    CAMERA = dict(platform="DJI Mavic 3T", altitude_m="100-120", fps=10, width=640, height=512,
                  rgb="telephoto camera, f = 29.85 mm (35 mm equivalent 162 mm), frames registered to IR",
                  ir="uncooled thermal camera 640 x 512, f = 9.1 mm (35 mm equivalent 40 mm)")
    SEQUENCES = {"1-01": "day, suburban road", "1-02": "night, city crossroads", "1-03": "day, city road",
                 "1-04": "day, suburban highway", "1-05": "day, crossroads", "1-06": "dusk, highway",
                 "1-07": "day, crossroads", "1-08": "dusk, crossroads", "1-09": "day, city street with parked cars"}

    def __init__(self):
        self.z = RemoteZip(self.URL, "m3ot")
        self._seq = {}
        for n in self.z.namelist():
            parts = n.split("/")        # M3OT/<drone>/<rgb|ir>/<split>/<sequence>/img1/<frame>.PNG
            if len(parts) == 7 and parts[5] == "img1":
                self._seq.setdefault((parts[4].rstrip("T"), parts[2]), []).append(n)
        for k in self._seq:
            self._seq[k].sort()

    def sequences(self):
        return sorted({k[0] for k in self._seq})

    def n_frames(self, seq, channel="rgb"):
        return len(self._seq[(seq, channel)])

    def frame_names(self, seq, channel="rgb", start=0, count=None, step=1):
        names = self._seq[(seq, channel)]
        end = len(names) if count is None else min(len(names), start + count * step)
        return names[start:end:step]

    def prefetch(self, seq, channel="rgb", start=0, count=None, step=1, threads=8):
        return self.z.get(self.frame_names(seq, channel, start, count, step), threads)

    def frames(self, seq, channel="rgb", start=0, count=None, step=1):
        """List of RGB images (H x W x 3) of a sequence; IR frames are grey images in 3 equal channels."""
        return [_rgb(p) for p in self.prefetch(seq, channel, start, count, step)]

    def tracks(self, seq, channel="rgb"):
        """Annotation table: frame (from 1), id, x, y, w, h, cx, cy."""
        import pandas as pd
        names = self._seq[(seq, channel)]
        gt = names[0].rsplit("/img1/", 1)[0] + "/gt/gt.txt"
        g = pd.read_csv(self.z.get(gt), header=None, names=["frame", "id", "x", "y", "w", "h", "flag", "cls", "conf"])
        g["cx"], g["cy"] = g.x + g.w / 2, g.y + g.h / 2
        return g[["frame", "id", "x", "y", "w", "h", "cx", "cy"]]

    def objects(self, table, frame):
        """Objects of one frame in the format of draw_boxes."""
        t = table[table.frame == frame]
        return [dict(cls="vehicle", id=int(r.id), box=[r.x, r.y, r.x + r.w, r.y + r.h]) for r in t.itertuples()]


class GOOSE:
    """GOOSE: front camera of a ground vehicle (2048x1000) with pixel labels of 64 classes."""

    URL = "https://goose-dataset.de/storage/goose_2d_val.zip"
    TRAVERSABLE = ("asphalt", "cobble", "gravel", "soil", "low_grass", "moss", "leaves", "snow",
                   "road_marking", "pedestrian_crossing", "sidewalk", "bikeway")
    CONDITIONAL = ("high_grass", "crops", "debris", "tree_root")
    IGNORE = ("undefined", "sky", "ego_vehicle", "outlier")
    CURATED = (
        'labels/val/2022-07-22_flight/2022-07-22_flight__0078_1658494246011438339_labelids.png',
        'labels/val/2022-07-22_flight/2022-07-22_flight__0158_1658494465090456964_labelids.png',
        'labels/val/2022-08-30_siegertsbrunn_feldwege/2022-08-30_siegertsbrunn_feldwege__0528_1661860582736903436_labelids.png',
        'labels/val/2022-08-30_siegertsbrunn_feldwege/2022-08-30_siegertsbrunn_feldwege__0551_1661860697315889089_labelids.png',
        'labels/val/2022-09-21_garching_uebungsplatz_2/2022-09-21_garching_uebungsplatz_2__0002_1663755194931038715_labelids.png',
        'labels/val/2022-09-21_garching_uebungsplatz_2/2022-09-21_garching_uebungsplatz_2__0038_1663755353192423465_labelids.png',
        'labels/val/2023-03-03_garching_2/2023-03-03_garching_2__0114_1677850404544550420_labelids.png',
        'labels/val/2023-03-03_garching_2/2023-03-03_garching_2__0156_1677850669766430912_labelids.png',
        'labels/val/2022-12-07_aying_hills/2022-12-07_aying_hills__0007_1670420722571435228_labelids.png',
        'labels/val/2022-12-07_aying_hills/2022-12-07_aying_hills__0057_1670421449778652057_labelids.png',
        'labels/val/2022-12-07_aying_hills/2022-12-07_aying_hills__0037_1670421230548919820_labelids.png',
        'labels/val/2023-05-15_neubiberg_rain/2023-05-15_neubiberg_rain__0650_1684157928693659670_labelids.png',
    )

    def __init__(self):
        self.z = RemoteZip(self.URL, "goose_val")
        self.names, self.colors = {}, {}
        with open(self.z.get("goose_label_mapping.csv"), newline="") as fh:
            for row in csv.DictReader(fh):
                k = int(row["label_key"])
                h = row["hex"].lstrip("#")
                self.names[k] = row["class_name"]
                self.colors[k] = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        self.lut = np.zeros((256, 3), np.uint8)
        for k, c in self.colors.items():
            self.lut[k] = c
        self.id_of = {v: k for k, v in self.names.items()}

    def frames(self):
        return sorted(n for n in self.z.namelist() if n.endswith("_labelids.png"))

    @staticmethod
    def _image_member(label_member):
        return label_member.replace("labels/", "images/").replace("_labelids.png", "_windshield_vis.png")

    def prefetch(self, members, threads=6):
        self.z.get([m for lm in members for m in (self._image_member(lm), lm)], threads)

    def load(self, label_member, scale=0.5):
        """RGB image and label map (both resized by `scale`)."""
        pi, pl = self.z.get([self._image_member(label_member), label_member])
        img, lab = _rgb(pi), cv2.imread(pl, cv2.IMREAD_UNCHANGED)
        if scale != 1:
            w, h = int(img.shape[1] * scale), int(img.shape[0] * scale)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        return img, lab

    def traversability(self, lab):
        """0 = ignored (sky, own vehicle), 1 = traversable, 2 = conditionally traversable, 3 = obstacle."""
        lut = np.full(256, 3, np.uint8)
        for group, v in ((self.TRAVERSABLE, 1), (self.CONDITIONAL, 2), (self.IGNORE, 0)):
            for n in group:
                if n in self.id_of:
                    lut[self.id_of[n]] = v
        return lut[lab]


# =============================================================================
# 3. Camera geometry
# =============================================================================
def gsd_nadir(altitude, hfov_deg, width_px):
    """Ground sampling distance of a vertical view, m/px: GSD = 2 H tan(alpha/2) / N."""
    return 2 * altitude * math.tan(math.radians(hfov_deg) / 2) / width_px


def hfov_from_sensor(focal_mm, pixel_um, width_px):
    """Horizontal field of view, deg, from the focal length and the pixel pitch."""
    return math.degrees(2 * math.atan(width_px * pixel_um * 1e-3 / 2 / focal_mm))


# =============================================================================
# 4. Degradations of the video signal (applied to real frames)
# =============================================================================
def fractal_noise(shape, rng, octaves=5, start=3, persistence=0.5):
    """Smooth random field in [0, 1] (sum of up-sampled random grids)."""
    h, w = shape
    out = np.zeros((h, w), np.float32)
    amp, m = 1.0, max(h, w)
    for o in range(octaves):
        n = start * (2 ** o)
        grid = rng.random((max(2, int(round(n * h / m)) + 1), max(2, int(round(n * w / m)) + 1)), dtype=np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        amp *= persistence
    out -= out.min()
    return out / max(float(out.max()), 1e-6)


def _f(img):
    return img.astype(np.float32)


def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def motion_blur(img, length=9.0, angle=0.0):
    """Linear motion blur: the point spread function is a line of `length` px at `angle` deg."""
    if length < 1.5:
        return img.copy()
    size = int(math.ceil(length)) | 1
    k = np.zeros((size * 4, size * 4), np.uint8)
    c = size * 2
    dx = math.cos(math.radians(angle)) * (length - 1) * 2
    dy = -math.sin(math.radians(angle)) * (length - 1) * 2
    cv2.line(k, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 255, 4, cv2.LINE_AA)
    k = cv2.resize(k.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6), borderType=cv2.BORDER_REFLECT)


def defocus(img, sigma=1.5):
    return img.copy() if sigma <= 0 else cv2.GaussianBlur(img, (0, 0), sigma)


def add_noise(img, sigma=10.0, seed=None):
    rng = np.random.default_rng(seed)
    return _u8(_f(img) + rng.normal(0, sigma, img.shape))


def low_light(img, level=0.15, gain_max=8.0, seed=None):
    """Dark scene and automatic gain: light reduced by `level`, photon and read noise added."""
    if level >= 0.999:
        return img.copy()
    rng = np.random.default_rng(seed)
    lin = (_f(img) / 255.0) ** 2.2 * level
    full_well = 3000.0
    noisy = lin + rng.standard_normal(lin.shape).astype(np.float32) * np.sqrt(lin / full_well + (4.0 / full_well) ** 2)
    gain = float(np.clip(0.16 / max(float(lin.mean()), 1e-6), 1.0, gain_max))
    return _u8(np.clip(noisy * gain, 0, 1) ** (1 / 2.2) * 255)


def haze(img, density=0.4, airlight=200.0):
    """Uniform haze: I = J t + A (1 - t), transmission t = 1 - density."""
    t = 1.0 - float(np.clip(density, 0, 0.98))
    return _u8(_f(img) * t + airlight * (1 - t))


def smoke(img, density=0.5, seed=0, color=170.0, opacity=0.92):
    """Smoke: spatially variable veil."""
    if density <= 0:
        return img.copy()
    n = fractal_noise(img.shape[:2], np.random.default_rng(seed))
    m = np.clip((n - (1.0 - density)) * 3.0, 0, 1) * opacity
    if img.ndim == 3:
        m = m[..., None]
    return _u8(_f(img) * (1 - m) + color * m)


def jpeg(img, quality=30):
    """JPEG compression artefacts."""
    if quality >= 100:
        return img.copy()
    x = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    y = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return y if img.ndim == 2 else cv2.cvtColor(y, cv2.COLOR_BGR2RGB)


def downscale(img, factor=0.5):
    """Loss of resolution: down-sample by `factor`, then up-sample back."""
    if factor >= 0.999:
        return img.copy()
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def rolling_shutter(img, amplitude=6.0, cycles=1.5, phase=0.0):
    """Rolling-shutter distortion: every row is shifted by a sinusoid of the row number."""
    if amplitude <= 0:
        return img.copy()
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shift = amplitude * np.sin(2 * np.pi * cycles * yy / h + phase)
    return cv2.remap(img, xx + shift, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def analog_interference(img, strength=0.5, seed=None):
    """Analog FPV video link with a weak or jammed signal: colour loss, snow, line
    noise, horizontal tearing (loss of synchronisation) and a rolling brightness bar."""
    if strength <= 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    h = img.shape[0]
    x = _f(img)
    if x.ndim == 3:
        gray = x.mean(axis=2, keepdims=True)
        x = gray + (x - gray) * (1 - 0.85 * min(strength * 1.3, 1))
    x += rng.normal(0, 45 * strength, x.shape)
    for r in rng.integers(0, h, int(h * 0.06 * strength)):
        x[r] += rng.normal(70, 40) * strength
    if strength > 0.25:
        y0, band = rng.integers(0, h), int(h * 0.25 * strength)
        offs = np.cumsum(rng.normal(0, 3.0 * strength, band))
        for i in range(band):
            r = (y0 + i) % h
            x[r] = np.roll(x[r], int(offs[i] + 30 * strength), axis=0)
    bar = 1.0 + 0.3 * strength * np.sin(2 * np.pi * (np.arange(h, dtype=np.float32)[:, None] / h * 1.3 + rng.random()))
    return _u8(x * (bar if x.ndim == 2 else bar[..., None]))


def packet_loss(img, rate=0.05, block=16, seed=None):
    """Digital link with lost packets: damaged macroblock slices are filled with a smear
    of the row above or with a grey block."""
    if rate <= 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    out = img.copy()
    gh, gw = img.shape[0] // block, img.shape[1] // block
    lost = rng.random((gh, gw)) < rate
    for r in range(gh):
        c = 0
        while c < gw:
            if lost[r, c]:
                run = int(rng.integers(1, max(2, gw // 3)))
                lost[r, c:c + run] = True
                c += run
            c += 1
    for r, c in zip(*np.nonzero(lost)):
        ys, xs = slice(r * block, (r + 1) * block), slice(c * block, (c + 1) * block)
        if r > 0 and rng.random() < 0.6:
            out[ys, xs] = out[r * block - 1:r * block, xs]
        else:
            out[ys, xs] = 128 if img.ndim == 2 else (60, 128, 60)
    return out


def degrade(img, sensor="visible", motion_px=0.0, motion_angle=0.0, defocus_sigma=0.0, light=1.0, noise=0.0,
            haze_density=0.0, smoke_density=0.0, rolling=0.0, scale=1.0, jpeg_quality=100, analog=0.0,
            packet=0.0, seed=0):
    """Chain of degradations in physical order: atmosphere -> optics and motion -> sensor ->
    resolution -> compression -> radio link. For the thermal sensor haze and smoke are
    attenuated (long-wave infrared passes thin smoke and haze) and low light does not apply."""
    x, ir = img, sensor == "thermal"
    if haze_density > 0:
        x = haze(x, haze_density * (0.25 if ir else 1.0), airlight=128 if ir else 200)
    if smoke_density > 0:
        x = smoke(x, smoke_density, seed=seed, color=128 if ir else 170, opacity=0.18 if ir else 0.92)
    if motion_px > 0:
        x = motion_blur(x, motion_px, motion_angle)
    if defocus_sigma > 0:
        x = defocus(x, defocus_sigma)
    if rolling > 0:
        x = rolling_shutter(x, rolling)
    if light < 1.0 and not ir:
        x = low_light(x, light, seed=seed)
    if noise > 0:
        x = add_noise(x, noise, seed=seed + 1)
    if scale < 1.0:
        x = downscale(x, scale)
    if jpeg_quality < 100:
        x = jpeg(x, jpeg_quality)
    if analog > 0:
        x = analog_interference(x, analog, seed=seed + 2)
    if packet > 0:
        x = packet_loss(x, packet, seed=seed + 3)
    return x


# =============================================================================
# 5. Measures, classical detection, tracking
# =============================================================================
def gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def psnr(ref, test):
    """Peak signal-to-noise ratio, dB."""
    mse = float(np.mean((_f(ref) - _f(test)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def ssim(ref, test):
    """Structural similarity index of grey levels (Gaussian window 11 x 11, sigma 1.5)."""
    a, b = _f(gray(ref)), _f(gray(test))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cab = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))).mean())


def sharpness(img):
    """Variance of the Laplacian: a measure of focus (small values = blurred image)."""
    return float(cv2.Laplacian(gray(img), cv2.CV_64F).var())


def to_white_hot(img, palette):
    """Thermal frame in the white-hot convention (warm = bright) as a 3-channel uint8 image.
    palette: "white-hot" (HIT-UAV), "black-hot" (M3OT, inverted here) or a pseudo-colour palette
    whose luminance grows with the temperature (Kust4K: purple = cold, yellow = warm)."""
    g = gray(img)
    if palette.startswith("black"):
        g = 255 - g
    return np.dstack([g, g, g])


def vegetation_mask(rgb, exg_min=0.12, v_min=35, erode_px=15):
    """Vegetation of a visible RGB frame by the excess-green index ExG = (2G - R - B) / (R + G + B).
    The mask is eroded by erode_px so that a residual offset between the visible and the thermal
    channel does not move it onto neighbouring surfaces."""
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    exg = (2 * g - r - b) / (r + g + b + 1e-6)
    m = ((exg > exg_min) & (f.max(axis=2) > v_min)).astype(np.uint8)
    if erode_px > 1:
        m = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8))
    return m.astype(bool)


def surroundings_mask(shape, boxes, scale=1.0, gap=3):
    """Background ring around annotated objects (e.g. the road surface around vehicles): every box
    [x1, y1, x2, y2] is expanded by scale x its longer side; the boxes themselves, expanded by gap px,
    are excluded."""
    ring = np.zeros(shape[:2], np.uint8)
    core = np.zeros(shape[:2], np.uint8)
    for x1, y1, x2, y2 in boxes:
        s = int(round(scale * max(x2 - x1, y2 - y1)))
        cv2.rectangle(ring, (int(x1) - s, int(y1) - s), (int(x2) + s, int(y2) + s), 1, -1)
        cv2.rectangle(core, (int(x1) - gap, int(y1) - gap), (int(x2) + gap, int(y2) + gap), 1, -1)
    return (ring > 0) & (core == 0)


def palette_check(thermal, road, vegetation, margin=15, min_pixels=200):
    """Palette of a thermal frame established from the scene. In a road scene the asphalt is, as a
    rule, warmer than vegetation (by day it absorbs most of the solar radiation, at night it gives off
    the stored heat), therefore a road darker than vegetation indicates the black-hot palette
    (warm = dark) and a brighter road the white-hot convention (warm = bright). A shaded or wet road
    and sunlit tree crowns can reverse the order in single frames: check several frames."""
    g = gray(thermal)
    road = road & ~vegetation
    nr, nv = int(road.sum()), int(vegetation.sum())
    r = float(np.median(g[road])) if nr >= min_pixels else float("nan")
    v = float(np.median(g[vegetation])) if nv >= min_pixels else float("nan")
    if np.isnan(r) or np.isnan(v):
        verdict = "not determined: too few road or vegetation pixels"
    elif r < v - margin:
        verdict = "road darker than vegetation: black-hot (warm = dark)"
    elif r > v + margin:
        verdict = "road brighter than vegetation: white-hot (warm = bright)"
    else:
        verdict = f"difference below {margin} grey levels: not determined"
    return dict(road_median=r, vegetation_median=v, road_pixels=nr, vegetation_pixels=nv, verdict=verdict)


def signal_to_clutter(img, mask, exclude=None, gap=2, ring=8):
    """Signal-to-clutter ratio of an object: |mean of object - mean of background ring| /
    standard deviation of the background ring (grey levels)."""
    g = _f(gray(img))
    m = mask.astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (gap + ring) + 1, 2 * (gap + ring) + 1))
    bg = (cv2.dilate(m, k2) > 0) & (cv2.dilate(m, k1) == 0)
    if exclude is not None:
        bg &= ~exclude
    t, b = g[m > 0], g[bg]
    if t.size < 3 or b.size < 10:
        return float("nan")
    return abs(float(t.mean()) - float(b.mean())) / max(float(b.std()), 1.0)


def box_iou(a, b):
    """Intersection over union of two boxes [x1, y1, x2, y2]."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def boxes_from_labels(labels, class_ids, names, min_area=20):
    """Boxes of connected regions of the given classes in a label map."""
    out = []
    for c in class_ids:
        n, cc, stats, _ = cv2.connectedComponentsWithStats((labels == c).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a >= min_area:
                out.append(dict(cls=names[c], class_id=int(c), box=[float(x), float(y), float(x + w), float(y + h)],
                                area=int(a)))
    return out


def patch_labels(labels, tile, object_ids, names, min_px=10):
    """Patch classification from a label map: every tile receives the object class with
    the largest number of pixels in it, or 'background'."""
    rows, cols = labels.shape[0] // tile, labels.shape[1] // tile
    grid = np.empty((rows, cols), object)
    for r in range(rows):
        for c in range(cols):
            t = labels[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
            n, k = max((int((t == k).sum()), k) for k in object_ids)
            grid[r, c] = names[k] if n >= min_px else "background"
    return grid


def detect_hotspots(img, k=5.0, se_px=21, min_area=4, max_area=5000):
    """Classical detector of small warm objects in a white-hot thermal image: white
    top-hat (local contrast) -> threshold mean + k std -> connected components.
    Returns boxes (N, 4) [x1, y1, x2, y2] and the binary mask."""
    g = cv2.GaussianBlur(gray(img), (3, 3), 0)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(se_px) | 1, int(se_px) | 1))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se).astype(np.float32)
    mask = (th > th.mean() + k * th.std()).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = [[x, y, x + w, y + h] for x, y, w, h, a in stats[1:] if min_area <= a <= max_area]
    return np.array(boxes, float).reshape(-1, 4), mask


def expected_size_px(gsd, size_m=0.45, margin_px=6.0):
    """Expected size of an object in an annotated box, px: size_m / GSD + margin_px.
    Defaults: nominal shoulder width of a person 0.45 m and a margin of 6 px (blur, thermal halo,
    annotation margin). For comparison, a least-squares fit to the HIT-UAV boxes of persons in
    vertical views gives d = 0.38 m and m = 6.6 px; block 1.1.1-B computes the fit for every class."""
    return size_m / gsd + margin_px


def detect_warm_objects(img, gsd, k=6.0, size_m=0.45, margin_px=6.0):
    """Hot-spot detector adapted to the ground sampling distance: the structuring element
    and the admissible blob areas follow the expected size of a person in pixels."""
    s = expected_size_px(gsd, size_m, margin_px)
    return detect_hotspots(img, k=k, se_px=int(max(7, round(2.5 * s))) | 1,
                           min_area=max(3, int((0.3 * s) ** 2)), max_area=int((2.0 * s) ** 2))


def match_detections(pred, objects, target_classes, pad=2.0):
    """A detection is correct when its centre lies inside a (padded) box of an object of a
    target class. Detections inside boxes of other annotated objects are counted apart.
    Returns dict(tp, fp, other, missed, recall, precision)."""
    tgt = [o for o in objects if o["cls"] in target_classes]
    oth = [o for o in objects if o["cls"] not in target_classes]
    used, tp, fp, other = set(), [], [], []
    inside = lambda o, cx, cy: (o["box"][0] - pad <= cx <= o["box"][2] + pad and o["box"][1] - pad <= cy <= o["box"][3] + pad)
    for b in pred:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        hit = next((i for i, o in enumerate(tgt) if i not in used and inside(o, cx, cy)), None)
        if hit is not None:
            used.add(hit)
            tp.append(list(b))
        elif any(inside(o, cx, cy) for o in oth) or any(inside(o, cx, cy) for o in tgt):
            other.append(list(b))
        else:
            fp.append(list(b))
    n = len(tgt)
    return dict(tp=tp, fp=fp, other=other, missed=[o for i, o in enumerate(tgt) if i not in used],
                recall=len(tp) / n if n else float("nan"), precision=len(tp) / max(len(tp) + len(fp), 1))


def global_shift(prev, cur):
    """Translation of the image content from `prev` to `cur` (phase correlation), px."""
    (dx, dy), _ = cv2.phaseCorrelate(_f(gray(prev)), _f(gray(cur)))
    return dx, dy


class NearestNeighbourTracker:
    """Simple multi-object tracker: the detections of a frame are linked to the tracks of
    the previous frame by the smallest distance after compensation of the camera motion.
    A track is closed after `max_missed` frames without a detection."""

    def __init__(self, max_dist=12.0, max_missed=4):
        self.max_dist, self.max_missed = max_dist, max_missed
        self.tracks, self.closed, self.next_id = {}, {}, 1
        self.offset = np.zeros(2)                    # accumulated camera motion, px

    def update(self, centres, shift=(0.0, 0.0), frame=0):
        """centres: (N, 2) detections of the frame; shift: image motion since the previous frame.
        Returns the list of (track id, centre) of the frame."""
        self.offset += np.asarray(shift, float)
        pts = np.asarray(centres, float).reshape(-1, 2) - self.offset      # coordinates of the first frame
        ids = list(self.tracks)
        pairs = []
        for i, tid in enumerate(ids):
            last = self.tracks[tid]["hist"][-1][1]
            for j, p in enumerate(pts):
                d = float(np.hypot(*(p - last)))
                if d <= self.max_dist:
                    pairs.append((d, tid, j))
        pairs.sort()
        used_t, used_d, out = set(), set(), []
        for d, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            self.tracks[tid]["hist"].append((frame, pts[j]))
            self.tracks[tid]["missed"] = 0
            out.append((tid, pts[j] + self.offset))
        for tid in ids:
            if tid not in used_t:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    self.closed[tid] = self.tracks.pop(tid)
        for j, p in enumerate(pts):
            if j not in used_d:
                self.tracks[self.next_id] = dict(hist=[(frame, p)], missed=0)
                out.append((self.next_id, p + self.offset))
                self.next_id += 1
        return out

    def history(self, tid):
        """Trajectory of a track in the coordinates of the current frame."""
        t = self.tracks.get(tid) or self.closed.get(tid)
        return np.array([p for _, p in t["hist"]]) + self.offset


def track_warm_objects(frames, gsd, fps, k=5.0, max_speed_ms=2.5, max_missed=4, min_len=12, min_disp_m=2.0):
    """Detection and tracking of warm objects in a thermal video:
    detect_warm_objects -> camera motion by phase correlation -> NearestNeighbourTracker.
    A track is reported as a moving object when it lasts at least `min_len` frames and its
    start and end points are at least `min_disp_m` apart on the ground.
    Returns (tracker, moving: {id: trajectory (K, 2) in the last frame}, rows: list of
    (frame, track id, x, y), detections per frame)."""
    tr = NearestNeighbourTracker(max_dist=max(5.0, max_speed_ms / gsd / max(fps, 1e-6) + 3.0), max_missed=max_missed)
    prev, rows, n_det = None, [], []
    for i, f in enumerate(frames):
        g = gray(f)
        b, _ = detect_warm_objects(g, gsd, k=k)
        c = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2] if len(b) else np.zeros((0, 2))
        shift = global_shift(prev, g) if prev is not None else (0.0, 0.0)
        for tid, p in tr.update(c, shift, i):
            rows.append((i, tid, float(p[0]), float(p[1])))
        n_det.append(len(b))
        prev = g
    moving = {}
    for tid, t in {**tr.closed, **tr.tracks}.items():
        pts = np.array([p for _, p in t["hist"]])
        if len(pts) >= min_len and float(np.hypot(*(pts[-1] - pts[0]))) * gsd >= min_disp_m:
            moving[tid] = pts + tr.offset
    return tr, moving, rows, n_det


def video_info(path):
    """Properties of a video file read by OpenCV."""
    cap = cv2.VideoCapture(path)
    info = dict(path=path, opened=cap.isOpened(), fps=cap.get(cv2.CAP_PROP_FPS),
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                codec=int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1", "replace"))
    cap.release()
    info["duration_s"] = info["frames"] / info["fps"] if info["fps"] else float("nan")
    info["size_mb"] = os.path.getsize(path) / 1e6 if os.path.exists(path) else float("nan")
    return info


def iter_video(path, every_n=1, max_frames=None, start=0):
    """Generator of (frame number, timestamp s, RGB frame) of every n-th frame of a video:
    frames are decoded one by one, so a long video does not fill the memory."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    k, n_out = start, 0
    try:
        while True:
            ok, f = cap.read()
            if not ok or (max_frames is not None and n_out >= max_frames):
                break
            if (k - start) % every_n == 0:
                yield k, k / fps, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                n_out += 1
            k += 1
    finally:
        cap.release()


def write_video(path, frames, fps, codec="mp4v"):
    """Write RGB or grey frames to a video file; returns the path. Codec 'mp4v' (.mp4) works in
    Colab and in most OpenCV builds; 'MJPG' (.avi) stores every frame as JPEG."""
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), float(fps), (w, h))
    if not wr.isOpened():
        raise IOError(f"OpenCV cannot write the codec {codec}")
    for f in frames:
        f = to_rgb(f)
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()
    return path


def contact_sheet(frames, cols=6, width=1600, labels=None, font=0.5):
    """Grid of frames for a quick visual review of a sequence."""
    n = len(frames)
    rows = int(math.ceil(n / cols))
    tw = width // cols
    th = int(round(tw * frames[0].shape[0] / frames[0].shape[1]))
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        t = cv2.resize(to_rgb(f), (tw, th), interpolation=cv2.INTER_AREA)
        if labels is not None:
            cv2.rectangle(t, (0, 0), (tw, 18), (0, 0, 0), -1)
            cv2.putText(t, str(labels[i]), (4, 13), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, (255, 255, 0), 1, cv2.LINE_AA)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return sheet


def frame_difference(a, b):
    """Mean absolute difference of the grey levels of two frames."""
    return float(np.mean(np.abs(_f(gray(a)) - _f(gray(b)))))


def grad_magnitude(img, sigma=1.5):
    """Normalised gradient magnitude; edges of a visible and a thermal image of the same scene
    are similar even when their brightness is not, so it is used to compare the two channels."""
    g = cv2.GaussianBlur(_f(gray(img)), (0, 0), sigma)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return (m - m.mean()) / (m.std() + 1e-6)


def channel_shift(img_a, img_b):
    """Translation between two images of different sensors (phase correlation of gradient
    magnitudes): (dx, dy, response). The response is low when the images have little common structure."""
    a, b = grad_magnitude(img_a), grad_magnitude(img_b)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


def track_offsets(table_a, table_b, lag=0):
    """Offsets of the centres of the same track (same id) in two annotated channels:
    channel b at frame t is compared with channel a at frame t + lag. Returns arrays dx, dy."""
    bb = table_b[["id", "frame", "cx", "cy"]].copy()
    bb["frame"] = bb["frame"] + lag
    m = bb.merge(table_a[["id", "frame", "cx", "cy"]], on=["id", "frame"], suffixes=("_b", "_a"))
    return (m.cx_b - m.cx_a).values, (m.cy_b - m.cy_a).values


def read_video(path, step=1, max_frames=None, start=0):
    """Frames of a video file (BGR from OpenCV converted to RGB) and the frame rate."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, k = [], 0
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        if k % step == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        k += 1
    cap.release()
    return frames, fps


# =============================================================================
# 6. Visualisation
# =============================================================================
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img


def colorize(labels, colors):
    """Label map -> RGB image with the colour table `colors` (array K x 3)."""
    return np.asarray(colors, np.uint8)[np.clip(labels, 0, len(colors) - 1)]


def overlay(img, color_img, alpha=0.5, mask=None):
    base, col = _f(to_rgb(img)), _f(color_img)
    a = np.full(base.shape[:2], alpha, np.float32) if mask is None else mask.astype(np.float32) * alpha
    return _u8(base * (1 - a[..., None]) + col * a[..., None])


def draw_boxes(img, objects, colors=None, labels=True, scores=None, ids=None, thickness=1, font=0.4, color=None):
    """Draw boxes of objects (dicts with 'box' and 'cls')."""
    out = to_rgb(img).copy()
    for i, o in enumerate(objects):
        x1, y1, x2, y2 = [int(round(v)) for v in o["box"]]
        c = color if color is not None else (colors or {}).get(o["cls"], (255, 255, 0))
        c = tuple(int(v) for v in c)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness, cv2.LINE_AA)
        txt = []
        if labels:
            txt.append(str(o["cls"]))
        if ids is not None:
            txt.append(f"#{ids[i]}")
        if scores is not None:
            txt.append(f"{scores[i]:.2f}")
        if txt:
            s = " ".join(txt)
            (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
            y0 = max(y1 - 3, th + 2)
            cv2.rectangle(out, (x1, y0 - th - 2), (x1 + tw + 2, y0 + 2), c, -1)
            cv2.putText(out, s, (x1 + 1, y0), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def legend(ax, names, colors, loc="lower right", fontsize=7, ncol=1):
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=np.array(c) / 255.0, label=n) for n, c in zip(names, colors)],
              loc=loc, fontsize=fontsize, ncol=ncol, framealpha=0.85)


def setup_plots():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "figure.facecolor": "white"})


def catalogue():
    """Table of the data sets of the course."""
    return [
        ("HIT-UAV", "UAV DJI M210 V2, thermal camera Zenmuse XT2 (640x512, 25 x 20 deg)",
         "2898 thermal images, 5 thermal videos; boxes of persons and vehicles; 60-130 m, day and night",
         "CC BY 4.0", "Suo J. et al., Scientific Data 10, 227 (2023)",
         "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset"),
        ("Kust4K", "UAV hexacopter HET-3021C, visible + thermal cameras",
         "4024 pixel-aligned RGB-thermal pairs 640x512 over urban roads, day and night; pixel labels of 8 classes",
         "CC BY 4.0", "Scientific Data 12, 1701 (2025)", "https://doi.org/10.6084/m9.figshare.29476610"),
        ("VisDrone2019-DET (val)", "camera drones of different types, 14 Chinese cities",
         "548 RGB images; boxes of pedestrians, cars, vans, trucks, buses, motors and others",
         "research use, see the data set site", "Zhu P. et al., IEEE TPAMI 44(11), 7380-7399 (2022)",
         "https://github.com/VisDrone/VisDrone-Dataset"),
        ("GOOSE (val)", "ground vehicle, front RGB camera 2048x1000",
         "962 images of roads, field tracks, forests and a training area; pixel labels of 64 classes",
         "CC BY-SA 4.0", "IEEE ICRA 2024 (arXiv:2310.16788)", "https://goose-dataset.de"),
    ]
