# =====================================================================================
md(r"""
---
## Topic 5. Detection, segmentation and tracking

**Segmentation** assigns a class to every pixel (a mask), **detection** finds objects and outlines them by boxes, **tracking** links the detections of successive frames into trajectories with stable identifiers. In this topic the classical chain *mask → boxes → evaluation against the annotation* is built on thermal frames, then a real vehicle trajectory is filtered with a Kalman filter; the optional blocks use the neural detector YOLO11 and the tracker ByteTrack.

### 5.1. Mask and boxes of warm objects in a thermal frame

**Purpose.** Find warm objects (people, vehicles) in a HIT-UAV thermal frame (white-hot: warm = bright) by two classical methods and evaluate them against the manual annotation.

| Method | Principle |
|---|---|
| Otsu threshold | one global threshold that best separates the histogram into two classes; the mask is cleaned by morphological opening and closing |
| Top-hat + $k\sigma$ | the white top-hat (frame minus its morphological opening) keeps bright details smaller than the structuring element; threshold $\mu + k\sigma$ of the top-hat. The element size follows the expected size of a person in pixels from the ground sampling distance (GSD) |

The GSD at the frame centre is $\mathrm{GSD} = \dfrac{2\,D \tan(\alpha/2)}{N}$, m/px, where $D = h/\sin\beta$ is the distance to the ground at altitude $h$ and camera angle $\beta$, $\alpha$ the horizontal field of view (25°) and $N = 640$ px the frame width. Altitude and angle are coded in the file name of every HIT-UAV image.

A detection is **correct** (TP) when its centre lies inside an annotated box of a target class; a detection inside a box of another class is counted apart ("other"); the remaining detections are false alarms (FP).

**Parameters.** `IMAGE_FILE` — thermal frame; `TARGET` — target classes; `K_STD` — threshold $k$ of the top-hat method; `MORPH_KERNEL`, `MIN_AREA` — cleaning of the Otsu mask; `N_EVAL_IMAGES` — number of day and of night test images for the precision–recall curves.

**What to look at.** The binary masks; TP (green), FP (red), "other" (orange) and missed objects (white); the **precision–recall curves** over $k$ for day and night: at night the background is cooler and warm objects are more contrasted.
""")

code(r"""
#@title 5.1 Mask and boxes of warm objects (HIT-UAV) { display-mode: "form" }
IMAGE_FILE = "1_60_30_0_00047.jpg · night, 60 m, 30°, 18 persons" #@param ["1_60_30_0_00047.jpg · night, 60 m, 30°, 18 persons", "0_60_60_0_01734.jpg · day, 60 m, 60°, 9 persons", "1_70_90_0_07776.jpg · night, 70 m, 90°, persons, car, bicycles", "0_100_50_0_08162.jpg · day, 100 m, 50°, 15 cars", "1_120_90_0_06161.jpg · night, 120 m, 90°, 9 cars", "0_130_40_0_05603.jpg · day, 130 m, 40°, 6 cars", "1_90_70_0_04990.jpg · night, 90 m, 70°, cars and bicycles", "0_110_80_0_08518.jpg · day, 110 m, 80°, bicycles"]
TARGET = "Person" #@param ["Person", "Person + Bicycle", "all classes"]
K_STD = 5.0 #@param {type:"slider", min:1.0, max:12.0, step:0.5}
MORPH_KERNEL = 3 #@param {type:"slider", min:1, max:9, step:2}
MIN_AREA = 6 #@param {type:"slider", min:1, max:100, step:1}
N_EVAL_IMAGES = 30 #@param {type:"slider", min:10, max:100, step:10}

TARGETS = {"Person": ("Person",), "Person + Bicycle": ("Person", "Bicycle"),
           "all classes": ("Person", "Car", "Bicycle", "OtherVehicle")}[TARGET]


def geometry(file):
    _, h_m, angle, _ = file.split("_")[:4]
    return int(h_m), int(angle)


def detect_otsu(img, kernel=MORPH_KERNEL, min_area=MIN_AREA):
    g = cv2.GaussianBlur(img, (3, 3), 0)
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    se = np.ones((kernel, kernel), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, se), cv2.MORPH_CLOSE, se)
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = [[x, y, x + w, y + h] for x, y, w, h, a in st[1:] if a >= min_area]
    return np.array(boxes, float).reshape(-1, 4), mask


def detect_tophat(img, file, k=K_STD):
    h_m, angle = geometry(file)
    return aitbc.detect_warm_objects(img, hit.gsd(h_m, angle), k=k)


file = IMAGE_FILE.split(" ")[0]
thermal, t_objects = hit.load(file)
h_m, angle = geometry(file)
gsd = hit.gsd(h_m, angle)
results, panels = [], []
for name, (boxes, mask) in [("Otsu threshold", detect_otsu(thermal)), (f"Top-hat + {K_STD}σ", detect_tophat(thermal, file))]:
    m = aitbc.match_detections(boxes, t_objects, TARGETS)
    results.append({"Method": name, "Detections": len(boxes), "TP": len(m["tp"]), "FP": len(m["fp"]),
                    "Other class": len(m["other"]), "Missed": len(m["missed"]),
                    "Precision": m["precision"], "Recall": m["recall"]})
    out = aitbc.draw_boxes(thermal, [o for o in t_objects if o["cls"] in TARGETS], color=(255, 255, 255), labels=False)
    for group, color in [("tp", (40, 220, 60)), ("fp", (255, 40, 40)), ("other", (255, 150, 0))]:
        out = aitbc.draw_boxes(out, [dict(cls="", box=b) for b in m[group]], color=color, labels=False, thickness=2)
    panels += [mask, out]
display(pd.DataFrame(results).set_index("Method").round(3))
print(f"{file}: {'night' if file[0] == '1' else 'day'}, altitude {h_m} m, camera angle {angle}°, GSD {gsd * 100:.1f} cm/px, "
      f"expected person size {aitbc.expected_size_px(gsd):.1f} px; annotated: "
      + ", ".join(f"{c} {sum(o['cls'] == c for o in t_objects)}" for c in ("Person", "Car", "Bicycle") if any(o['cls'] == c for o in t_objects)))
show_images(panels, ["Otsu: mask", "Otsu: TP green, FP red, other orange, annotated targets white",
                     f"Top-hat + {K_STD}σ: mask", "Top-hat: TP green, FP red, other orange"],
            "5_1_warm_objects.png", cols=2, width=6.0)

# precision-recall curves of the top-hat detector over k, day and night
tp_pool = hit.images[(hit.images.split == "test") & (hit.images[list(t for t in TARGETS if t in hit.images)].sum(axis=1) > 0)]
eval_files = {}
for night, label in [(False, "day"), (True, "night")]:
    cand = tp_pool[tp_pool.night == night].file.values
    eval_files[label] = list(np.random.default_rng(SEED).choice(cand, size=min(N_EVAL_IMAGES, len(cand)), replace=False))
hit.prefetch(eval_files["day"] + eval_files["night"])
ks = [2, 3, 4, 5, 6, 8, 10, 12]
pr = []
for label, files in eval_files.items():
    loaded = [(f, *hit.load(f)) for f in files]
    for k in ks:
        tp = fp = n_obj = 0
        for f, img, objs in loaded:
            boxes, _ = detect_tophat(img, f, k)
            m = aitbc.match_detections(boxes, objs, TARGETS)
            tp += len(m["tp"]); fp += len(m["fp"]); n_obj += sum(o["cls"] in TARGETS for o in objs)
        pr.append(dict(time=label, k=k, precision=tp / max(tp + fp, 1), recall=tp / max(n_obj, 1), objects=n_obj))
pr = pd.DataFrame(pr)
pr.to_csv(os.path.join(OUT, "warm_objects_pr.csv"), index=False)
fig, ax = plt.subplots(figsize=(7, 4.6))
for (label, g), color in zip(pr.groupby("time"), [COLORS[3], COLORS[0]]):
    ax.plot(g.recall, g.precision, marker="o", color=color, label=f"{label} ({len(eval_files[label])} images, {g.objects.iloc[0]} targets)")
    for r in g.itertuples():
        ax.annotate(f"k={r.k}", (r.recall, r.precision), textcoords="offset points", xytext=(4, 4), fontsize=7, color=color)
ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1.02), ylim=(0, 1.02),
       title=f"Top-hat detector: precision–recall over k, target {TARGET}")
ax.grid(alpha=0.3); ax.legend(loc="lower left")
fig.tight_layout()
save_figure(fig, "5_1_pr_curves.png")
""")

md(r"""
### 5.2. Kalman filter for a real vehicle trajectory

**Purpose.** Track the position of a vehicle in the UAV video when the measurements of a detector are noisy and some of them are missing (the object is occluded, the detector failed, a packet of the video link was lost).

The true trajectory is the **manual annotation** of a vehicle of the M3OT sequence chosen in 1.1 (centre of its box, 10 times per second). The "detector measurements" are produced from it by adding Gaussian noise with $\sigma_z$ and deleting a share of the frames at random.

**Model of constant velocity.** State $\mathbf{x} = (x, y, v_x, v_y)^\top$, transition over $\Delta t$:
$$\mathbf{x}_k = F\,\mathbf{x}_{k-1} + \mathbf{w}_k,\quad F = \begin{pmatrix} 1 & 0 & \Delta t & 0\\ 0 & 1 & 0 & \Delta t\\ 0 & 0 & 1 & 0\\ 0 & 0 & 0 & 1 \end{pmatrix},\quad Q = \sigma_a^2\, G G^\top,\ G = \begin{pmatrix} \Delta t^2/2 & 0\\ 0 & \Delta t^2/2\\ \Delta t & 0\\ 0 & \Delta t \end{pmatrix},$$
measurement $\mathbf{z}_k = (x, y)^\top + \mathbf{v}_k$, $R = \sigma_z^2 I$.

**One step:** *prediction* $\hat{\mathbf{x}}^- = F\hat{\mathbf{x}}$, $P^- = F P F^\top + Q$; *correction* (only when a measurement exists) $K = P^- H^\top (H P^- H^\top + R)^{-1}$, $\hat{\mathbf{x}} = \hat{\mathbf{x}}^- + K(\mathbf{z} - H\hat{\mathbf{x}}^-)$, $P = (I - KH)P^-$. In a gap the filter continues the trajectory by the prediction, and its uncertainty $P$ grows.

**Parameters.** `TRACK_RANK` — vehicle (1 = the longest annotated track of the sequence); `MEAS_SIGMA_PX` — noise of the detector $\sigma_z$, px; `DROPOUT` — share of missing measurements; `ACCEL_STD` — $\sigma_a$, px/s²: how much the filter trusts the model of constant velocity (small = smooth but slow reaction to manoeuvres, large = follows the noise); `N_RUNS` — number of random repetitions for the curves.

**What to look at.** The trajectory over the frame; the error against time with the $2\sigma$ band of the filter (gaps are shaded); the curves RMSE(dropout) and RMSE($\sigma_a$) — the second one has a minimum: this is the tuning of the filter.
""")

code(r"""
#@title 5.2 Kalman filter for a real vehicle trajectory (M3OT) { display-mode: "form" }
TRACK_RANK = 1 #@param {type:"slider", min:1, max:10, step:1}
MEAS_SIGMA_PX = 4.0 #@param {type:"slider", min:0.5, max:15, step:0.5}
DROPOUT = 0.25 #@param {type:"slider", min:0.0, max:0.7, step:0.05}
ACCEL_STD = 20.0 #@param {type:"slider", min:1, max:300, step:1}
N_RUNS = 20 #@param {type:"slider", min:5, max:100, step:5}

lengths = TRACKS.groupby("id").size().sort_values(ascending=False)
track_id = int(lengths.index[min(TRACK_RANK, len(lengths)) - 1])
trk = TRACKS[TRACKS.id == track_id].sort_values("frame")
t_trk = (trk.frame.values - 1) / FPS
truth = trk[["cx", "cy"]].values


def simulate(truth, sigma, dropout, seed):
    g = np.random.default_rng(seed)
    z = truth + g.normal(0, sigma, truth.shape)
    available = g.random(len(truth)) >= dropout
    available[0] = True
    return z, available


def kalman_track(t, z, available, sigma_z, sigma_a):
    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
    P = np.diag([sigma_z ** 2, sigma_z ** 2, 50.0 ** 2, 50.0 ** 2])
    Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    R = np.eye(2) * sigma_z ** 2
    est, var = [x[:2].copy()], [P[0, 0] + P[1, 1]]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        x = F @ x
        P = F @ P @ F.T + sigma_a ** 2 * G @ G.T
        if available[k]:
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ (z[k] - Hm @ x)
            P = (np.eye(4) - K @ Hm) @ P
        est.append(x[:2].copy()); var.append(P[0, 0] + P[1, 1])
    return np.array(est), np.array(var)


def hold_last(z, available):
    out = z.copy()
    for k in range(1, len(z)):
        if not available[k]:
            out[k] = out[k - 1]
    return out


rmse = lambda a, b: float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
z, available = simulate(truth, MEAS_SIGMA_PX, DROPOUT, SEED)
est, var = kalman_track(t_trk, z, available, MEAS_SIGMA_PX, ACCEL_STD)
raw = hold_last(z, available)
err = np.linalg.norm(est - truth, axis=1)
speed = np.linalg.norm(np.diff(truth, axis=0), axis=1) / np.diff(t_trk)
display(pd.DataFrame([
    ("Track", f"id {track_id} of sequence {SEQ}, frames {trk.frame.min()}–{trk.frame.max()} ({len(trk)} positions, {t_trk[-1] - t_trk[0]:.1f} s)"),
    ("Mean image speed of the vehicle", f"{speed.mean():.1f} px/s"),
    ("Missing measurements", f"{(~available).sum()} of {len(available)} ({100 * (~available).mean():.0f} %)"),
    ("RMSE of the measurements (last value held in gaps)", f"{rmse(raw, truth):.2f} px"),
    ("RMSE of the Kalman estimate", f"{rmse(est, truth):.2f} px ({100 * (1 - rmse(est, truth) / rmse(raw, truth)):+.0f} % against the measurements)"),
    ("Share of time inside the 2σ band of the filter", f"{100 * np.mean(err <= 2 * np.sqrt(var)):.0f} %"),
], columns=["Quantity", "Value"]).set_index("Quantity"))

mid_frame = int(trk.frame.values[len(trk) // 2])
background = m3ot.frames(SEQ, CH, start=mid_frame - 1, count=1)[0]
fig = plt.figure(figsize=(15, 8.4))
grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
ax = fig.add_subplot(grid[:, 0])
ax.imshow(background)
ax.plot(truth[:, 0], truth[:, 1], color="w", lw=2.2, label="annotation (truth)")
ax.scatter(z[available, 0], z[available, 1], s=9, color="#ffb000", alpha=0.8, label="noisy measurements")
ax.plot(est[:, 0], est[:, 1], color="#00d0ff", lw=1.6, label="Kalman estimate")
pad = 60
ax.set_xlim(max(truth[:, 0].min() - pad, 0), min(truth[:, 0].max() + pad, background.shape[1]))
ax.set_ylim(min(truth[:, 1].max() + pad, background.shape[0]), max(truth[:, 1].min() - pad, 0))
ax.set_title(f"Vehicle {track_id}, sequence {SEQ}, background frame {mid_frame}"); ax.legend(loc="best"); ax.axis("off")
ax = fig.add_subplot(grid[0, 1])
ax.plot(t_trk, np.linalg.norm(raw - truth, axis=1), color="#ffb000", lw=1, label="measurement error")
ax.plot(t_trk, err, color=COLORS[0], lw=1.6, label="Kalman error")
ax.fill_between(t_trk, 0, 2 * np.sqrt(var), color=COLORS[0], alpha=0.15, label="2σ band of the filter")
for k in np.where(~available)[0]:
    ax.axvspan(t_trk[k] - 0.5 / FPS, t_trk[k] + 0.5 / FPS, color="grey", alpha=0.15, lw=0)
ax.set(xlabel="time, s", ylabel="position error, px", title="Error against time (grey = missing measurements)")
ax.legend(loc="upper right"); ax.grid(alpha=0.3)

drops = np.arange(0, 0.71, 0.1)
accels = np.geomspace(1, 1000, 13)
curve_d = np.array([[(rmse(hold_last(*simulate(truth, MEAS_SIGMA_PX, d, s)), truth),
                      rmse(kalman_track(t_trk, *simulate(truth, MEAS_SIGMA_PX, d, s), MEAS_SIGMA_PX, ACCEL_STD)[0], truth))
                     for s in range(N_RUNS)] for d in drops]).mean(1)
curve_a = np.array([np.mean([rmse(kalman_track(t_trk, *simulate(truth, MEAS_SIGMA_PX, DROPOUT, s), MEAS_SIGMA_PX, a)[0], truth)
                             for s in range(N_RUNS)]) for a in accels])
ax = fig.add_subplot(grid[1, 1])
ax.plot(100 * drops, curve_d[:, 0], marker="o", color="#ffb000", label="measurements, last value held")
ax.plot(100 * drops, curve_d[:, 1], marker="s", color=COLORS[0], label=f"Kalman, σa = {ACCEL_STD:g} px/s²")
ax.set(xlabel="missing measurements, %", ylabel=f"RMSE, px (mean of {N_RUNS} runs)", title="Robustness to missing measurements")
ax.grid(alpha=0.3); ax.legend(loc="upper left")
inset = ax.inset_axes([0.62, 0.12, 0.35, 0.42])
inset.semilogx(accels, curve_a, color=COLORS[1], marker=".")
inset.axvline(ACCEL_STD, color="grey", ls=":")
best_a = accels[np.argmin(curve_a)]
inset.set_title(f"RMSE against σa (min at {best_a:.0f})", fontsize=8); inset.tick_params(labelsize=7)
fig.tight_layout()
save_figure(fig, "5_2_kalman.png")
TRACKING_RESULT = dict(sequence=SEQ, track_id=track_id, positions=len(trk), meas_sigma_px=MEAS_SIGMA_PX, dropout=DROPOUT,
                       accel_std=ACCEL_STD, rmse_measurements_px=rmse(raw, truth), rmse_kalman_px=rmse(est, truth),
                       best_accel_std=float(best_a))
""")

md(r"""
### 5.3. YOLO11: neural detection in a drone frame (optional block)

**Purpose.** Run the pre-trained detector YOLO11 (trained on the COCO data set of ordinary photographs) on the VisDrone frame from 2.1 and evaluate it against the manual annotation.

The COCO classes are mapped to the VisDrone classes: *person* → pedestrian and people; *car* → car and van; *truck*, *bus* → truck and bus; *bicycle*, *motorcycle* → bicycle and motor. A detection is correct when its intersection over union with an unmatched annotated box of the same group is at least `IOU_THRESHOLD`:
$$\mathrm{IoU} = \frac{|A \cap B|}{|A \cup B|}.$$

**Parameters.** `RUN_YOLO`; `YOLO_MODEL` — `n` (2.6 M parameters, fast) or `s` (9.4 M, more accurate); `CONFIDENCE` — minimum confidence of a detection; `IMAGE_SIZE` — input size of the network (letterbox, block 2.3); `IOU_THRESHOLD`; `SWEEP_IMAGE_SIZE` — repeat the evaluation for several input sizes.

**What to look at.** The table per class group; missed objects (yellow): mostly small and dense objects; the curve recall($S$): an input size that fits the small objects improves the recall without any training. The **confidence of the model is not its accuracy**: the accuracy is known only from the comparison with an annotation.
""")

code(r"""
#@title 5.3 YOLO11 on the drone frame (optional) { display-mode: "form" }
RUN_YOLO = False #@param {type:"boolean"}
YOLO_MODEL = "yolo11n.pt" #@param ["yolo11n.pt", "yolo11s.pt"]
CONFIDENCE = 0.25 #@param {type:"slider", min:0.05, max:0.9, step:0.05}
IMAGE_SIZE = 1280 #@param [640, 960, 1280, 1600] {type:"raw"}
IOU_THRESHOLD = 0.4 #@param {type:"slider", min:0.2, max:0.8, step:0.05}
SWEEP_IMAGE_SIZE = True #@param {type:"boolean"}

YOLO_RESULT = None
GROUPS_GT = {"persons": ("pedestrian", "people"), "cars": ("car", "van"), "trucks and buses": ("truck", "bus"),
             "two-wheelers": ("bicycle", "motor")}
GROUPS_COCO = {"persons": ("person",), "cars": ("car",), "trucks and buses": ("truck", "bus"),
               "two-wheelers": ("bicycle", "motorcycle")}


def evaluate_detections(dets, objects, iou_min=IOU_THRESHOLD):
    rows, tp_boxes, fp_boxes, missed = [], [], [], []
    for group in GROUPS_GT:
        gt = [o["box"] for o in objects if o["cls"] in GROUPS_GT[group]]
        pr = sorted([d for d in dets if d["cls"] in GROUPS_COCO[group]], key=lambda d: -d["conf"])
        used, tp = set(), 0
        for d in pr:
            ious = [aitbc.box_iou(d["box"], g) if j not in used else 0.0 for j, g in enumerate(gt)]
            j = int(np.argmax(ious)) if ious else -1
            if j >= 0 and ious[j] >= iou_min:
                used.add(j); tp += 1; tp_boxes.append(d["box"])
            else:
                fp_boxes.append(d["box"])
        missed += [g for j, g in enumerate(gt) if j not in used]
        rows.append({"Group": group, "Annotated": len(gt), "Detections": len(pr), "TP": tp,
                     "Precision": tp / len(pr) if pr else np.nan, "Recall": tp / len(gt) if gt else np.nan})
    return pd.DataFrame(rows).set_index("Group"), tp_boxes, fp_boxes, missed


if RUN_YOLO:
    if importlib.util.find_spec("ultralytics") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    from ultralytics import YOLO
    detector = YOLO(YOLO_MODEL)
    frame_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)

    def run_yolo(size):
        t0 = time.time()
        r = detector.predict(frame_bgr, imgsz=size, conf=CONFIDENCE, verbose=False)[0]
        dt = time.time() - t0
        dets = [dict(cls=r.names[int(c)], conf=float(p), box=[float(v) for v in b])
                for b, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        return dets, dt

    detector.predict(frame_bgr, imgsz=IMAGE_SIZE, conf=CONFIDENCE, verbose=False)        # warm-up
    dets, dt = run_yolo(IMAGE_SIZE)
    table, tp_b, fp_b, missed_b = evaluate_detections(dets, full_objects)
    total = table[["Annotated", "Detections", "TP"]].sum()
    table.loc["all groups"] = [total.Annotated, total.Detections, total.TP, total.TP / max(total.Detections, 1), total.TP / max(total.Annotated, 1)]
    display(table.round(3))
    table.to_csv(os.path.join(OUT, "yolo_evaluation.csv"))
    out = aitbc.draw_boxes(full_rgb, [dict(cls="", box=b) for b in missed_b], color=(255, 230, 0), labels=False)
    out = aitbc.draw_boxes(out, [dict(cls="", box=b) for b in fp_b], color=(255, 40, 40), labels=False, thickness=2)
    out = aitbc.draw_boxes(out, [dict(cls="", box=b) for b in tp_b], color=(40, 220, 60), labels=False, thickness=2)
    show_images([out], [f"{YOLO_MODEL}, S = {IMAGE_SIZE}, conf ≥ {CONFIDENCE}: TP green, FP red, missed yellow "
                        f"({dt * 1000:.0f} ms per frame)"], "5_3_yolo.png", width=14)
    YOLO_RESULT = dict(model=YOLO_MODEL, image=IMAGE_NAME, image_size=IMAGE_SIZE, confidence=CONFIDENCE,
                       precision=float(table.loc["all groups", "Precision"]), recall=float(table.loc["all groups", "Recall"]))
    if SWEEP_IMAGE_SIZE:
        sweep_rows = []
        for size in [640, 960, 1280, 1600, 1920]:
            d, dt_s = run_yolo(size)
            tb, *_ = evaluate_detections(d, full_objects)
            s = tb[["Annotated", "Detections", "TP"]].sum()
            sweep_rows.append(dict(size=size, recall=s.TP / max(s.Annotated, 1), precision=s.TP / max(s.Detections, 1), ms=1000 * dt_s))
        sweep_rows = pd.DataFrame(sweep_rows)
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.plot(sweep_rows["size"], sweep_rows.recall, marker="o", color=COLORS[0], label="recall")
        ax.plot(sweep_rows["size"], sweep_rows.precision, marker="s", color=COLORS[1], label="precision")
        ax.set(xlabel="input size S, px", ylabel="share", ylim=(0, 1), title=f"{YOLO_MODEL} on the drone frame against the input size")
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
        ax2.bar(sweep_rows["size"], sweep_rows.ms, width=60, color="grey", alpha=0.3)
        ax2.set_ylabel("time per frame, ms")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        fig.tight_layout()
        save_figure(fig, "5_3_yolo_input_size.png")
else:
    print("The YOLO block is switched off. Set RUN_YOLO in the form to run it.")
""")

md(r"""
### 5.4. ByteTrack: multi-object tracking in the UAV video (optional block)

**Purpose.** Detect vehicles with YOLO11 in every frame of a real M3OT video and link them into tracks with ByteTrack. ByteTrack predicts every track with a Kalman filter (as in 5.2) and matches the predictions with the detections in two stages: first with confident detections, then with weak ones — this keeps the tracks of partly occluded objects.

**Parameters.** `RUN_BYTETRACK`; `TRACK_SEQUENCE` — visible sequence of M3OT; `TRACK_MODEL`; `TRACK_CONFIDENCE`; `TRACK_IMAGE_SIZE` — vehicles at 100–120 m occupy 10–25 px, therefore the frame (640 × 512) is enlarged to 1920 px at the input of the network; `MAX_TRACK_FRAMES`. On a CPU YOLO11s at 1920 px takes about 1 s per frame, on a T4 GPU about 0.05 s.

The detector was trained on COCO photographs taken from the ground. From 100–120 m a car is a small rectangle seen from above, and the pre-trained network finds only part of the vehicles; on the highway scenes 1-03 and 1-04, where the cars are smallest, it may find none. This is a direct illustration of the **domain gap**: a network for UAV video has to be fine-tuned on aerial images (e.g. VisDrone).

**What to look at.** The trajectories of the tracks over the last frame; the number of detected and of annotated vehicles in every frame; the number of track identifiers against the number of annotated vehicles: extra identifiers mean that tracks were broken and started again (*ID switches*). The annotated video is saved in `outputs`.
""")

code(r"""
#@title 5.4 ByteTrack on the UAV video (optional) { display-mode: "form" }
RUN_BYTETRACK = False #@param {type:"boolean"}
TRACK_SEQUENCE = "1-05 · day, crossroads" #@param ["1-05 · day, crossroads", "1-07 · day, crossroads", "1-09 · day, city street with parked cars", "1-01 · day, suburban road", "1-04 · day, suburban highway"]
TRACK_MODEL = "yolo11s.pt" #@param ["yolo11n.pt", "yolo11s.pt"]
TRACK_CONFIDENCE = 0.1 #@param {type:"slider", min:0.05, max:0.8, step:0.05}
TRACK_IMAGE_SIZE = 1920 #@param [640, 1280, 1920] {type:"raw"}
MAX_TRACK_FRAMES = 60 #@param {type:"slider", min:10, max:150, step:10}

BYTETRACK_RESULT = None
if RUN_BYTETRACK:
    for module, package in [("ultralytics", "ultralytics"), ("lap", "lap>=0.5.12")]:      # lap: assignment solver of ByteTrack
        if importlib.util.find_spec(module) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    from ultralytics import YOLO
    tracker_model = YOLO(TRACK_MODEL)
    vehicle_classes = [2, 5, 7]                                    # COCO: car, bus, truck
    TSEQ = TRACK_SEQUENCE.split(" ")[0]
    n_proc = int(min(MAX_TRACK_FRAMES, m3ot.n_frames(TSEQ, "rgb")))
    TFRAMES = m3ot.frames(TSEQ, "rgb", start=0, count=n_proc)
    TTRACKS = m3ot.tracks(TSEQ, "rgb")
    out_path = os.path.join(OUT, f"bytetrack_{TSEQ}.mp4")
    rows, annotated = [], []
    t0 = time.time()
    for k in range(n_proc):
        r = tracker_model.track(cv2.cvtColor(TFRAMES[k], cv2.COLOR_RGB2BGR), persist=True, tracker="bytetrack.yaml",
                                conf=TRACK_CONFIDENCE, imgsz=TRACK_IMAGE_SIZE, classes=vehicle_classes, verbose=False)[0]
        annotated.append(cv2.cvtColor(r.plot(line_width=1, font_size=8), cv2.COLOR_BGR2RGB))
        if r.boxes.id is not None:
            for b, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
                rows.append(dict(frame=k, id=int(tid), cx=(b[0] + b[2]) / 2, cy=(b[1] + b[3]) / 2))
    elapsed = time.time() - t0
    aitbc.write_video(out_path, annotated, fps=FPS)
    tr = pd.DataFrame(rows, columns=["frame", "id", "cx", "cy"])
    gt_part = TTRACKS[TTRACKS.frame <= n_proc]
    det_per_frame = tr.groupby("frame").size().reindex(range(n_proc), fill_value=0)
    gt_per_frame = gt_part.groupby("frame").size().reindex(range(1, n_proc + 1), fill_value=0)
    display(pd.DataFrame([
        ("Sequence", f"{TSEQ} ({m3ot.SEQUENCES[TSEQ]}), visible channel, {TRACK_MODEL}, S = {TRACK_IMAGE_SIZE}"),
        ("Processed frames", f"{n_proc} ({elapsed / n_proc * 1000:.0f} ms per frame)"),
        ("Track identifiers", tr.id.nunique()),
        ("Annotated vehicles in the same frames", gt_part.id.nunique()),
        ("Mean detected / annotated vehicles per frame", f"{det_per_frame.mean():.1f} / {gt_per_frame.mean():.1f}"),
        ("Annotated video", out_path),
    ], columns=["Quantity", "Value"]).set_index("Quantity"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].imshow(TFRAMES[n_proc - 1])
    cmap = plt.cm.tab20
    for j, (tid, g) in enumerate(tr.groupby("id")):
        axes[0].plot(g.cx, g.cy, color=cmap(j % 20), lw=1.5)
    axes[0].set_title(f"Trajectories of {tr.id.nunique()} tracks over frame {n_proc}"); axes[0].axis("off")
    axes[1].plot(np.arange(n_proc) / FPS, det_per_frame.values, color=COLORS[0], label="tracked by YOLO + ByteTrack")
    axes[1].plot(np.arange(n_proc) / FPS, gt_per_frame.values, color="k", ls="--", label="annotated")
    axes[1].set(xlabel="time, s", ylabel="vehicles in the frame", title="Detected and annotated vehicles")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "5_4_bytetrack.png")
    BYTETRACK_RESULT = dict(sequence=TSEQ, model=TRACK_MODEL, image_size=TRACK_IMAGE_SIZE, frames=n_proc, track_ids=int(tr.id.nunique()), annotated_vehicles=int(gt_part.id.nunique()),
                            mean_detected=float(det_per_frame.mean()), mean_annotated=float(gt_per_frame.mean()))
else:
    print("The ByteTrack block is switched off. Set RUN_BYTETRACK in the form to run it.")
""")

md(r"""
**Task 5.**
1. In 5.1 compare the two methods on a day frame and on a night frame. Why does a single global threshold fail when the scene contains warm asphalt or roofs?
2. From the precision–recall curves choose $k$ for night surveillance where a missed person is more costly than a false alarm.
3. In 5.2 set `DROPOUT = 0.6`. How much does the filter gain over the held measurements? Find the minimum of the curve RMSE($\sigma_a$) for `MEAS_SIGMA_PX` = 2 and 10 and explain the shift.
4. (Optional) In 5.3 compare the recall for `IMAGE_SIZE` 640 and 1600 and relate it to the plot of small objects of block 2.3.
""")

# =====================================================================================
md(r"""
---
## Final mini-project

Choose one task according to the available data: preprocessing of degraded frames, classification, segmentation, or detection and tracking. The blocks of this notebook are the templates.

### Requirements to the result
1. Describe the source, the content and the limitations of the data set (sensor, altitude, time of day, annotation).
2. Separate training and test data **before** the choice of the parameters. For video, split by independent recordings or images, not by neighbouring frames of one recording.
3. Save the source data, the parameters, the code and the results in the project structure (`data`, `frames`, `models`, `outputs`).
4. Justify the choice of the metrics. For detection, precision and recall require a reference annotation; the confidence of a model does not replace its accuracy.
5. Explain the typical errors and the limits of applicability of the method.

### Template of the conclusion
**Task:** …
**Data and the way of splitting:** …
**Method and parameters:** …
**Computed metrics:** …
**Typical errors:** …
**Limitations and further changes:** …

The cell below collects the parameters and the main results of this run in `outputs/course_summary.json`.
""")

code(r"""
#@title Final report of the run { display-mode: "form" }
summary = {
    "seed": SEED, "environment": ENVIRONMENT,
    "video": dict(dataset="M3OT", sequence=SEQ, channel=CHANNEL, first_frame=start + 1, frames=count, fps=FPS, file=VIDEO,
                  sampling_fps=SAMPLING_FPS, saved_frames=int(len(frames_table))),
    "image": IMAGE_SOURCE,
    "restoration": {"noise_sigma": NOISE_SIGMA, "motion_blur_px": MOTION_BLUR_PX, "jpeg_quality": JPEG_QUALITY,
                    "psnr_db": {k: float(v) for k, v in restoration_table["PSNR, dB"].items() if np.isfinite(v)}},
    "classification": {"data": f"HIT-UAV patches, {TIME_OF_DAY}, {len(train_files)} training / {len(test_files)} test images",
                       "patches_train": int(len(train_idx)), "patches_test": int(len(test_idx)),
                       "metrics": {f: {k: float(v) for k, v in CLASSIFICATION.loc[f].items()} for f in CLASSIFICATION.index},
                       "mobilenet": MOBILENET_RESULT},
    "warm_objects": pr.to_dict(orient="records"),
    "tracking": TRACKING_RESULT,
    "yolo": YOLO_RESULT,
    "bytetrack": BYTETRACK_RESULT,
}
path = os.path.join(OUT, "course_summary.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, default=str)
files = sorted(os.listdir(OUT))
display(Markdown(f"**The base route is complete.** {len(files)} files (figures, tables, report) are saved in `{OUT}`."))
print("\n".join(files))
""")

md(r"""
---
## Self-check and sources

### Self-check
- Run the notebook again from the beginning after a restart of the runtime (*Runtime → Restart session and run all*): the numbers must be the same for the same `SEED`.
- Check that the test images were used neither for training nor for the choice of the hyperparameters (4.2: grouped cross-validation on the training images only).
- Change one parameter at a time and keep the others fixed; save the tables of `outputs` before the next change.
- Distinguish the measured result (a table, a metric against the annotation) from an illustration and from an assumption.

### Data sources and licences
The course library reads the files directly from the repositories of the authors; the data are not redistributed with the notebook. When images or results are published, the data sets must be cited.

1. **M3OT.** Nie Z. et al. M3OT: A Multi-Drone Multi-Modality dataset for Multi-Object Tracking. *Scientific Data*, 12, 1927 (2025). https://doi.org/10.1038/s41597-025-06204-0 · Data: https://doi.org/10.6084/m9.figshare.28308887 · Licence CC BY 4.0.
2. **HIT-UAV.** Suo J. et al. HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection. *Scientific Data*, 10, 227 (2023). https://doi.org/10.1038/s41597-023-02066-6 · Data: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset · Licence CC BY 4.0.
3. **VisDrone.** Zhu P. et al. Detection and tracking meet drones challenge. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(11), 7380–7399 (2022). Data: https://github.com/VisDrone/VisDrone-Dataset · Terms of use: see the data set page.

### Documentation
- [Google Colab: FAQ and resource limits](https://research.google.com/colaboratory/faq.html)
- [OpenCV: image processing tutorials](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html) · [ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html) · [Homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
- [scikit-learn: cross-validation with groups](https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold) · [SVM](https://scikit-learn.org/stable/modules/svm.html)
- [scikit-image: HOG](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)
- [PyTorch: transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) · [Ultralytics: tracking](https://docs.ultralytics.com/modes/track/)

The total volume of 40 hours belongs to the training programme. The notebook is a practical companion of the programme and does not replace the lectures and methodical materials. The optional blocks MobileNet, YOLO and ByteTrack depend on the availability of packages, model weights and computing resources.
""")
