# =====================================================================================
md(r"""
---
## Topic 3. Classical methods of feature extraction

Before neural networks, computer vision was built on hand-designed features: brightness gradients, edges, corners and local descriptors. They are still used on board of UAVs because they are fast, need no training and are easy to interpret: visual odometry, image stitching, stabilisation of the video, detection of moving objects.

### 3.1. Gradients, edges and contours

**Purpose.** Compute the brightness gradient with the Sobel operator, find edges with the Canny detector and outline the contours.

The Sobel operator estimates the derivatives $G_x = \partial I/\partial x$, $G_y = \partial I/\partial y$; the gradient magnitude $|G| = \sqrt{G_x^2 + G_y^2}$ is large at edges and the orientation $\theta = \operatorname{atan2}(G_y, G_x)$ shows their direction. The Canny detector keeps the local maxima of $|G|$ and uses two thresholds: pixels above `CANNY_HIGH` are strong edges, pixels between `CANNY_LOW` and `CANNY_HIGH` are kept when they are connected to strong edges (hysteresis).

**Parameters.**
- `BLUR_SIGMA` — Gaussian smoothing before Canny (suppresses noise and fine texture);
- `AUTO_THRESHOLDS` — thresholds from the median $m$ of the frame: $T_{low} = 0.66\,m$, $T_{high} = 1.33\,m$;
- `CANNY_LOW`, `CANNY_HIGH` — manual thresholds; `MIN_CONTOUR_LENGTH` — shortest contour to draw, px.

**What to look at.** The colour map of the orientation (hue = direction of the edge); the curve of edge density against the threshold; the edge density inside the object boxes and in the background: vehicles and people are richer in edges than the road surface, therefore edges are a first cue for detection.
""")

code(r"""
#@title 3.1 Gradients, edges and contours { display-mode: "form" }
BLUR_SIGMA = 1.0 #@param {type:"slider", min:0, max:4, step:0.25}
AUTO_THRESHOLDS = False #@param {type:"boolean"}
CANNY_LOW = 60 #@param {type:"slider", min:0, max:250, step:5}
CANNY_HIGH = 150 #@param {type:"slider", min:10, max:255, step:5}
MIN_CONTOUR_LENGTH = 60 #@param {type:"slider", min:10, max:300, step:10}

smooth = cv2.GaussianBlur(image_gray, (0, 0), BLUR_SIGMA) if BLUR_SIGMA > 0 else image_gray
if AUTO_THRESHOLDS:
    m = float(np.median(smooth))
    low, high = int(max(0, 0.66 * m)), int(min(255, 1.33 * m))
else:
    low, high = CANNY_LOW, CANNY_HIGH
if low >= high:
    raise ValueError("The low threshold of Canny must be smaller than the high threshold.")
gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
edges = cv2.Canny(smooth, low, high)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
long_contours = [c for c in contours if cv2.arcLength(c, False) > MIN_CONTOUR_LENGTH]
contours_rgb = image_rgb.copy()
cv2.drawContours(contours_rgb, long_contours, -1, (255, 60, 40), 1)

# orientation map: hue = direction, value = magnitude
hsv = np.zeros(image_rgb.shape, np.uint8)
hsv[:, :, 0] = (angle % 180).astype(np.uint8)
hsv[:, :, 1] = 255
hsv[:, :, 2] = np.clip(magnitude / np.percentile(magnitude, 99) * 255, 0, 255).astype(np.uint8)
orientation_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
show_images([np.clip(magnitude / np.percentile(magnitude, 99), 0, 1), orientation_rgb, edges, contours_rgb],
            ["Sobel gradient magnitude |G|", "Orientation θ (hue) and |G| (brightness)",
             f"Canny edges, thresholds {low} / {high}", f"{len(long_contours)} contours longer than {MIN_CONTOUR_LENGTH} px"],
            "3_1_edges.png", cols=2, width=6.2)

object_mask = np.zeros(image_gray.shape, bool)
for o in image_objects:
    x1, y1, x2, y2 = [int(round(v)) for v in o["box"]]
    object_mask[max(y1, 0):y2, max(x1, 0):x2] = True
highs = np.arange(20, 256, 10)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
for ratio, color in zip([0.3, 0.5, 0.8], COLORS):
    axes[0].plot(highs, [100 * np.mean(cv2.Canny(smooth, int(ratio * h), int(h)) > 0) for h in highs],
                 color=color, marker=".", label=f"T_low = {ratio} · T_high")
axes[0].scatter([high], [100 * np.mean(edges > 0)], color="k", zorder=3, label="this run")
axes[0].set(xlabel="high threshold T_high", ylabel="edge pixels, %", title="Edge density against the Canny threshold")
axes[0].grid(alpha=0.3); axes[0].legend()
inside, outside = 100 * np.mean(edges[object_mask] > 0), 100 * np.mean(edges[~object_mask] > 0)
axes[1].bar(["inside object boxes", "background"], [inside, outside], color=[COLORS[1], COLORS[0]])
for i, v in enumerate([inside, outside]):
    axes[1].text(i, v, f"{v:.1f} %", ha="center", va="bottom")
axes[1].set(ylabel="edge pixels, %", title=f"Edges on objects and on the background ({inside / max(outside, 1e-6):.1f} ×)")
fig.tight_layout()
save_figure(fig, "3_1_edge_density.png")
""")

md(r"""
### 3.2. ORB features and estimation of a homography

**Purpose.** Find corresponding points of two views of the same scene and estimate the geometric transformation between them.

**ORB** (Oriented FAST and Rotated BRIEF) finds corners (FAST) and describes the neighbourhood of every corner by a 256-bit binary string (BRIEF), rotated by the dominant direction of the corner. Descriptors are compared by the Hamming distance (number of different bits). A match is accepted by the **ratio test**: the best distance must be clearly smaller than the second best, $d_1 < \rho\, d_2$.

A plane scene viewed by two cameras is related by a **homography** — a $3 \times 3$ matrix $H$:
$$\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} \sim H \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}.$$
**RANSAC** repeatedly estimates $H$ from 4 random matches and keeps the estimate with the largest number of matches (inliers) whose reprojection error is below `RANSAC_PX`.

In this block the second view is produced from the frame by a **known** homography $H_{true}$ (rotation, scale and perspective tilt). This makes it possible to measure the error exactly: the corner error is the mean distance between the frame corners transformed by $H_{est}$ and by $H_{true}$.

**Parameters.** `ORB_FEATURES` — number of ORB corners; `RATIO_TEST` — $\rho$; `RANSAC_PX` — inlier threshold; `ROTATION_DEG`, `SCALE`, `TILT` — the known transformation (tilt = perspective, as when the camera is inclined).

**What to look at.** Inlier matches (green) and rejected matches (red); the corner error; the robustness curves: ORB descriptors are rotated with the corner, so the number of inliers hardly depends on rotation, but it falls quickly when the scale changes by more than 1.5–2 times.
""")

code(r"""
#@title 3.2 ORB features and homography with a known transformation { display-mode: "form" }
ORB_FEATURES = 2000 #@param {type:"slider", min:250, max:6000, step:250}
RATIO_TEST = 0.75 #@param {type:"slider", min:0.5, max:0.95, step:0.05}
RANSAC_PX = 3.0 #@param {type:"slider", min:0.5, max:10, step:0.5}
ROTATION_DEG = 20 #@param {type:"slider", min:-90, max:90, step:5}
SCALE = 0.9 #@param {type:"slider", min:0.4, max:2.0, step:0.05}
TILT = 2.0 #@param {type:"slider", min:0, max:6, step:0.5}

H_img, W_img = image_gray.shape
corners = np.float32([[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]]).reshape(-1, 1, 2)


def known_homography(rotation_deg, scale, tilt):
    # rotation and scale about the frame centre, then a perspective tilt (units 1e-4 per px)
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    to_c = np.array([[1, 0, -W_img / 2], [0, 1, -H_img / 2], [0, 0, 1]])
    rs = np.array([[scale * c, -scale * s, 0], [scale * s, scale * c, 0], [0, 0, 1]])
    persp = np.array([[1, 0, 0], [0, 1, 0], [0, tilt * 1e-4, 1]])
    back = np.array([[1, 0, W_img / 2], [0, 1, H_img / 2], [0, 0, 1]])
    return back @ persp @ rs @ to_c


def match_orb(gray_a, gray_b, n_features=ORB_FEATURES, ratio=RATIO_TEST, ransac_px=RANSAC_PX):
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gray_a, None)
    kp2, des2 = orb.detectAndCompute(gray_b, None)
    out = dict(kp1=kp1, kp2=kp2, good=[], H=None, inliers=np.zeros(0, bool))
    if des1 is None or des2 is None:
        return out
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    out["good"] = good
    if len(good) >= 8:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
        if H is not None:
            out.update(H=H, inliers=mask.ravel().astype(bool), src=src, dst=dst)
    return out


def corner_error(H_est, H_true):
    if H_est is None:
        return np.nan
    a = cv2.perspectiveTransform(corners, H_est)
    b = cv2.perspectiveTransform(corners, H_true.astype(np.float32))
    return float(np.linalg.norm(a - b, axis=2).mean())


H_true = known_homography(ROTATION_DEG, SCALE, TILT)
warped_rgb = cv2.warpPerspective(image_rgb, H_true, (W_img, H_img), borderMode=cv2.BORDER_CONSTANT)
res = match_orb(image_gray, cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY))
if res["H"] is None:
    print(f"The homography was not estimated: {len(res['good'])} matches after the ratio test.")
else:
    inl = res["inliers"]
    reproj = np.linalg.norm(cv2.perspectiveTransform(res["src"][inl], res["H"]) - res["dst"][inl], axis=2).ravel()
    orb_table = pd.DataFrame([("ORB corners in the frames", f"{len(res['kp1'])} / {len(res['kp2'])}"),
                              ("Matches after the ratio test", len(res["good"])),
                              ("RANSAC inliers", f"{inl.sum()} ({100 * inl.mean():.0f} %)"),
                              ("Median reprojection error of inliers", f"{np.median(reproj):.2f} px"),
                              ("Corner error of H_est against H_true", f"{corner_error(res['H'], H_true):.2f} px")],
                             columns=["Quantity", "Value"]).set_index("Quantity")
    display(orb_table)
    sel = np.random.default_rng(SEED).choice(len(res["good"]), size=min(80, len(res["good"])), replace=False)
    vis_matches = cv2.drawMatches(image_rgb, res["kp1"], warped_rgb, res["kp2"], [res["good"][i] for i in sel if inl[i]], None,
                                  matchColor=(40, 220, 60), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_matches = cv2.drawMatches(image_rgb, res["kp1"], warped_rgb, res["kp2"], [res["good"][i] for i in sel if not inl[i]],
                                  vis_matches, matchColor=(255, 40, 40),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    show_images([vis_matches], [f"ORB matches (80 shown): green = inliers, red = rejected by RANSAC; "
                                f"rotation {ROTATION_DEG}°, scale {SCALE}, tilt {TILT}"], "3_2_orb_matches.png", width=14)

# robustness of ORB to rotation and scale
rows = []
for r in range(-90, 91, 15):
    H = known_homography(r, 1.0, 0.0)
    out = match_orb(image_gray, cv2.warpPerspective(image_gray, H, (W_img, H_img)))
    rows.append(dict(kind="rotation", value=r, inliers=int(out["inliers"].sum()), error=corner_error(out["H"], H)))
for s in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    H = known_homography(0, s, 0.0)
    out = match_orb(image_gray, cv2.warpPerspective(image_gray, H, (W_img, H_img)))
    rows.append(dict(kind="scale", value=s, inliers=int(out["inliers"].sum()), error=corner_error(out["H"], H)))
robust = pd.DataFrame(rows)
robust.to_csv(os.path.join(OUT, "orb_robustness.csv"), index=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, kind, xlabel in [(axes[0], "rotation", "rotation, degrees"), (axes[1], "scale", "scale factor")]:
    g = robust[robust.kind == kind]
    ax.plot(g.value, g.inliers, marker="o", color=COLORS[0])
    ax.set(xlabel=xlabel, ylabel="RANSAC inliers", title=f"ORB against {kind}")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.semilogy(g.value, g.error.clip(lower=0.01).fillna(1e3), marker="s", ls="--", color=COLORS[1])
    ax2.set_ylabel("corner error, px (log)", color=COLORS[1])
fig.tight_layout()
save_figure(fig, "3_2_orb_robustness.png")
""")

md(r"""
### 3.3. Compensation of the camera motion in a real video

**Purpose.** Apply the same method to two real frames of the UAV video from block 1.1. Even a hovering UAV drifts and sways; after the estimated homography is applied to the earlier frame, the static background cancels in the difference image and the **moving vehicles stand out**. This is the basis of the detection of moving objects from a moving camera.

**Parameters.** `FIRST_FRAME` — number of the first frame inside the fragment; `FRAME_GAP` — distance to the second frame (frames; at 10 frames/s a gap of 10 is 1 s).

**What to look at.** The difference images without and with compensation; the table with the mean residual; the annotated vehicles (yellow boxes) on the compensated difference image.
""")

code(r"""
#@title 3.3 Compensation of the camera motion in the UAV video { display-mode: "form" }
FIRST_FRAME = 1 #@param {type:"slider", min:1, max:150, step:1}
FRAME_GAP = 10 #@param {type:"slider", min:1, max:50, step:1}

i1 = int(min(FIRST_FRAME - 1, count - 2))
i2 = int(min(i1 + FRAME_GAP, count - 1))
fa, fb = FRAMES[i1], FRAMES[i2]
ga, gb = cv2.cvtColor(fa, cv2.COLOR_RGB2GRAY), cv2.cvtColor(fb, cv2.COLOR_RGB2GRAY)
res = match_orb(ga, gb, n_features=3000, ransac_px=2.0)
if res["H"] is None:
    raise RuntimeError("The homography between the frames was not estimated; choose other frames.")
h, w = ga.shape
aligned = cv2.warpPerspective(ga, res["H"], (w, h))
valid = cv2.warpPerspective(np.ones_like(ga), res["H"], (w, h)) > 0
diff_raw = cv2.absdiff(ga, gb)
diff_comp = cv2.absdiff(aligned, gb) * valid
shift = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), res["H"])[0, 0] - [w / 2, h / 2]
objs_b = m3ot.objects(TRACKS, start + i2 + 1)
box_mask = np.zeros(gb.shape, bool)
for o in objs_b:
    x1, y1, x2, y2 = [int(round(v)) for v in o["box"]]
    box_mask[max(y1, 0):y2, max(x1, 0):x2] = True
display(pd.DataFrame([
    ("Frames", f"{start + i1 + 1} and {start + i2 + 1} ({(i2 - i1) / FPS:.1f} s apart)"),
    ("Inliers of the homography", f"{res['inliers'].sum()} of {len(res['good'])}"),
    ("Image shift at the frame centre", f"({shift[0]:.1f}, {shift[1]:.1f}) px"),
    ("Mean residual on the background, without / with compensation",
     f"{diff_raw[~box_mask].mean():.2f} / {diff_comp[valid & ~box_mask].mean():.2f} grey levels"),
    ("Mean residual inside vehicle boxes, with compensation", f"{diff_comp[box_mask & valid].mean():.2f} grey levels"),
], columns=["Quantity", "Value"]).set_index("Quantity"))
vmax = np.percentile(diff_raw, 99.5)
show_images([fb, np.clip(diff_raw / vmax, 0, 1),
             aitbc.draw_boxes(aitbc.colorize(np.clip(diff_comp / vmax * 255, 0, 255).astype(np.uint8) // 1,
                                             (plt.cm.inferno(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)),
                              objs_b, color=(255, 230, 0), labels=False)],
            [f"Frame {start + i2 + 1}", "|difference| without compensation", "|difference| after compensation + annotated vehicles"],
            "3_3_motion_compensation.png", cols=3, width=4.8)
""")

md(r"""
**Task 3.**
1. In block 3.2 set the rotation to 90° and then the scale to 0.5. Which change does ORB tolerate better and why?
2. Change `RATIO_TEST` from 0.6 to 0.9. How do the numbers of matches and inliers change, and what happens with the corner error?
3. In block 3.3 increase `FRAME_GAP`. Why does the residual on the background grow with the gap even after compensation? (Hint: the homography describes a plane; buildings, trees and lamp posts are not in the plane of the road.)
""")
