# =====================================================================================
md(r"""
---
## Topic 2. The digital image and its preprocessing

A colour frame is a three-dimensional array $H \times W \times 3$ of 8-bit numbers (0–255): rows, columns and the channels R, G, B. A thermal frame is a two-dimensional array $H \times W$ of grey levels. Every later step (filters, features, neural networks) works with these numbers, therefore their distribution, noise and scale define what a model can see.

### 2.1. Colour channels, histograms and object sizes

**Purpose.** Load a real VisDrone frame with annotated pedestrians and vehicles, look at its channels and histograms, and measure how many pixels an object occupies.

**Parameters.**
- `IMAGE_NUMBER` — one of ten selected VisDrone frames (different heights, view angles and scenes).
- `WORK_WIDTH` — width of the working copy, px. The frame and its boxes are scaled together; a smaller width speeds up the following blocks but makes small objects smaller.

**What to look at.**
1. The histograms: a narrow histogram means low contrast; peaks at 0 or 255 mean clipping (lost details in shadows or highlights).
2. The table of channel statistics: the entropy $H = -\sum_i p_i \log_2 p_i$ (bits) measures how many grey levels the channel actually uses (maximum 8 bits).
3. The distribution of the object size $\sqrt{w h}$: in drone images most objects are only 10–40 px across. This is the main difficulty of detection from a UAV.

Note: OpenCV reads files in the order **BGR**; after `cv2.imread` the channels must be converted with `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`. The course library returns RGB images.
""")

code(r"""
#@title 2.1 Colour channels, histograms and object sizes { display-mode: "form" }
IMAGE_NUMBER = 1 #@param {type:"slider", min:1, max:10, step:1}
WORK_WIDTH = 960 #@param {type:"slider", min:480, max:1600, step:80}

IMAGE_NAME = vis.CURATED[IMAGE_NUMBER - 1]
full_rgb, full_objects = vis.load(IMAGE_NAME)
k_scale = WORK_WIDTH / full_rgb.shape[1]
image_rgb = cv2.resize(full_rgb, (WORK_WIDTH, int(round(full_rgb.shape[0] * k_scale))),
                       interpolation=cv2.INTER_AREA if k_scale < 1 else cv2.INTER_LINEAR)
image_objects = [dict(o, box=[v * k_scale for v in o["box"]]) for o in full_objects]
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
IMAGE_SOURCE = f"VisDrone2019-DET val, {IMAGE_NAME}"
print(f"{IMAGE_SOURCE}: original {full_rgb.shape[1]} × {full_rgb.shape[0]} px, working copy "
      f"{image_rgb.shape[1]} × {image_rgb.shape[0]} px, array {image_rgb.shape}, type {image_rgb.dtype}, "
      f"{len(image_objects)} annotated objects")


def entropy_bits(channel):
    p = np.bincount(channel.ravel(), minlength=256) / channel.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


stats = []
for i, name in enumerate("RGB"):
    c = image_rgb[:, :, i]
    stats.append(dict(Channel=name, Mean=c.mean(), Std=c.std(), P1=np.percentile(c, 1), P99=np.percentile(c, 99),
                      **{"Entropy, bits": entropy_bits(c)}, **{"Clipped, %": 100 * np.mean((c == 0) | (c == 255))}))
stats.append(dict(Channel="Grey", Mean=image_gray.mean(), Std=image_gray.std(), P1=np.percentile(image_gray, 1),
                  P99=np.percentile(image_gray, 99), **{"Entropy, bits": entropy_bits(image_gray)},
                  **{"Clipped, %": 100 * np.mean((image_gray == 0) | (image_gray == 255))}))
display(pd.DataFrame(stats).set_index("Channel").round(2))

fig = plt.figure(figsize=(15, 6.2))
grid = fig.add_gridspec(3, 2, width_ratios=[2.6, 1], hspace=0.25, wspace=0.03)
ax = fig.add_subplot(grid[:, 0])
ax.imshow(aitbc.draw_boxes(image_rgb, image_objects, colors=vis.COLORS, labels=False, thickness=1))
present = sorted({o["cls"] for o in image_objects})
aitbc.legend(ax, present, [vis.COLORS[c] for c in present], ncol=2)
ax.set_title(f"{IMAGE_SOURCE}: {len(image_objects)} annotated objects"); ax.axis("off")
for i, name in enumerate("RGB"):
    ax = fig.add_subplot(grid[i, 1])
    ax.imshow(image_rgb[:, :, i], cmap="gray", vmin=0, vmax=255); ax.axis("off")
    ax.set_title(f"channel {name} as grey levels", fontsize=9)
save_figure(fig, "2_1_frame_channels.png")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
ax = axes[0]
for i, (name, color) in enumerate(zip("RGB", ["#c0392b", "#27ae60", "#2e6db4"])):
    h = np.bincount(image_rgb[:, :, i].ravel(), minlength=256) / image_gray.size
    ax.plot(np.arange(256), h, color=color, lw=1.3, label=name)
ax2 = ax.twinx()
ax2.plot(np.arange(256), np.cumsum(np.bincount(image_gray.ravel(), minlength=256)) / image_gray.size,
         color="k", lw=1, ls="--", label="cumulative (grey)")
ax2.set_ylabel("cumulative share"); ax2.spines["right"].set_visible(True)
ax.set(xlabel="brightness level", ylabel="relative frequency", title="Histograms of the channels")
ax.legend(loc="upper left"); ax2.legend(loc="center right")
ax = axes[1]
sizes = np.array([math.sqrt(max(o["box"][2] - o["box"][0], 1) * max(o["box"][3] - o["box"][1], 1)) for o in image_objects])
groups = {"persons": ("pedestrian", "people"), "two-wheelers": ("bicycle", "motor", "tricycle", "awning-tricycle"),
          "cars and vans": ("car", "van"), "trucks and buses": ("truck", "bus")}
bins = np.linspace(0, max(80, sizes.max() + 5), 25)
data = [[z for z, o in zip(sizes, image_objects) if o["cls"] in members] for members in groups.values()]
ax.hist([d for d in data if d], bins=bins, stacked=True, color=[c for d, c in zip(data, COLORS) if d],
        label=[f"{g} ({len(d)})" for g, d in zip(groups, data) if d])
ax.axvline(np.median(sizes), color="k", ls="--", lw=1)
ax.text(np.median(sizes), ax.get_ylim()[1] * 0.92, f"  median {np.median(sizes):.0f} px")
ax.set(xlabel="object size \u221a(w\u00b7h), px", ylabel="number of objects", title="How many pixels an object occupies (stacked)")
ax.legend()
fig.tight_layout(w_pad=4)
save_figure(fig, "2_1_histograms_sizes.png")
""")

md(r"""
### 2.2. Degradations of the image and noise filters

**Purpose.** Reproduce typical degradations of a UAV video (sensor noise, motion blur, compression in the radio link) and compare four noise filters by objective metrics.

The metrics compare the restored image $\hat I$ with the known undistorted frame $I$:
- **PSNR** (peak signal-to-noise ratio), dB: $\mathrm{PSNR} = 10 \log_{10} \dfrac{255^2}{\mathrm{MSE}}$, where MSE is the mean squared difference of the pixels. +6 dB means half the error amplitude.
- **SSIM** (structural similarity, 0…1): compares local brightness, contrast and structure; it is closer to visual quality than PSNR.
- **Sharpness**: variance of the Laplacian; filters that remove noise also lower it.

| Filter | Principle | Typical property |
|---|---|---|
| Gaussian | weighted mean of a $k \times k$ window | fast, blurs edges |
| Median | median of a $k \times k$ window | removes impulse noise, keeps edges |
| Bilateral | weights by distance **and** by brightness difference | keeps edges, slow on large windows |
| Non-local means | averages similar patches of the whole neighbourhood | best quality, the slowest |

**Parameters.** `NOISE_SIGMA` — standard deviation of the Gaussian sensor noise, grey levels; `MOTION_BLUR_PX` — length of the motion blur, px; `JPEG_QUALITY` — JPEG quality of the video link (100 = no compression); `KERNEL` — window size $k$ of the filters; `NLM_STRENGTH` — strength $h$ of non-local means.

**What to look at.** The table and the enlarged fragment with small objects; the curves PSNR($\sigma$) and SSIM($\sigma$): which filter is best at weak noise and which one at strong noise.
""")

code(r"""
#@title 2.2 Degradations of the image and noise filters { display-mode: "form" }
NOISE_SIGMA = 20 #@param {type:"slider", min:0, max:60, step:2}
MOTION_BLUR_PX = 0 #@param {type:"slider", min:0, max:20, step:1}
JPEG_QUALITY = 100 #@param {type:"slider", min:10, max:100, step:5}
KERNEL = 5 #@param [3, 5, 7, 9] {type:"raw"}
NLM_STRENGTH = 10 #@param {type:"slider", min:3, max:30, step:1}

degraded_rgb = aitbc.degrade(image_rgb, motion_px=MOTION_BLUR_PX, noise=NOISE_SIGMA, jpeg_quality=JPEG_QUALITY, seed=SEED)


def apply_filters(img, k=KERNEL, h=NLM_STRENGTH, sigma=NOISE_SIGMA):
    return {
        "Gaussian": cv2.GaussianBlur(img, (k, k), 0),
        "Median": cv2.medianBlur(img, k),
        "Bilateral": cv2.bilateralFilter(img, k, 2.0 * sigma + 10.0, k),
        "Non-local means": cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21),
    }


t0 = time.time()
filtered = apply_filters(degraded_rgb)
rows = [dict(Method="Degraded frame", **{"PSNR, dB": aitbc.psnr(image_rgb, degraded_rgb), "SSIM": aitbc.ssim(image_rgb, degraded_rgb),
             "Sharpness": aitbc.sharpness(degraded_rgb)})]
for name, img in filtered.items():
    rows.append(dict(Method=name, **{"PSNR, dB": aitbc.psnr(image_rgb, img), "SSIM": aitbc.ssim(image_rgb, img),
                                     "Sharpness": aitbc.sharpness(img)}))
restoration_table = pd.DataFrame(rows).set_index("Method")
restoration_table.loc["Undistorted frame (reference)"] = [np.inf, 1.0, aitbc.sharpness(image_rgb)]
display(restoration_table.round(3))
restoration_table.to_csv(os.path.join(OUT, "restoration_metrics.csv"))

# enlarged fragment around the densest group of objects
centres = np.array([[(o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2] for o in image_objects])
H_img, W_img = image_gray.shape
cw, ch = W_img // 4, H_img // 4
if len(centres):
    best = max(centres, key=lambda c: np.sum((np.abs(centres[:, 0] - c[0]) < cw / 2) & (np.abs(centres[:, 1] - c[1]) < ch / 2)))
else:
    best = (W_img / 2, H_img / 2)
x0 = int(np.clip(best[0] - cw / 2, 0, W_img - cw)); y0 = int(np.clip(best[1] - ch / 2, 0, H_img - ch))
crop = lambda im: cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (cw * 3, ch * 3), interpolation=cv2.INTER_NEAREST)
panels = [image_rgb, degraded_rgb] + list(filtered.values())
names = ["Reference", f"Degraded: σ = {NOISE_SIGMA}, blur {MOTION_BLUR_PX} px, JPEG {JPEG_QUALITY}"] + \
        [f"{n}: {restoration_table.loc[n, 'PSNR, dB']:.1f} dB, SSIM {restoration_table.loc[n, 'SSIM']:.2f}" for n in filtered]
show_images([crop(p) for p in panels], names, "2_2_filters_fragment.png", cols=3, width=4.6)
print(f"Filters computed in {time.time() - t0:.1f} s; fragment {cw} × {ch} px at x = {x0}, y = {y0}, enlarged 3 times")

# sweep of the noise level (noise only, the other degradations off)
sigmas = [0, 5, 10, 20, 30, 45, 60]
sweep = []
for s in sigmas:
    noisy = aitbc.add_noise(image_rgb, s, seed=SEED) if s > 0 else image_rgb.copy()
    sweep.append(dict(sigma=s, method="No filter", psnr=aitbc.psnr(image_rgb, noisy), ssim=aitbc.ssim(image_rgb, noisy)))
    for name, img in apply_filters(noisy, sigma=max(s, 1), h=max(3, 0.6 * s)).items():
        sweep.append(dict(sigma=s, method=name, psnr=aitbc.psnr(image_rgb, img), ssim=aitbc.ssim(image_rgb, img)))
sweep = pd.DataFrame(sweep)
sweep.to_csv(os.path.join(OUT, "restoration_sweep.csv"), index=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for (name, g), color, marker in zip(sweep.groupby("method", sort=False), ["k"] + COLORS, "osD^v"):
    axes[0].plot(g.sigma, g.psnr.clip(upper=50), marker=marker, color=color, label=name)
    axes[1].plot(g.sigma, g.ssim, marker=marker, color=color, label=name)
axes[0].set(xlabel="noise σ, grey levels", ylabel="PSNR, dB (clipped at 50)", title=f"PSNR against noise, window k = {KERNEL}")
axes[1].set(xlabel="noise σ, grey levels", ylabel="SSIM", title="SSIM against noise (h of NL-means = 0.6 σ)")
for ax in axes:
    ax.axvline(NOISE_SIGMA, color="grey", ls=":", lw=1)
    ax.grid(alpha=0.3)
axes[1].legend()
fig.tight_layout()
save_figure(fig, "2_2_noise_sweep.png")
""")

md(r"""
### 2.3. Low light, contrast enhancement and letterbox

**Purpose.** Simulate a frame recorded at dusk (less light reaches the sensor, the photon noise grows relative to the signal), compare three methods of contrast enhancement, and prepare the frame for a neural network with a fixed input size (letterbox).

| Method | Principle |
|---|---|
| Gamma correction | $I_{out} = 255\,(I_{in}/255)^{\gamma}$, $\gamma < 1$ brightens dark tones |
| Global histogram equalisation | one brightness mapping for the whole frame, the histogram becomes flat |
| **CLAHE** | equalisation in tiles of a $n \times n$ grid with a clip limit of the histogram; limits the amplification of noise |

The enhancement is applied to the lightness channel $L$ of the colour space LAB, so that the colours are not distorted.

**Letterbox.** A detector such as YOLO takes a square input, e.g. 640 × 640. The frame is scaled with the factor $r = \min(S/W,\ S/H)$ without changing the proportions and the free area is filled with grey. The boxes of the objects must be transformed in the same way: $x' = r\,x + \Delta x$, $y' = r\,y + \Delta y$.

**Parameters.** `LIGHT_LEVEL` — share of the light that reaches the sensor (1 = normal day); `GAMMA`; `CLAHE_CLIP` — clip limit; `CLAHE_TILES` — size $n$ of the tile grid; `TARGET_SIZE` — input size $S$ of the network.

**What to look at.**
1. The table: SSIM with the normal frame and the noise in a flat area. A stronger enhancement makes the frame brighter, and the noise brighter too.
2. The plot "objects smaller than 8 px against the input size": at small $S$ most VisDrone objects become too small for any detector.
""")

code(r"""
#@title 2.3 Low light, contrast enhancement and letterbox { display-mode: "form" }
LIGHT_LEVEL = 0.08 #@param {type:"slider", min:0.02, max:0.5, step:0.01}
GAMMA = 0.5 #@param {type:"slider", min:0.2, max:1.0, step:0.05}
CLAHE_CLIP = 2.5 #@param {type:"slider", min:1.0, max:8.0, step:0.5}
CLAHE_TILES = 8 #@param [2, 4, 8, 16] {type:"raw"}
TARGET_SIZE = 640 #@param [320, 416, 640, 960, 1280] {type:"raw"}

dark_rgb = aitbc.low_light(image_rgb, level=LIGHT_LEVEL, gain_max=1.0, seed=SEED)   # no automatic gain


def on_lightness(rgb, fn):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = fn(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def clahe(rgb, clip=CLAHE_CLIP, tiles=CLAHE_TILES):
    return on_lightness(rgb, cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply)


gamma_lut = (255.0 * (np.arange(256) / 255.0) ** GAMMA).astype(np.uint8)
enhanced = {
    f"Gamma γ = {GAMMA}": on_lightness(dark_rgb, lambda L: cv2.LUT(L, gamma_lut)),
    "Global equalisation": on_lightness(dark_rgb, cv2.equalizeHist),
    f"CLAHE clip {CLAHE_CLIP}, grid {CLAHE_TILES}×{CLAHE_TILES}": clahe(dark_rgb),
}
# flat area for the noise estimate: the 10 % of 16x16 blocks with the smallest gradient in the normal frame
grad = cv2.blur(np.abs(cv2.Laplacian(image_gray, cv2.CV_32F)), (16, 16))
flat = grad <= np.percentile(grad, 10)


def noise_in_flat(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float((g - cv2.GaussianBlur(g, (0, 0), 2.0))[flat].std())


rows = []
for name, img in [("Normal frame", image_rgb), (f"Dark frame, light {LIGHT_LEVEL:.2f}", dark_rgb)] + list(enhanced.items()):
    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0]
    rows.append({"Method": name, "Mean L": L.mean(), "RMS contrast": L.std(), "Entropy, bits": entropy_bits(L),
                 "SSIM with normal": aitbc.ssim(image_rgb, img), "Noise in flat area": noise_in_flat(img)})
contrast_table = pd.DataFrame(rows).set_index("Method")
display(contrast_table.round(3))
contrast_table.to_csv(os.path.join(OUT, "contrast_metrics.csv"))
show_images([dark_rgb] + list(enhanced.values()), [f"Dark frame, light {LIGHT_LEVEL:.2f}"] + list(enhanced),
            "2_3_contrast.png", cols=4, width=3.8)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for (name, img), color in zip([("Normal", image_rgb), ("Dark", dark_rgb)] + list(enhanced.items()), ["k", "grey"] + COLORS):
    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0]
    axes[0].plot(np.bincount(L.ravel(), minlength=256) / L.size, color=color, lw=1.2, label=name)
axes[0].set(xlabel="lightness L", ylabel="relative frequency", title="Histograms of L", ylim=(0, None))
axes[0].legend(fontsize=8)
for tiles, color in zip([2, 4, 8, 16], COLORS):
    clips = np.arange(1.0, 8.01, 1.0)
    res = [clahe(dark_rgb, c, tiles) for c in clips]
    axes[1].plot(clips, [aitbc.ssim(image_rgb, r) for r in res], marker="o", color=color, label=f"grid {tiles}×{tiles}")
axes[1].set(xlabel="CLAHE clip limit", ylabel="SSIM with the normal frame", title="Tuning of CLAHE on the dark frame")
axes[1].axvline(CLAHE_CLIP, color="grey", ls=":"); axes[1].grid(alpha=0.3); axes[1].legend()
fig.tight_layout()
save_figure(fig, "2_3_clahe_tuning.png")


def letterbox(rgb, size=640, color=(114, 114, 114)):
    # scale without changing the proportions and pad to size x size; returns the image and the parameters
    h, w = rgb.shape[:2]
    r = min(size / w, size / h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    left, top = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, dict(ratio=r, left=left, top=top)


def letterbox_boxes(objects, p):
    return [dict(o, box=[o["box"][0] * p["ratio"] + p["left"], o["box"][1] * p["ratio"] + p["top"],
                         o["box"][2] * p["ratio"] + p["left"], o["box"][3] * p["ratio"] + p["top"]]) for o in objects]


lb_rgb, lb_info = letterbox(full_rgb, TARGET_SIZE)
lb_objects = letterbox_boxes(full_objects, lb_info)
min_side = lambda objs: np.array([min(o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]) for o in objs])
sizes_S = np.array([256, 320, 416, 512, 640, 800, 960, 1280, 1600, 2048])
share_small = [100 * np.mean(min_side(letterbox_boxes(full_objects, dict(ratio=min(S / full_rgb.shape[1], S / full_rgb.shape[0]),
                                                                          left=0, top=0))) < 8) for S in sizes_S]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
axes[0].imshow(aitbc.draw_boxes(lb_rgb, lb_objects, colors=vis.COLORS, labels=False))
axes[0].set_title(f"Letterbox {TARGET_SIZE} × {TARGET_SIZE}: r = {lb_info['ratio']:.3f}, "
                  f"Δx = {lb_info['left']}, Δy = {lb_info['top']} px"); axes[0].axis("off")
axes[1].plot(sizes_S, share_small, marker="o", color=COLORS[1])
axes[1].axvline(TARGET_SIZE, color="grey", ls=":")
axes[1].set(xscale="log", xlabel="input size S of the network, px", ylabel="objects with a side < 8 px, %",
            title=f"Small objects after scaling (original frame {full_rgb.shape[1]} × {full_rgb.shape[0]} px)")
axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); axes[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[1].set_xticks(sizes_S); axes[1].set_xticklabels(sizes_S, rotation=45); axes[1].grid(alpha=0.3)
fig.tight_layout()
save_figure(fig, "2_3_letterbox.png")
""")

md(r"""
**Task 2.**
1. Set `NOISE_SIGMA` to 5 and to 50. Which filter has the highest PSNR in each case? Does the order of the filters by SSIM coincide with the order by PSNR?
2. Add a motion blur of 8 px. Why do the noise filters not restore a blurred frame?
3. Change `CLAHE_CLIP` from 1 to 8 and look at the column "Noise in flat area". Explain why a better visual impression does not guarantee a higher accuracy of the next model.
4. With the plot of small objects choose the input size of the detector for this frame. What is the price of a large input size?
""")
