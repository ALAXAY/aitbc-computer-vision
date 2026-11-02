# =====================================================================================
md(r"""
---
## Topic 4. Machine learning and neural networks

A classifier learns a mapping from an image (or its features) to a class from labelled examples. The quality of the learned mapping is measured on data that were **not used** for training or for the choice of the hyperparameters. In video and aerial data, neighbouring frames and objects of the same image are very similar; if they fall into both the training and the test set, the test score is too optimistic (*data leakage*). Therefore the split is made **by images**, not by individual patches.

### 4.1. Data set of thermal patches from real UAV images

**Purpose.** Build a classification data set from the annotated HIT-UAV thermal images (DJI Matrice M210 V2, Zenmuse XT2 camera, 60–130 m, day and night): square patches around persons, cars and bicycles plus random background patches without objects.

**Parameters.**
- `N_IMAGES` — number of thermal images (about 0.1 MB each). 75 % are taken from the training part of HIT-UAV and 25 % from its validation and test parts: the split by images is defined by the data set itself.
- `PATCH_SIZE` — size of a patch after scaling, px. The patch side before scaling is 1.25 × the longer side of the box.
- `BACKGROUND_PER_IMAGE` — number of background patches per image.
- `TIME_OF_DAY` — day images, night images or both.

**What to look at.** The number of patches per class (classes are not balanced, therefore macro-averaged metrics are used later); the gallery; the **mean patch** of every class — the average thermal signature that the classifier has to recognise.
""")

code(r"""
#@title 4.1 Data set of thermal patches (HIT-UAV) { display-mode: "form" }
N_IMAGES = 240 #@param {type:"slider", min:60, max:800, step:20}
PATCH_SIZE = 32 #@param [24, 32, 48] {type:"raw"}
BACKGROUND_PER_IMAGE = 4 #@param {type:"slider", min:0, max:12, step:1}
TIME_OF_DAY = "day and night" #@param ["day and night", "day only", "night only"]

CLASSES = ["Person", "Car", "Bicycle", "Background"]
pool = hit.images
if TIME_OF_DAY == "day only":
    pool = pool[~pool.night]
elif TIME_OF_DAY == "night only":
    pool = pool[pool.night]
pool = pool[(pool.Person + pool.Car + pool.Bicycle) > 0]
pick = np.random.default_rng(SEED)
n_train_img = int(round(0.75 * N_IMAGES))
train_pool, test_pool = pool[pool.split == "train"], pool[pool.split != "train"]
train_files = list(pick.choice(train_pool.file.values, size=min(n_train_img, len(train_pool)), replace=False))
test_files = list(pick.choice(test_pool.file.values, size=min(N_IMAGES - n_train_img, len(test_pool)), replace=False))
t0 = time.time()
hit.prefetch(train_files + test_files)


def square_crop(img, cx, cy, side, size):
    pad = int(side) + 2
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    x1, y1 = int(round(cx - side / 2)) + pad, int(round(cy - side / 2)) + pad
    s = int(round(side))
    return cv2.resize(big[y1:y1 + s, x1:x1 + s], (size, size), interpolation=cv2.INTER_AREA)


patches, labels, groups, subsets, nights = [], [], [], [], []
for subset, files in [("train", train_files), ("test", test_files)]:
    for f in files:
        img, objs = hit.load(f)
        info_f = hit.images.loc[hit.images.file == f].iloc[0]
        sides = []
        for o in objs:
            if o["cls"] not in CLASSES[:3]:
                continue
            x1, y1, x2, y2 = o["box"]
            side = max(8.0, 1.25 * max(x2 - x1, y2 - y1))
            sides.append(side)
            patches.append(square_crop(img, (x1 + x2) / 2, (y1 + y2) / 2, side, PATCH_SIZE))
            labels.append(o["cls"]); groups.append(f); subsets.append(subset); nights.append(bool(info_f.night))
        # background: random squares that do not touch any annotated object
        boxes = np.array([o["box"] for o in objs], float).reshape(-1, 4)
        side_bg = float(np.median(sides)) if sides else 24.0
        placed, tries = 0, 0
        while placed < BACKGROUND_PER_IMAGE and tries < 200:
            tries += 1
            cx, cy = pick.uniform(side_bg / 2, img.shape[1] - side_bg / 2), pick.uniform(side_bg / 2, img.shape[0] - side_bg / 2)
            m = side_bg / 2 + 4
            if len(boxes) and np.any((boxes[:, 0] < cx + m) & (boxes[:, 2] > cx - m) & (boxes[:, 1] < cy + m) & (boxes[:, 3] > cy - m)):
                continue
            patches.append(square_crop(img, cx, cy, side_bg, PATCH_SIZE))
            labels.append("Background"); groups.append(f); subsets.append(subset); nights.append(bool(info_f.night))
            placed += 1

PATCHES = np.stack(patches)
LABELS = np.array(labels)
GROUPS = np.array(groups)
SUBSET = np.array(subsets)
NIGHT = np.array(nights)
y_all = np.array([CLASSES.index(c) for c in LABELS])
train_idx, test_idx = np.where(SUBSET == "train")[0], np.where(SUBSET == "test")[0]
counts = pd.crosstab(pd.Categorical(LABELS, CLASSES), SUBSET).reindex(columns=["train", "test"])
counts.index.name = "Class"
counts.loc["All"] = counts.sum()
print(f"{len(train_files)} training and {len(test_files)} test images ({TIME_OF_DAY}), "
      f"{len(PATCHES)} patches {PATCH_SIZE} × {PATCH_SIZE} px, {time.time() - t0:.0f} s")
display(counts)

fig = plt.figure(figsize=(15, 6.2))
grid = fig.add_gridspec(4, 13, width_ratios=[1] * 10 + [0.3, 1.3, 1.3])
for r, cls in enumerate(CLASSES):
    idx = np.where(LABELS == cls)[0]
    for c, i in enumerate(pick.choice(idx, size=min(10, len(idx)), replace=False)):
        ax = fig.add_subplot(grid[r, c]); ax.imshow(PATCHES[i], cmap="inferno", vmin=0, vmax=255); ax.axis("off")
        if c == 0:
            ax.set_title(cls, loc="left", fontsize=10)
    ax = fig.add_subplot(grid[r, 11]); ax.imshow(PATCHES[idx].mean(0), cmap="inferno"); ax.axis("off")
    if r == 0:
        ax.set_title("mean patch", fontsize=10)
    ax = fig.add_subplot(grid[r, 12])
    ax.hist(PATCHES[idx].reshape(len(idx), -1).mean(1), bins=20, color=COLORS[r])
    ax.set_yticks([]); ax.tick_params(labelsize=7)
    if r == 0:
        ax.set_title("mean brightness", fontsize=10)
fig.suptitle("Thermal patches (white-hot: warm = bright)", y=1.0)
fig.tight_layout()
save_figure(fig, "4_1_patches.png")
""")

md(r"""
### 4.2. HOG + SVM classifier with a choice of hyperparameters

**Purpose.** Describe every patch by a **histogram of oriented gradients** (HOG) and train a support vector machine (SVM) with a radial basis kernel. The result is compared with an SVM trained on raw pixels.

**HOG.** The patch is divided into cells of $c \times c$ pixels; in every cell a histogram of the gradient orientations with `HOG_BINS` bins is built (weighted by the gradient magnitude); histograms of $2 \times 2$ neighbouring cells are normalised together (function `skimage.feature.hog`). The descriptor keeps the shape of the object and is insensitive to the absolute brightness — important for thermal images, where the brightness depends on the weather and the time of day.

**SVM.** The kernel $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \lVert \mathbf{x} - \mathbf{x}' \rVert^2)$; $C$ sets the penalty for training errors (large $C$ — a more complex boundary), $\gamma$ the width of the kernel (large $\gamma$ — a more local boundary).

**Procedure.** (1) The test images are put aside. (2) $C$ and $\gamma$ are chosen by grouped cross-validation on the training images: all patches of one image are in the same fold. (3) The chosen model is evaluated on the test images once.

**Metrics.** Accuracy is the share of correct answers; for every class precision $P = TP/(TP+FP)$, recall $R = TP/(TP+FN)$ and $F_1 = 2PR/(P+R)$; **macro** averaging takes the mean over classes, so that the rare class "Bicycle" weighs as much as "Person".

**Parameters.** `HOG_CELL` — cell size $c$, px; `HOG_BINS` — number of orientation bins; `CV_FOLDS` — number of folds.

**What to look at.** The map of the cross-validation score over $C$ and $\gamma$; the comparison HOG / pixels; the confusion matrix; the misclassified patches; the learning curve: how the quality grows with the number of training images.
""")

code(r"""
#@title 4.2 HOG + SVM classifier { display-mode: "form" }
HOG_CELL = 8 #@param [4, 8] {type:"raw"}
HOG_BINS = 9 #@param {type:"slider", min:6, max:18, step:3}
CV_FOLDS = 3 #@param {type:"slider", min:2, max:5, step:1}

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from skimage.feature import hog as skimage_hog

if PATCH_SIZE % HOG_CELL:
    raise ValueError("PATCH_SIZE must be a multiple of HOG_CELL.")
hog_features = lambda p: skimage_hog(p, orientations=HOG_BINS, pixels_per_cell=(HOG_CELL, HOG_CELL),
                                     cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
t0 = time.time()
FEATURES = {"HOG": np.stack([hog_features(p) for p in PATCHES]).astype(np.float32),
            "Pixels": PATCHES.reshape(len(PATCHES), -1).astype(np.float32) / 255.0}
print(f"Feature length: HOG {FEATURES['HOG'].shape[1]} | pixels {FEATURES['Pixels'].shape[1]}; "
      f"HOG of {len(PATCHES)} patches computed in {time.time() - t0:.1f} s")


def fit_svm(X, idx, folds=CV_FOLDS):
    g0 = 1.0 / X.shape[1]                                   # value of gamma = "scale" after standardisation
    search = GridSearchCV(Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", class_weight="balanced"))]),
                          {"svm__C": [0.5, 2.0, 8.0, 32.0], "svm__gamma": [0.25 * g0, g0, 4 * g0]},
                          scoring="f1_macro", cv=GroupKFold(n_splits=folds), n_jobs=-1)
    search.fit(X[idx], y_all[idx], groups=GROUPS[idx])
    return search


def scores(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0)}


t0 = time.time()
SEARCH, PRED, rows = {}, {}, []
for name, X in FEATURES.items():
    SEARCH[name] = fit_svm(X, train_idx)
    PRED[name] = SEARCH[name].predict(X[test_idx])
    row = {"Features": name, **scores(y_all[test_idx], PRED[name])}
    for flag, label in [(False, "F1 day"), (True, "F1 night")]:
        sel = NIGHT[test_idx] == flag
        row[label] = f1_score(y_all[test_idx][sel], PRED[name][sel], average="macro", zero_division=0) if sel.any() else np.nan
    best = SEARCH[name].best_params_
    row["Chosen C"], row["Chosen γ·d"] = best["svm__C"], best["svm__gamma"] * X.shape[1]
    rows.append(row)
CLASSIFICATION = pd.DataFrame(rows).set_index("Features")
display(CLASSIFICATION.round(3))
CLASSIFICATION.to_csv(os.path.join(OUT, "classification_metrics.csv"))
print(f"Grid search and test in {time.time() - t0:.0f} s; γ·d is γ multiplied by the feature length d")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1.1, 1]})
cv = pd.DataFrame(SEARCH["HOG"].cv_results_)
heat = cv.pivot_table(index="param_svm__C", columns="param_svm__gamma", values="mean_test_score")
im = axes[0].imshow(heat.values, cmap="viridis", origin="lower")
axes[0].set_xticks(range(3)); axes[0].set_xticklabels(["γ₀/4", "γ₀", "4γ₀"])
axes[0].set_yticks(range(len(heat.index))); axes[0].set_yticklabels(heat.index)
for (r, c), v in np.ndenumerate(heat.values):
    axes[0].text(c, r, f"{v:.3f}", ha="center", va="center", color="w" if v < heat.values.mean() else "k")
axes[0].set(xlabel="kernel width γ (γ₀ = 1/d)", ylabel="C", title="HOG + SVM: cross-validation F1 (macro)")
cm = confusion_matrix(y_all[test_idx], PRED["HOG"], labels=range(4), normalize="true")
axes[1].imshow(cm, cmap="Greens", vmin=0, vmax=1)
for (r, c), v in np.ndenumerate(cm):
    axes[1].text(c, r, f"{v:.2f}", ha="center", va="center", color="w" if v > 0.6 else "k")
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(CLASSES, rotation=20)
axes[1].set_yticks(range(4)); axes[1].set_yticklabels(CLASSES)
axes[1].set(xlabel="predicted class", ylabel="true class", title="HOG + SVM: normalised confusion matrix (test)")
rec = {n: recall_score(y_all[test_idx], PRED[n], labels=range(4), average=None, zero_division=0) for n in FEATURES}
xx = np.arange(4)
for k, (n, r) in enumerate(rec.items()):
    axes[2].bar(xx + (k - 0.5) * 0.38, r, width=0.38, color=COLORS[k], label=n)
axes[2].set_xticks(xx); axes[2].set_xticklabels(CLASSES); axes[2].set_ylim(0, 1.05)
axes[2].set(ylabel="recall", title="Recall per class: HOG against raw pixels"); axes[2].legend(loc="lower right")
fig.tight_layout()
save_figure(fig, "4_2_svm_results.png")

# what HOG sees, and which patches were misclassified
fig = plt.figure(figsize=(15, 4.6))
grid = fig.add_gridspec(2, 14)
for k, cls in enumerate(CLASSES):
    i = np.where(LABELS == cls)[0][0]
    _, hog_img = skimage_hog(PATCHES[i], orientations=HOG_BINS, pixels_per_cell=(HOG_CELL, HOG_CELL),
                             cells_per_block=(2, 2), visualize=True)
    ax = fig.add_subplot(grid[0, 2 * k]); ax.imshow(PATCHES[i], cmap="inferno"); ax.axis("off"); ax.set_title(cls, fontsize=9)
    ax = fig.add_subplot(grid[0, 2 * k + 1]); ax.imshow(hog_img, cmap="gray"); ax.axis("off"); ax.set_title("HOG", fontsize=9)
wrong = test_idx[PRED["HOG"] != y_all[test_idx]]
for k, i in enumerate(wrong[:14]):
    ax = fig.add_subplot(grid[1, k]); ax.imshow(PATCHES[i], cmap="inferno"); ax.axis("off")
    ax.set_title(f"{LABELS[i][:6]}→{CLASSES[PRED['HOG'][np.where(test_idx == i)[0][0]]][:6]}", fontsize=8)
fig.suptitle(f"Top: patches and their HOG (cell {HOG_CELL} px, {HOG_BINS} bins). Bottom: misclassified test patches "
             f"(true → predicted), {len(wrong)} of {len(test_idx)}", y=1.02)
save_figure(fig, "4_2_hog_and_errors.png")

# learning curve: test F1 against the number of training images
train_imgs = np.array(sorted(set(GROUPS[train_idx])))
fractions = [0.1, 0.2, 0.4, 0.7, 1.0]
curve = []
for frac in fractions:
    chosen = set(np.random.default_rng(SEED).choice(train_imgs, size=max(4, int(frac * len(train_imgs))), replace=False))
    sub = np.array([i for i in train_idx if GROUPS[i] in chosen])
    for name, X in FEATURES.items():
        p = SEARCH[name].best_params_
        model = Pipeline([("scaler", StandardScaler()),
                          ("svm", SVC(kernel="rbf", C=p["svm__C"], gamma=p["svm__gamma"], class_weight="balanced"))])
        model.fit(X[sub], y_all[sub])
        curve.append(dict(features=name, images=len(chosen), patches=len(sub),
                          f1=f1_score(y_all[test_idx], model.predict(X[test_idx]), average="macro")))
curve = pd.DataFrame(curve)
curve.to_csv(os.path.join(OUT, "learning_curve.csv"), index=False)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for (name, g), color in zip(curve.groupby("features", sort=False), COLORS):
    ax.plot(g.images, g.f1, marker="o", color=color, label=name)
ax.set(xlabel="number of training images", ylabel="test F1 (macro)", title="Learning curve of the SVM")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
save_figure(fig, "4_2_learning_curve.png")
""")

md(r"""
### 4.3. MobileNet: transfer learning (optional block)

**Purpose.** Use a convolutional network pre-trained on ImageNet (MobileNetV3 Small, 2.5 million parameters) as a feature extractor for the **same** thermal patches and the **same** split as in 4.2, and compare it with HOG + SVM.

Transfer learning: the convolutional layers that have learned general visual features (edges, textures, shapes) are frozen; only the last classification layer (or also the last blocks) is trained on the new classes. The grey patches are scaled to `INPUT_SIZE` and repeated in three channels.

**Parameters.** `RUN_MOBILENET` — switch the block on (downloads 10 MB of weights; a GPU is advisable, on a CPU one epoch takes 0.5–2 min); `EPOCHS`; `INPUT_SIZE`; `TRAINABLE` — which layers are trained; `LEARNING_RATE`; `EXPORT_ONNX` — save the model in the ONNX format for deployment on an on-board computer.

**What to look at.** The curves of the training loss and of the test F1 over the epochs; the comparison table with HOG + SVM.
""")

code(r"""
#@title 4.3 MobileNet transfer learning (optional) { display-mode: "form" }
RUN_MOBILENET = False #@param {type:"boolean"}
EPOCHS = 5 #@param {type:"slider", min:1, max:20, step:1}
INPUT_SIZE = 96 #@param [64, 96, 128, 160] {type:"raw"}
TRAINABLE = "classifier layer only" #@param ["classifier layer only", "classifier + last 2 blocks"]
LEARNING_RATE = 0.002 #@param {type:"number"}
EXPORT_ONNX = False #@param {type:"boolean"}

MOBILENET_RESULT = None
if RUN_MOBILENET:
    import torch
    from torch import nn
    from torchvision import models
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def to_tensor(idx):
        x = np.stack([cv2.resize(PATCHES[i], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR) for i in idx])
        x = torch.from_numpy(x).float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        return (x - mean) / std

    X_tr, X_te = to_tensor(train_idx), to_tensor(test_idx)
    y_tr, y_te = torch.from_numpy(y_all[train_idx]).long(), y_all[test_idx]
    network = models.mobilenet_v3_small(weights=weights)
    for p in network.parameters():
        p.requires_grad = False
    network.classifier[-1] = nn.Linear(network.classifier[-1].in_features, len(CLASSES))
    trainable = list(network.classifier.parameters())
    if TRAINABLE.startswith("classifier +"):
        for block in network.features[-2:]:
            for p in block.parameters():
                p.requires_grad = True
                trainable.append(p)
    for p in network.classifier.parameters():
        p.requires_grad = True
    network = network.to(device)
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE)
    weight = torch.tensor(len(y_tr) / (len(CLASSES) * np.bincount(y_tr.numpy(), minlength=len(CLASSES)).clip(1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device))      # balanced classes, as class_weight in the SVM

    def predict(X):
        network.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(X), 256):
                out.append(network(X[k:k + 256].to(device)).argmax(1).cpu())
        return torch.cat(out).numpy()

    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        network.train()
        network.features.eval()                                  # statistics of BatchNorm stay those of ImageNet
        perm = torch.randperm(len(X_tr))
        total = 0.0
        for k in range(0, len(perm), 64):
            b = perm[k:k + 64]
            xb, yb = X_tr[b], y_tr[b]
            flip = torch.rand(len(b)) < 0.5                      # augmentation: horizontal flip
            xb[flip] = xb[flip].flip(3)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(network(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        pred = predict(X_te)
        history.append(dict(epoch=epoch + 1, loss=total / len(X_tr), f1=f1_score(y_te, pred, average="macro"),
                            accuracy=accuracy_score(y_te, pred)))
        print(f"epoch {epoch + 1}: loss {history[-1]['loss']:.4f}, test F1 {history[-1]['f1']:.3f}, {time.time() - t0:.0f} s")
    history = pd.DataFrame(history)
    MOBILENET_PRED = predict(X_te)
    MOBILENET_RESULT = {"Features": f"MobileNetV3 Small ({TRAINABLE}, {INPUT_SIZE} px, {EPOCHS} epochs)",
                        **scores(y_te, MOBILENET_PRED)}
    comparison = pd.concat([CLASSIFICATION[list(scores(y_te, MOBILENET_PRED))].reset_index(),
                            pd.DataFrame([MOBILENET_RESULT])]).set_index("Features")
    display(comparison.round(3))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].plot(history.epoch, history.loss, marker="o", color=COLORS[0])
    axes[0].set(xlabel="epoch", ylabel="training loss", title="MobileNet: training loss")
    axes[1].plot(history.epoch, history.f1, marker="o", color=COLORS[1], label="MobileNet")
    axes[1].axhline(CLASSIFICATION.loc["HOG", "F1 (macro)"], color=COLORS[2], ls="--", label="HOG + SVM")
    axes[1].set(xlabel="epoch", ylabel="test F1 (macro)", title="MobileNet against HOG + SVM"); axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "4_3_mobilenet.png")
    network = network.cpu().eval()
    torch.save(network.state_dict(), os.path.join(DIRS["models"], "mobilenet_hituav_patches.pt"))
    if EXPORT_ONNX:
        try:
            torch.onnx.export(network, torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE), os.path.join(DIRS["models"], "mobilenet_hituav_patches.onnx"),
                              input_names=["images"], output_names=["logits"], opset_version=17,
                              dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}})
            print("ONNX model saved in", DIRS["models"])
        except Exception as error:
            print("ONNX export failed (install the packages onnx and onnxscript):", type(error).__name__, error)
else:
    print("The MobileNet block is switched off. Set RUN_MOBILENET in the form to run it.")
""")

md(r"""
**Task 4.**
1. Explain the difference between the accuracy and the macro-averaged F1 using the table of 4.1 (number of patches per class). Which class has the largest share of errors and with which class is it confused?
2. Why do HOG features work better than raw pixels on thermal patches? Compare the columns "F1 day" and "F1 night".
3. Run 4.1 and 4.2 with `TIME_OF_DAY = day only` and then with `night only`. Under which conditions are the thermal signatures easier to separate, and what does it mean for a model that was trained under one condition and used under the other?
4. Read the learning curve: would more labelled images improve the classifier noticeably?
""")
