"""
G10 Marine Debris: data preparation.

Builds  <ROOT>/prepared/  from the four original datasets:
    prepared/images/<dataset>/<split>/...   tiles (352x352 JPEG) / MADOS patches (240x240 PNG)
    prepared/manifests/{train,val,test}.csv one row per image: path, source, split, clean, labels, ...
    prepared/classes.json                   the class list (= output neuron order) of every head
    prepared/report.txt                     counts per class / split, warnings

Run (Windows, in the G10 environment):
    python 1_prepare_data.py                  # everything
    python 1_prepare_data.py --limit 20       # quick test: 20 images per dataset per split
    python 1_prepare_data.py --only mados     # one dataset (seaclear, bepli, ucwd, mados)

Safe to re-run: tiles that already exist are not written again.

Relabel SeaClear only (after the v3 labelling-rule change; images are reused, only labels change):
    python 1_prepare_data.py --only seaclear
"""
import os, re, sys, csv, json, math, time, random, argparse, collections
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageOps

# ============================================================================
# SETTINGS
# ============================================================================
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datasets")   # <project>/Datasets
DIRS = {
    "seaclear": "Seaclear Marine Debris Dataset (underwater)",
    "bepli":    os.path.join("BePLi_dataset_v2 (shore)", "plastic_coco"),
    "ucwd":     os.path.join("UCWD_v1 (drone)", "UCWD.coco"),
    "mados":    "MADOS (satellite)",
}
SAVE_SIZE   = 352      # saved tile size (model input is 320; ~10% margin for zoom/crop augmentation)
OVERLAP     = 0.30     # object counts for a tile if >= 30% of it is inside ...
# ... OR (v3 rule, datasets in VISIBLE_RULE) its visible part covers >= VISIBLE_MIN of the tile.
# Why: with the 30% rule alone, a big object (a net spanning the photo) puts < 30% of itself in
# every tile, so every tile showing it was labelled CLEAN. 20% of SeaClear's "clean" tiles showed trash.
VISIBLE_RULE = {"seaclear"}   # BePLi barely has this issue (4.7%), UCWD/MADOS unchanged
VISIBLE_MIN  = 0.02    # visible part >= 2% of the tile area -> the tile gets the label
SLIVER_MIN   = 0.001   # a piece between 0.1% and 2% of the tile is ambiguous: a tile with only such
                       # pieces (or an unknown object) is DROPPED, neither "trash" nor "clean"
JPEG_QUALITY = 95      # + no colour subsampling (sharper, no colour blocks)
SEED        = 42

# SeaClear video-block split
SC_SPLIT    = {"train": 0.70, "val": 0.15, "test": 0.15}
SC_BLOCK    = 50       # consecutive frames per block
SC_GAP      = 2        # frames dropped at a block boundary between different splits
SC_TRIES    = 300      # random block assignments tried; the most class-balanced one wins

# UCWD: sources dropped (not coastal / not aerial) and our own leak-free re-split
UCWD_DROP_PREFIXES = ("BATCH", "GOPR", "PHOTO", "CAMERA")   # parks, grass, concrete, handheld ground shots
UCWD_DROP_CD_SIZE  = (3024, 3024)                            # CD handheld close-ups (not drone)
UCWD_SPLIT  = {"train": 0.75, "val": 0.125, "test": 0.125}
UCWD_TRIES  = 300

BEPLI_GRID = (3, 3)    # 3x3 tiles: objects ~1.5x bigger than 2x2 (median ~18 px at the 320 input)

MADOS_TRAIN_COPIES = 8 # 4 rotations x flip (training only)

VIEWPOINT = {"seaclear": "underwater", "bepli": "shore", "ucwd": "drone", "mados": "satellite"}

# ============================================================================
# CLASS LISTS  (order = output neuron order; keep in sync with the sheet)
# ============================================================================
CLASSES = {
    "underwater": ["tire_rubber", "rope", "bottle_glass", "tube_cement", "bottle_plastic",
                   "net_plastic", "bag_plastic", "can_metal", "brick_clay", "other_waste",
                   "other_plastic", "cup_plastic", "metal_debris", "clothing_fiber",
                   "branch_wood", "snack_wrapper"],
    "shore":      ["fragment", "others", "styrene_foam", "pet_bottle", "buoy", "other_bottle",
                   "plastic_bag", "rope", "fishing_net", "other_container", "box_shaped_case",
                   "other_fishing_gear", "other_string"],
    "drone":      ["Plastic bottle", "Metal", "Plastic fragment", "Plastic container", "Glass",
                   "Plastic bag", "Packet"],
    "satellite":  ["Marine Debris", "Oil Spill", "Oil Platform", "Ship",
                   "Sargassum & Floating Algae", "Foam, Waves & Wakes", "Natural Organic Material"],
    "viewpoint":  ["underwater", "shore", "drone", "satellite"],
}

# SeaClear: original class -> final class (None = not trash, ignored)
SEACLEAR_MERGE = {
    "rope_fiber": "rope", "rope_plastic": "rope",
    "snack_wrapper_plastic": "snack_wrapper", "snack_wrapper_paper": "snack_wrapper",
    "branch_wood": "branch_wood", "furniture_wood": "branch_wood",
    "cable_metal": "metal_debris", "wreckage_metal": "metal_debris",
    "lid_plastic": "other_plastic", "tarp_plastic": "other_plastic", "sanitaries_plastic": "other_plastic",
    "container_plastic": "other_plastic", "pipe_plastic": "other_plastic",
    "cardboard_paper": "other_waste", "jar_glass": "other_waste", "container_middle_size_metal": "other_waste",
    "cup_ceramic": "other_waste", "boot_rubber": "other_waste",
}
AMBIGUOUS = "__ambiguous__"   # object we cannot call trash or not-trash (SeaClear "unknown_instance")

def seaclear_map(name):
    if name == "unknown_instance":
        return AMBIGUOUS                  # a tile showing only this is dropped (v3)
    if name.startswith("animal") or name.startswith("rov") or name == "plant":
        return None                       # not trash: a tile with only these is correctly CLEAN
    return SEACLEAR_MERGE.get(name, name)

# MADOS: mask value -> final class (None = background / dropped)
MADOS_MAP = {1: "Marine Debris", 2: "Sargassum & Floating Algae", 3: "Sargassum & Floating Algae",
             4: "Natural Organic Material", 5: "Ship", 6: "Oil Spill", 9: "Foam, Waves & Wakes",
             12: "Foam, Waves & Wakes", 13: "Oil Platform", 15: "Natural Organic Material"}

OVERWRITE = os.environ.get("G10_OVERWRITE") == "1"   # set by --overwrite (env var so worker processes see it)

HEAD = {"seaclear": "underwater", "bepli": "shore", "ucwd": "drone", "mados": "satellite"}


# ============================================================================
# HELPERS
# ============================================================================
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def load_coco(path, mapper):
    """-> list of dicts {file, W, H, anns:[(class, (x,y,w,h), polygons)], amb:[(box, polygons)]}.
    Classes already mapped. `polygons` = the object's outline(s) if the dataset has them, else [].
    `amb` holds ambiguous objects (mapper returned AMBIGUOUS); they never count as a class."""
    d = json.load(open(path, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by, amb = collections.defaultdict(list), collections.defaultdict(list)
    for a in d["annotations"]:
        cls = mapper(cats[a["category_id"]])
        if cls is None or a["bbox"][2] <= 0 or a["bbox"][3] <= 0:
            continue
        seg = a.get("segmentation")
        polys = [p for p in seg if len(p) >= 6] if isinstance(seg, list) else []   # RLE masks -> use the box
        if cls == AMBIGUOUS:
            amb[a["image_id"]].append((tuple(a["bbox"]), polys))
        else:
            by[a["image_id"]].append((cls, tuple(a["bbox"]), polys))
    return [{"file": i["file_name"], "W": i["width"], "H": i["height"], "anns": by[i["id"]], "amb": amb[i["id"]]}
            for i in d["images"]]

def _clip(poly, x0, y0, x1, y1):
    """Cut a polygon [(x,y),...] to the rectangle (Sutherland-Hodgman). Returns the inside part."""
    def cut(pts, inside, cross):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev): out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out
    def at_x(x):
        return lambda p, q: (x, p[1] + (q[1] - p[1]) * (x - p[0]) / ((q[0] - p[0]) or 1e-12))
    def at_y(y):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (y - p[1]) / ((q[1] - p[1]) or 1e-12), y)
    pts = poly
    for inside, cross in ((lambda p: p[0] >= x0, at_x(x0)), (lambda p: p[0] <= x1, at_x(x1)),
                          (lambda p: p[1] >= y0, at_y(y0)), (lambda p: p[1] <= y1, at_y(y1))):
        if not pts: break
        pts = cut(pts, inside, cross)
    return pts

def _area(pts):
    if len(pts) < 3: return 0.0
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))) / 2

def object_inside(box, polys, x0, y0, tw, th, use_polys):
    """-> (area of the object, area of the object inside the tile). Uses the outline when allowed."""
    if use_polys and polys:
        rings = [list(zip(p[0::2], p[1::2])) for p in polys]
        total = sum(_area(r) for r in rings)
        inside = sum(_area(_clip(r, x0, y0, x0 + tw, y0 + th)) for r in rings)
        if total > 0:
            return total, inside
    x, y, w, h = box
    ix = max(0.0, min(x + w, x0 + tw) - max(x, x0))
    iy = max(0.0, min(y + h, y0 + th) - max(y, y0))
    return w * h, ix * iy

def tile_labels(anns, x0, y0, tw, th, visible_rule=False, amb=()):
    """-> (set of classes in the tile, ambiguous?).
    Old rule (all datasets): an object counts if >= OVERLAP of its box is inside the tile.
    v3 rule (datasets in VISIBLE_RULE): measured on the object's outline, and it also counts if its
    visible part covers >= VISIBLE_MIN of the tile. A tile with no class but a sliver of trash
    (SLIVER_MIN..VISIBLE_MIN of the tile) or an unknown object is ambiguous (the caller drops it)."""
    present, sliver, tile_area = set(), False, tw * th
    for cls, box, polys in anns:
        total, inside = object_inside(box, polys, x0, y0, tw, th, visible_rule)
        if total <= 0 or inside <= 0:
            continue
        if inside >= OVERLAP * total or (visible_rule and inside >= VISIBLE_MIN * tile_area):
            present.add(cls)
        elif visible_rule and inside >= SLIVER_MIN * tile_area:
            sliver = True
    if not visible_rule or present:
        return present, False
    for box, polys in amb:
        total, inside = object_inside(box, polys, x0, y0, tw, th, True)
        if total > 0 and (inside >= OVERLAP * total or inside >= VISIBLE_MIN * tile_area):
            return present, True
    return present, sliver


# ============================================================================
# WORKERS (run in parallel processes)
# ============================================================================
def tile_image(job):
    """Cut one image into tiles, label and save them. Returns (rows, warnings)."""
    src, split, img_path, rel_orig, out_dir, stem, W, H, anns, grid, amb = job
    rows, warns, dropped = [], [], 0
    visible_rule = src in VISIBLE_RULE
    try:
        im = Image.open(img_path)
        im.load()
    except Exception as e:
        return rows, [f"{src}: cannot open {rel_orig}: {e}"], 0
    if im.size != (W, H):
        t = ImageOps.exif_transpose(im)
        if t.size == (W, H):
            im = t
        else:
            warns.append(f"{src}: size {im.size} != annotation {(W, H)} for {rel_orig} (boxes rescaled)")
    im = im.convert("RGB")
    sx, sy = im.width / W, im.height / H
    if (sx, sy) != (1.0, 1.0):
        def scale_polys(polys):
            return [[v * (sx if k % 2 == 0 else sy) for k, v in enumerate(p)] for p in polys]
        anns = [(c, (x * sx, y * sy, w * sx, h * sy), scale_polys(pp)) for c, (x, y, w, h), pp in anns]
        amb = [((x * sx, y * sy, w * sx, h * sy), scale_polys(pp)) for (x, y, w, h), pp in amb]
    iw, ih = im.size
    cols, nrows = grid if grid else (max(1, round(iw / 640)), max(1, round(ih / 640)))
    tw, th = iw / cols, ih / nrows
    os.makedirs(out_dir, exist_ok=True)
    for r in range(nrows):
        for c in range(cols):
            x0, y0 = c * tw, r * th
            labels, ambiguous = tile_labels(anns, x0, y0, tw, th, visible_rule, amb)
            if ambiguous:
                dropped += 1          # neither trash nor clean: left out of every split
                continue
            name = f"{stem}_r{r}c{c}.jpg"
            out = os.path.join(out_dir, name)
            if OVERWRITE or not os.path.exists(out):
                box = (int(round(x0)), int(round(y0)), int(round(x0 + tw)), int(round(y0 + th)))
                t = im.crop(box)
                # big tiles: shrink to SAVE_SIZE (high-quality filter). Small tiles: keep native pixels,
                # no upscaling here (the loader resizes to 320 once, so there is only one resample).
                if t.width >= SAVE_SIZE and t.height >= SAVE_SIZE:
                    t = t.resize((SAVE_SIZE, SAVE_SIZE), Image.LANCZOS)
                t.save(out, quality=JPEG_QUALITY, subsampling=0)
            rows.append({"path": f"{src}/{split}/{name}", "source": src, "split": split,
                         "clean": int(not labels), "labels": ";".join(sorted(labels)),
                         "orig": rel_orig, "tile": f"r{r}c{c}", "copy": 0})
    return rows, warns, dropped


def read_mask(path):
    try:
        import tifffile
        return np.asarray(tifffile.imread(path))
    except Exception:
        return np.asarray(Image.open(path))

def mados_patch(job):
    split, scene, n, base, out_dir = job
    rgb = os.path.join(base, scene, "10", f"{scene}_L2R_rgb_{n}.png")
    cl = os.path.join(base, scene, "10", f"{scene}_L2R_cl_{n}.tif")
    rows = []
    try:
        img = np.asarray(Image.open(rgb).convert("RGB"))
        vals = set(int(v) for v in np.unique(read_mask(cl)))
    except Exception as e:
        return rows, [f"mados: cannot read {scene}_{n}: {e}"], 0
    labels = sorted({MADOS_MAP[v] for v in vals if v in MADOS_MAP})
    copies = MADOS_TRAIN_COPIES if split == "train" else 1
    os.makedirs(out_dir, exist_ok=True)
    for k in range(copies):
        arr = np.rot90(img, k % 4)
        if k >= 4:
            arr = np.fliplr(arr)
        name = f"{scene}_{n}_d{k}.png"
        out = os.path.join(out_dir, name)
        if OVERWRITE or not os.path.exists(out):
            Image.fromarray(np.ascontiguousarray(arr)).save(out)
        rows.append({"path": f"mados/{split}/{name}", "source": "mados", "split": split,
                     "clean": int(not labels), "labels": ";".join(labels),
                     "orig": f"{scene}/10/{scene}_L2R_rgb_{n}.png", "tile": "", "copy": k})
    return rows, [], 0


# ============================================================================
# SPLITS
# ============================================================================

def balanced_group_split(groups, shares, classes, tries, seed=SEED):
    """groups: list of (size, Counter of classes). Assign whole groups to splits so sizes match
    the shares and every class is represented in val/test as well as possible.
    Returns (list of split names, score). score = worst class coverage ratio in val/test."""
    total = sum(g[0] for g in groups)
    tot_cls = collections.Counter()
    for _, c in groups: tot_cls.update(c)
    best, best_score = None, -1.0
    rng = random.Random(seed)
    for _ in range(tries):
        order = list(range(len(groups))); rng.shuffle(order)
        size = {k: 0 for k in shares}; cls_in = {k: collections.Counter() for k in shares}
        assign = [None] * len(groups)
        for gi in order:
            k = max(shares, key=lambda k: (shares[k] * total - size[k]) / (shares[k] * total))
            assign[gi] = k; size[k] += groups[gi][0]; cls_in[k].update(groups[gi][1])
        score = min(cls_in[k][c] / max(1e-9, shares[k] * tot_cls[c])
                    for k in shares if k != "train" for c in classes if tot_cls[c] > 0)
        if score > best_score:
            best, best_score = assign, score
    return best, best_score

def seaclear_sequence(folder, fname):
    """Which video sequence a SeaClear frame belongs to."""
    if ".mp4_" in fname:
        return folder + "|" + fname.split(".mp4_")[0]
    m = re.match(r"^(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_.+)_\d{2}-\d{2}-\d{2}\.jpg$", fname)
    if m:
        return folder + "|" + m.group(1)
    return folder + "|numbered"

def seaclear_split(images, index, log):
    """Video-block split: consecutive frames stay together; returns {file: split or None}."""
    seqs = collections.defaultdict(list)
    for im in images:
        if im["file"] in index:
            folder = os.path.dirname(index[im["file"]])
            seqs[seaclear_sequence(folder, im["file"])].append(im)
    blocks = []
    for key, ims in seqs.items():
        ims.sort(key=lambda d: natural_key(d["file"]))
        for i in range(0, len(ims), SC_BLOCK):
            blk = ims[i:i + SC_BLOCK]
            cnt = collections.Counter(c for d in blk for c in {a[0] for a in d["anns"]})
            blocks.append({"seq": key, "pos": i // SC_BLOCK, "ims": blk, "cnt": cnt})
    assign, best_score = balanced_group_split([(len(b["ims"]), b["cnt"]) for b in blocks],
                                              SC_SPLIT, CLASSES["underwater"], SC_TRIES)
    best = dict(enumerate(assign))

    split_of = {}
    by_seq = collections.defaultdict(list)
    for bi, b in enumerate(blocks):
        by_seq[b["seq"]].append(bi)
        for d in b["ims"]:
            split_of[d["file"]] = best[bi]
    dropped = 0
    for key, bis in by_seq.items():
        bis.sort(key=lambda i: blocks[i]["pos"])
        for a, b in zip(bis, bis[1:]):
            if best[a] != best[b]:
                for d in blocks[a]["ims"][-SC_GAP:]:
                    split_of[d["file"]] = None; dropped += 1
    log.append(f"SeaClear video-block split: {len(seqs)} sequences, {len(blocks)} blocks, "
               f"worst class coverage ratio in val/test = {best_score:.2f}, gap frames dropped = {dropped}")
    return split_of

def ucwd_source(fname):
    return re.split(r"[_\-\s.]", fname.split(".rf.")[0])[0].upper()

def ucwd_resplit(items, log):
    """items: list of image dicts from all three official splits (official split ignored).
    Drops non-coastal / non-aerial sources, then re-splits: frames of one video stay together."""
    kept, dropped = [], collections.Counter()
    for im in items:
        src = ucwd_source(im["file"])
        if src.startswith(UCWD_DROP_PREFIXES):
            dropped[src.rstrip("0123456789")] += 1; continue
        if src == "CD" and (im["W"], im["H"]) == UCWD_DROP_CD_SIZE:
            dropped["CD 3024x3024"] += 1; continue
        kept.append(im)
    groups = collections.defaultdict(list)
    for im in kept:
        m = re.match(r"(VID_\d+_\d+)", im["file"].split(".rf.")[0])
        groups[m.group(1) if m else "img:" + im["file"]].append(im)
    keys = list(groups)
    g = [(len(groups[k]), collections.Counter(c for im in groups[k] for c in {a[0] for a in im["anns"]})) for k in keys]
    assign, score = balanced_group_split(g, UCWD_SPLIT, CLASSES["drone"], UCWD_TRIES)
    out = [(assign[i], im) for i, k in enumerate(keys) for im in groups[k]]
    cnt = collections.Counter(s for s, _ in out)
    log.append(f"UCWD: dropped {sum(dropped.values())} images ({dict(dropped)}); "
               f"re-split {len(kept)} images in {len(keys)} groups (videos kept whole): "
               f"train {cnt['train']} / val {cnt['val']} / test {cnt['test']}, "
               f"worst class coverage ratio in val/test = {score:.2f}")
    return out


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--only", default="seaclear,bepli,ucwd,mados")
    ap.add_argument("--limit", type=int, default=0, help="max images per dataset per split (quick test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--overwrite", action="store_true", help="re-write tiles that already exist")
    args = ap.parse_args()
    if args.overwrite:
        os.environ["G10_OVERWRITE"] = "1"   # inherited by the worker processes
    root = args.root
    out_root = os.path.join(root, "prepared")
    img_root = os.path.join(out_root, "images")
    only = set(args.only.split(","))
    log, jobs, mjobs = [], [], []
    t0 = time.time()

    def lim(lst):
        return lst[:args.limit] if args.limit else lst

    # ---------------- SeaClear ----------------
    if "seaclear" in only:
        base = os.path.join(root, DIRS["seaclear"])
        index = {}
        for dp, _, fs in os.walk(base):
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    index[f] = os.path.relpath(os.path.join(dp, f), base)
        images = load_coco(os.path.join(base, "dataset.json"), seaclear_map)
        missing = sum(1 for im in images if im["file"] not in index)
        if missing: log.append(f"SeaClear: {missing} annotated images not found on disk (skipped)")
        split_of = seaclear_split(images, index, log)
        per = collections.defaultdict(list)
        for im in images:
            s = split_of.get(im["file"])
            if s: per[s].append(im)
        for s, ims in per.items():
            for im in lim(ims):
                rel = index[im["file"]]
                jobs.append(("seaclear", s, os.path.join(base, rel), rel, os.path.join(img_root, "seaclear", s),
                             os.path.splitext(im["file"])[0], im["W"], im["H"], im["anns"], (3, 2), im["amb"]))

    # ---------------- BePLi ----------------
    if "bepli" in only:
        base = os.path.join(root, DIRS["bepli"])
        for s, fn in (("train", "train.json"), ("val", "val.json"), ("test", "test.json")):
            ims = load_coco(os.path.join(base, "annotation", fn), lambda n: n)
            for im in lim(ims):
                p = os.path.join(base, "images", "original_images", im["file"])
                if not os.path.exists(p):
                    p = os.path.join(base, "images", s, im["file"])
                jobs.append(("bepli", s, p, os.path.relpath(p, base), os.path.join(img_root, "bepli", s),
                             os.path.splitext(im["file"])[0], im["W"], im["H"], im["anns"], BEPLI_GRID, im["amb"]))

    # ---------------- UCWD ----------------
    if "ucwd" in only:
        base = os.path.join(root, DIRS["ucwd"])
        items = []
        for folder in ("train", "valid", "test"):
            for im in load_coco(os.path.join(base, folder, "_annotations.coco.json"),
                                lambda n: None if n == "obj" else n):
                im["folder"] = folder
                items.append(im)
        per = collections.defaultdict(list)
        for s, im in ucwd_resplit(items, log):
            per[s].append(im)
        for s, ims in per.items():
            for im in lim(ims):
                p = os.path.join(base, im["folder"], im["file"])
                stem = im["file"].split(".rf.")[0] + "_" + im["file"].split(".rf.")[-1][:6]
                jobs.append(("ucwd", s, p, os.path.relpath(p, base), os.path.join(img_root, "ucwd", s),
                             stem, im["W"], im["H"], im["anns"], None, im["amb"]))

    # ---------------- MADOS ----------------
    if "mados" in only:
        base = os.path.join(root, DIRS["mados"])
        for s in ("train", "val", "test"):
            ids = [l.strip() for l in open(os.path.join(base, "splits", f"{s}_X.txt")) if l.strip()]
            for pid in lim(ids):
                scene, n = pid.rsplit("_", 1)
                mjobs.append((s, scene, n, base, os.path.join(img_root, "mados", s)))

    # ---------------- run ----------------
    print(f"Jobs: {len(jobs)} images to tile, {len(mjobs)} MADOS patches | workers: {args.workers}")
    rows, warns, n_dropped = [], [], collections.Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(tile_image, j) for j in jobs] + [ex.submit(mados_patch, j) for j in mjobs]
        for k, f in enumerate(as_completed(futs), 1):
            r, w, dr = f.result(); rows += r; warns += w
            if dr: n_dropped["tiles"] += dr
            if k % 250 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"  {k}/{len(futs)} done  ({el/60:.1f} min, ~{el/k*(len(futs)-k)/60:.1f} min left)", flush=True)

    # ---------------- write manifests ----------------
    os.makedirs(os.path.join(out_root, "manifests"), exist_ok=True)
    fields = ["path", "source", "split", "viewpoint", "clean", "labels", "orig", "tile", "copy"]
    # --only run: keep the other datasets' rows from the existing manifests (full runs only, not --limit tests)
    if only != {"seaclear", "bepli", "ucwd", "mados"} and not args.limit:
        for s in ("train", "val", "test"):
            p = os.path.join(out_root, "manifests", f"{s}.csv")
            if os.path.exists(p):
                rows += [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["source"] not in only]
        for r in rows:
            r["clean"] = int(r["clean"])
    rows.sort(key=lambda r: (r["source"], r["split"], natural_key(r["path"])))
    for s in ("train", "val", "test"):
        with open(os.path.join(out_root, "manifests", f"{s}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in rows:
                if r["split"] == s:
                    r["viewpoint"] = VIEWPOINT[r["source"]]; w.writerow(r)
    json.dump(CLASSES, open(os.path.join(out_root, "classes.json"), "w", encoding="utf-8"), indent=2)

    # ---------------- report ----------------
    rep = [f"G10 data preparation report  ({time.strftime('%Y-%m-%d %H:%M')}, {(time.time()-t0)/60:.1f} min)",
           f"limit per dataset per split: {args.limit or 'none (full run)'}", ""]
    if VISIBLE_RULE & only:
        log.append(f"v3 labelling rule for {sorted(VISIBLE_RULE & only)}: object counts if >= {OVERLAP:.0%} of its outline "
                   f"is inside OR it covers >= {VISIBLE_MIN:.0%} of the tile; ambiguous tiles dropped: {n_dropped['tiles']}")
    rep += log + [""]
    for src in ("seaclear", "bepli", "ucwd", "mados"):
        sr = [r for r in rows if r["source"] == src]
        if not sr: continue
        rep.append(f"=== {src} ({HEAD[src]} head) ===")
        rep.append(f"{'split':6s} {'images':>8s} {'clean':>8s} {'clean %':>8s}")
        for s in ("train", "val", "test"):
            ss = [r for r in sr if r["split"] == s]
            if ss:
                cl = sum(r["clean"] for r in ss)
                rep.append(f"{s:6s} {len(ss):8d} {cl:8d} {100*cl/len(ss):7.1f}%")
        rep.append(f"\n{'class':28s} {'train':>7s} {'val':>7s} {'test':>7s}")
        cnt = {s: collections.Counter() for s in ("train", "val", "test")}
        for r in sr:
            for c in filter(None, r["labels"].split(";")):
                cnt[r["split"]][c] += 1
        for c in CLASSES[HEAD[src]]:
            flag = "   <-- missing in a split" if min(cnt[s][c] for s in cnt) == 0 else ""
            rep.append(f"{c:28s} {cnt['train'][c]:7d} {cnt['val'][c]:7d} {cnt['test'][c]:7d}{flag}")
        unknown = set(c for s in cnt for c in cnt[s]) - set(CLASSES[HEAD[src]])
        if unknown: rep.append(f"!! labels not in class list: {sorted(unknown)}")
        rep.append("")
    rep.append(f"Warnings: {len(warns)}")
    rep += ["  " + w for w in warns[:200]]
    if len(warns) > 200: rep.append(f"  ... {len(warns)-200} more")
    text = "\n".join(rep)
    open(os.path.join(out_root, "report.txt"), "w", encoding="utf-8").write(text)
    print("\n" + text)
    print(f"\nDone. Output: {out_root}")


if __name__ == "__main__":
    main()
