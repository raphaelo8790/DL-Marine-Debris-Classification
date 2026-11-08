"""
Marine Debris Classification: try the model on your own photos.

    pip install gradio            (once)
    python app.py                 -> open http://127.0.0.1:7860 in your browser

Pick a trained model, upload a sea photo, and the app shows:
  - which viewpoint the model thinks it is (underwater / shore / drone / satellite)
  - whether there is trash in the photo
  - which debris classes it found, with their scores
  - the photo cut into tiles, each coloured by how likely it holds trash

How it works: the model was trained on TILES (pieces of photos), so the photo is cut the same
way the training photos were (underwater 3x2, shore 3x3, drone ~640 px pieces, satellite whole),
each tile is scored, and the photo gets the highest score of any of its tiles for each class.
"""
import os

# ---------------------------------------------------------------- settings
PROJECT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the "Marine Debris Classification" folder
RUNS     = os.path.join(PROJECT, "Models")                       # where the *_best.keras models are
EVALS    = os.path.join(RUNS, "Evaluations")                    # eval_<model>/thresholds.json
PREPARED = os.path.join(PROJECT, "Datasets", "prepared")        # optional classes.json
USE_GPU  = False   # False = run on the CPU (fine for single photos, and safe while training runs)
GPU      = "1"     # which GPU to use if USE_GPU is True

os.environ["KERAS_BACKEND"] = os.environ.get("KERAS_BACKEND", "torch")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU if USE_GPU else ""

import glob, json
import numpy as np
import cv2
import keras
import gradio as gr

INPUT = 320
HEADS = ["underwater", "shore", "drone", "satellite"]
VIEWS = ["underwater", "shore", "drone", "satellite"]          # viewpoint head output order
TRASH_CLASSES = {"satellite": ["Marine Debris", "Oil Spill"]}   # other heads: every class is trash
# Class names in each head's output order (the same list 1_prepare_data.py writes to classes.json).
# Built in so the app runs without the datasets; a classes.json in Datasets/prepared, if present, is used instead.
DEFAULT_CLASSES = {
    "underwater": [
        "tire_rubber",
        "rope",
        "bottle_glass",
        "tube_cement",
        "bottle_plastic",
        "net_plastic",
        "bag_plastic",
        "can_metal",
        "brick_clay",
        "other_waste",
        "other_plastic",
        "cup_plastic",
        "metal_debris",
        "clothing_fiber",
        "branch_wood",
        "snack_wrapper"
    ],
    "shore": [
        "fragment",
        "others",
        "styrene_foam",
        "pet_bottle",
        "buoy",
        "other_bottle",
        "plastic_bag",
        "rope",
        "fishing_net",
        "other_container",
        "box_shaped_case",
        "other_fishing_gear",
        "other_string"
    ],
    "drone": [
        "Plastic bottle",
        "Metal",
        "Plastic fragment",
        "Plastic container",
        "Glass",
        "Plastic bag",
        "Packet"
    ],
    "satellite": [
        "Marine Debris",
        "Oil Spill",
        "Oil Platform",
        "Ship",
        "Sargassum & Floating Algae",
        "Foam, Waves & Wakes",
        "Natural Organic Material"
    ],
    "viewpoint": [
        "underwater",
        "shore",
        "drone",
        "satellite"
    ]
}
_cls_file = os.path.join(PREPARED, "classes.json")
CLASSES = json.load(open(_cls_file, encoding="utf-8")) if os.path.exists(_cls_file) else DEFAULT_CLASSES
_models = {}


# ---------------------------------------------------------------- model helpers
def list_models():
    names = [os.path.basename(p)[:-len("_best.keras")] for p in glob.glob(os.path.join(RUNS, "*_best.keras"))]
    return sorted(names, key=lambda n: os.path.getmtime(os.path.join(RUNS, n + "_best.keras")), reverse=True)

def get_model(name):
    if name not in _models:
        _models[name] = keras.models.load_model(os.path.join(RUNS, f"{name}_best.keras"), compile=False)
    return _models[name]

def get_thresholds(name, head):
    """Per-class thresholds tuned on validation by 5_evaluate (0.5 if that model wasn't evaluated)."""
    p = os.path.join(EVALS, f"eval_{name}", "thresholds.json")
    if os.path.exists(p):
        t = json.load(open(p))
        return np.array([t[head].get(c, 0.5) for c in CLASSES[head]])
    return np.full(len(CLASSES[head]), 0.5)

def to_input(img):
    """RGB uint8 image of any size -> 320x320 float32 (the model scales pixels itself)."""
    interp = cv2.INTER_AREA if min(img.shape[:2]) > INPUT else cv2.INTER_LINEAR
    return cv2.resize(img, (INPUT, INPUT), interpolation=interp).astype(np.float32)

def tile_grid(view, w, h):
    """Same tiling as the training data for that viewpoint -> (columns, rows)."""
    if view == "underwater": return 3, 2
    if view == "shore":      return 3, 3
    if view == "drone":      return max(1, round(w / 640)), max(1, round(h / 640))
    return 1, 1                                             # satellite: whole patch

def cut(img, cols, rows):
    h, w = img.shape[:2]; tw, th = w / cols, h / rows
    boxes = [(int(c * tw), int(r * th), int((c + 1) * tw), int((r + 1) * th)) for r in range(rows) for c in range(cols)]
    return [img[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes], boxes


# ---------------------------------------------------------------- the prediction
def analyse(image, model_name, view_choice):
    if image is None:
        return None, {}, "Upload a photo first.", {}
    if not model_name:
        return None, {}, f"No model found in {RUNS}.", {}
    model = get_model(model_name)

    # 1) viewpoint, from the whole photo
    whole = model.predict(to_input(image)[None], verbose=0)
    vp = np.asarray(whole["viewpoint"])[0]
    view_probs = {v: float(p) for v, p in zip(VIEWS, vp)}
    view = VIEWS[int(vp.argmax())] if view_choice == "auto" else view_choice

    # 2) cut like the training data, score every tile with that viewpoint's head
    cols, rows = tile_grid(view, image.shape[1], image.shape[0])
    tiles, boxes = cut(image, cols, rows)
    out = model.predict(np.stack([to_input(t) for t in tiles]), verbose=0)
    P = np.asarray(out[view])                                      # tiles x classes
    cls = CLASSES[view]
    trash_cols = [cls.index(c) for c in TRASH_CLASSES.get(view, cls)]
    if f"{view}_trash" in out:                                     # v3 models: dedicated trash neuron
        tile_trash = np.asarray(out[f"{view}_trash"])[:, 0]; trash_source = "the model's trash neuron"
    else:                                                          # older models: combine class scores
        tile_trash = 1 - np.prod(1 - P[:, trash_cols], axis=1); trash_source = "combined class scores (noisy-OR)"

    # 3) photo level: a photo has a class if any tile has it
    photo = P.max(axis=0); thr = get_thresholds(model_name, view)
    found = [(c, float(s), float(t)) for c, s, t in zip(cls, photo, thr) if s >= t]
    trash_p = float(tile_trash.max())

    # 4) picture: tiles coloured by trash probability
    vis = image.copy(); overlay = image.copy()
    for (x0, y0, x1, y1), p, row in zip(boxes, tile_trash, P):
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (230, 50, 50), -1)
        a = 0.55 * float(p)
        vis[y0:y1, x0:x1] = cv2.addWeighted(overlay[y0:y1, x0:x1], a, vis[y0:y1, x0:x1], 1 - a, 0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), max(1, image.shape[1] // 400))
        top = cls[int(row.argmax())]
        scale = max(0.4, image.shape[1] / 1600)
        cv2.putText(vis, f"trash {p:.0%} | {top}", (x0 + 6, y0 + int(26 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)

    # 5) text summary
    verdict = "**Trash found**" if trash_p >= 0.5 else "**No trash found**"
    lines = [f"### {verdict}: probability {trash_p:.0%}",
             f"Viewpoint used: **{view}**" + (" (auto)" if view_choice == "auto" else " (chosen by you)") +
             f" · {cols}×{rows} tiles · trash score from {trash_source}"]
    if view == "satellite":
        lines.append("Satellite 'trash' = Marine Debris or Oil Spill.")
    if found:
        lines.append("| Class found | Score | Threshold |\n|:--|--:|--:|\n" +
                     "\n".join(f"| {c} | {s:.2f} | {t:.2f} |" for c, s, t in sorted(found, key=lambda x: -x[1])))
    else:
        lines.append("No class reached its threshold.")
    lines.append("*Thresholds come from the validation set (5_evaluate). "
                 "Photos that look nothing like the four training datasets can get odd viewpoints: override it on the left.*")
    class_scores = {c: float(s) for c, s in zip(cls, photo)}
    return vis, view_probs, "\n\n".join(l for l in lines if l != ""), class_scores


# ---------------------------------------------------------------- the page
with gr.Blocks(title="Marine Debris Classification") as demo:
    gr.Markdown("# Marine Debris Classification\nUpload a sea photo (underwater, beach, drone or satellite) "
                "and see what the model finds.")
    with gr.Row():
        with gr.Column(scale=1):
            models = list_models()
            model_dd = gr.Dropdown(models, value=models[0] if models else None, label="Model")
            view_dd = gr.Dropdown(["auto"] + VIEWS, value="auto", label="Viewpoint (auto = let the model decide)")
            img_in = gr.Image(type="numpy", label="Your photo")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=2):
            img_out = gr.Image(label="Tiles coloured by trash probability")
            summary = gr.Markdown()
            with gr.Row():
                view_lbl = gr.Label(label="Viewpoint", num_top_classes=4)
                cls_lbl = gr.Label(label="Class scores (photo)", num_top_classes=8)
    btn.click(analyse, [img_in, model_dd, view_dd], [img_out, view_lbl, summary, cls_lbl])

if __name__ == "__main__":
    demo.launch()
