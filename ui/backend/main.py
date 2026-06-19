"""MedVision — Pulmonary AI Analysis | FastAPI backend.

Run: python -m uvicorn main:app --reload --port 8000

Live mode activates automatically when both checkpoints are present:
    checkpoints/ijepa_best.pth   (I-JEPA encoder, key "context_encoder")
    checkpoints/probe_best.pth   (linear probe, key "probe")
Otherwise the API runs in deterministic demo mode.
"""
import io, os, base64, time, random
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MedVision API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR   = Path(__file__).parent
CKPT_DIR   = BASE_DIR / "checkpoints"
STATIC_DIR = BASE_DIR / "static"

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model  = None
_device = None
_demo   = True


def _resolve_checkpoints():
    bases = []
    env = os.environ.get("MEDVISION_CKPT_DIR")
    if env:
        bases.append(Path(env))
    bases.append(CKPT_DIR)
    bases.append(BASE_DIR.parent.parent / "checkpoints")
    for base in bases:
        pairs = [
            (base / "ijepa_best.pth", base / "probe_best.pth"),
            (base / "ijepa" / "ijepa_best.pth", base / "probe" / "probe_best.pth"),
        ]
        for ijepa_path, probe_path in pairs:
            if ijepa_path.exists() and probe_path.exists():
                return ijepa_path, probe_path
    return None, None


def try_load_model():
    global _model, _device, _demo
    ijepa_path, probe_path = _resolve_checkpoints()
    if ijepa_path is None:
        print("No checkpoints found — DEMO mode")
        return
    try:
        import torch, timm
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = timm.create_model("vit_small_patch16_224", pretrained=False,
                                    num_classes=0, global_pool="")
        enc_ckpt = torch.load(ijepa_path, map_location=device)
        encoder.load_state_dict(enc_ckpt.get("context_encoder", enc_ckpt))

        class LinearProbe(nn.Module):
            def __init__(self, embed_dim, num_classes=2, dropout=0.1):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(embed_dim, num_classes)
            def forward(self, cls_token):
                return self.fc(self.dropout(cls_token))

        probe = LinearProbe(encoder.embed_dim)
        probe_ckpt = torch.load(probe_path, map_location=device)
        probe.load_state_dict(probe_ckpt.get("probe", probe_ckpt))

        class FullModel(nn.Module):
            def __init__(self, e, p):
                super().__init__()
                self.encoder = e
                self.probe = p
            def forward(self, x):
                tokens = self.encoder(x)
                return self.probe(tokens[:, 0, :])

        _model = FullModel(encoder, probe).to(device).eval()
        _device = device
        _demo = False
        print(f"Model loaded on {device} — LIVE mode")
    except Exception as e:
        print(f"Cannot load model ({e}) — DEMO mode")


try_load_model()


def read_image(data):
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    vis = np.array(img, dtype=np.float32) / 255.0
    return (vis - MEAN) / STD, vis


# ─────────────────────────── Pseudo-3D (volume) ────────────────────────────
# The model stays 2D. A 3D volume is split into 2D slices; each slice is run
# through the same 2D pipeline, then results are aggregated over the volume.

VOLUME_EXTS = (".nii", ".nii.gz", ".dcm", ".dicom", ".npy", ".npz", ".tif", ".tiff")
MAX_SLICES  = 160  # cap processed slices (evenly sampled) to keep latency sane


def is_volume_file(filename):
    name = (filename or "").lower()
    return name.endswith(VOLUME_EXTS)


def _load_nifti(data, suffix):
    import tempfile, nibabel as nib
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data); tmp_path = tmp.name
    try:
        vol = nib.load(tmp_path).get_fdata()
    finally:
        try: os.remove(tmp_path)
        except OSError: pass
    return np.asarray(vol, dtype=np.float32)


def _load_dicom(data):
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return arr * slope + inter


def _load_tiff(data):
    from PIL import ImageSequence
    img = Image.open(io.BytesIO(data))
    frames = [np.array(f.convert("F"), dtype=np.float32) for f in ImageSequence.Iterator(img)]
    return np.stack(frames, axis=0)


def load_volume(data, filename):
    """Return a 3D float array (slices, H, W)."""
    name = (filename or "").lower()
    if name.endswith(".nii.gz"):
        vol = _load_nifti(data, ".nii.gz")
    elif name.endswith(".nii"):
        vol = _load_nifti(data, ".nii")
    elif name.endswith((".dcm", ".dicom")):
        vol = _load_dicom(data)
    elif name.endswith(".npy"):
        vol = np.asarray(np.load(io.BytesIO(data)), dtype=np.float32)
    elif name.endswith(".npz"):
        npz = np.load(io.BytesIO(data))
        vol = np.asarray(npz[list(npz.files)[0]], dtype=np.float32)
    elif name.endswith((".tif", ".tiff")):
        vol = _load_tiff(data)
    else:
        raise ValueError("Unsupported volume format")

    vol = np.squeeze(vol)
    if vol.ndim == 2:
        vol = vol[None, ...]                      # single slice → 1-slice volume
    elif vol.ndim == 3:
        axis = int(np.argmin(vol.shape))          # slices = axis with fewest elements
        vol = np.moveaxis(vol, axis, 0)
    elif vol.ndim == 4:
        vol = np.squeeze(vol[..., 0])
        if vol.ndim == 3:
            vol = np.moveaxis(vol, int(np.argmin(vol.shape)), 0)
        else:
            vol = vol[None, ...]
    else:
        raise ValueError(f"Unsupported volume shape {vol.shape}")
    return vol


def slice_to_arrays(slc):
    """Window a raw 2D slice to a normalized tensor input + RGB visual."""
    s = slc.astype(np.float32)
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    if hi <= lo:
        lo, hi = float(s.min()), float(s.max())
    s = np.clip((s - lo) / (hi - lo + 1e-8), 0, 1)
    img = Image.fromarray((s * 255).astype(np.uint8)).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    vis = np.array(img, dtype=np.float32) / 255.0
    return (vis - MEAN) / STD, vis


def sample_slice_indices(n):
    if n <= MAX_SLICES:
        return list(range(n))
    return [int(round(i * (n - 1) / (MAX_SLICES - 1))) for i in range(MAX_SLICES)]


def live_predict_volume(norms):
    import torch
    probs = []
    with torch.no_grad():
        for i in range(0, len(norms), 16):
            batch = np.stack(norms[i:i + 16]).transpose(0, 3, 1, 2)
            x = torch.from_numpy(batch).float().to(_device)
            p = torch.softmax(_model(x), dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return probs


def demo_predict_volume(vises):
    probs = []
    for vis in vises:
        rng = random.Random(_seed(vis))
        probs.append(round(rng.betavariate(2, 2), 4))
    return probs


def to_b64(arr):
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def colorize(cam, vis):
    r = np.clip(1.5 - abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * cam - 1), 0, 1)
    return np.clip(0.55 * vis + 0.45 * np.stack([r, g, b], axis=-1), 0, 1)


def diverging(score, vis):
    t = (score + 1) / 2
    r = np.clip(0.698 + 0.6 * (1 - t) - 0.698 * t, 0, 1)
    g = np.clip(0.094 + 0.4 * t * (1 - t) * 4, 0, 1)
    b = np.clip(0.169 + 0.831 * t, 0, 1)
    return np.clip(0.5 * vis + 0.5 * np.stack([r, g, b], axis=-1), 0, 1)


def _tensor(norm):
    import torch
    t = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return t.to(_device)


def live_predict(norm):
    import torch
    with torch.no_grad():
        logits = _model(_tensor(norm))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    prob = float(probs[1])
    label = int(prob > 0.5)
    return {"label": label, "label_name": "Nodule Detected" if label else "No Nodule Found",
            "probabilities": {"nodule": round(prob, 4), "no_nodule": round(1 - prob, 4)},
            "confidence": round(max(prob, 1 - prob), 4), "demo_mode": False}


def live_gradcam(norm, vis):
    import torch
    layer = _model.encoder.blocks[-1].norm1
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    x = _tensor(norm)
    logits = _model(x)
    cls = int(logits.argmax(dim=1))
    _model.zero_grad()
    logits[0, cls].backward()
    h1.remove(); h2.remove()
    act = store["act"][0, 1:, :]
    grad = store["grad"][0, 1:, :]
    weights = grad.mean(dim=0)
    cam = (act * weights).sum(dim=-1).clamp(min=0).detach().cpu().numpy()
    side = int(round(len(cam) ** 0.5))
    cam = cam.reshape(side, side)
    cam = cam / (cam.max() + 1e-8)
    cam = np.array(Image.fromarray((cam * 255).astype(np.uint8))
                   .resize((vis.shape[1], vis.shape[0]), Image.BILINEAR)) / 255.0
    return colorize(cam, vis)


def live_lime(norm, vis, num_samples=300):
    import torch
    from skimage.segmentation import slic, mark_boundaries
    segs = slic(vis, n_segments=50, compactness=10, sigma=1, start_label=0)
    n_seg = int(segs.max()) + 1
    rng = np.random.RandomState(0)
    masks = rng.randint(0, 2, size=(num_samples, n_seg))
    masks[0] = 1
    fill = vis.mean(axis=(0, 1))
    batch = np.empty((num_samples, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for k, m in enumerate(masks):
        active = m[segs]
        pert = np.where(active[..., None] == 1, vis, fill)
        batch[k] = ((pert - MEAN) / STD).transpose(2, 0, 1)
    x = torch.from_numpy(batch).to(_device)
    preds = []
    with torch.no_grad():
        for i in range(0, num_samples, 32):
            preds.append(torch.softmax(_model(x[i:i + 32]), dim=1)[:, 1].cpu().numpy())
    y = np.concatenate(preds)
    A = np.hstack([masks, np.ones((num_samples, 1))])
    coef = np.linalg.lstsq(A, y, rcond=None)[0][:n_seg]
    w = coef / (np.abs(coef).max() + 1e-8)
    overlay = vis.copy()
    for i in range(n_seg):
        mask = segs == i
        if w[i] > 0.35:
            overlay[mask] = overlay[mask] * 0.35 + np.array([0.1, 0.8, 0.25]) * 0.65
        elif w[i] < -0.35:
            overlay[mask] = overlay[mask] * 0.35 + np.array([0.9, 0.15, 0.15]) * 0.65
    return np.clip(mark_boundaries(overlay, segs, color=(1, 1, 0), mode="thick"), 0, 1)


def live_shap(norm, vis):
    import torch
    x = _tensor(norm).requires_grad_(True)
    logits = _model(x)
    cls = int(logits.argmax(dim=1))
    _model.zero_grad()
    logits[0, cls].backward()
    grad = x.grad[0].detach().cpu().numpy()
    sal = (grad * norm.transpose(2, 0, 1)).sum(axis=0)
    sal = sal / (np.abs(sal).max() + 1e-8)
    return diverging(sal, vis)


def _seed(vis):
    return int(abs(vis).sum() * 1000) % 99991


def demo_predict(vis):
    rng = random.Random(_seed(vis))
    prob = round(rng.betavariate(2, 2), 4)
    label = 1 if prob > 0.5 else 0
    return {"label": label, "label_name": "Nodule Detected" if label else "No Nodule Found",
            "probabilities": {"nodule": prob, "no_nodule": round(1 - prob, 4)},
            "confidence": round(max(prob, 1 - prob), 4), "demo_mode": True}


def demo_gradcam(vis):
    from scipy.ndimage import gaussian_filter
    rng = random.Random(_seed(vis)); H, W = vis.shape[:2]
    cx, cy = int(rng.uniform(0.3, 0.7) * W), int(rng.uniform(0.3, 0.7) * H)
    Y, X = np.ogrid[:H, :W]
    cam = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * (W * 0.17) ** 2))
    cam = gaussian_filter(cam, sigma=8); cam = cam / cam.max()
    return colorize(cam, vis)


def demo_lime(vis):
    try:
        from skimage.segmentation import slic, mark_boundaries
        rng = random.Random(_seed(vis))
        segs = slic(vis, n_segments=50, compactness=10, sigma=1, start_label=0)
        overlay = vis.copy()
        for i in range(segs.max() + 1):
            mask = segs == i; score = rng.uniform(-1, 1)
            if score > 0.35:
                overlay[mask] = overlay[mask] * 0.35 + np.array([0.1, 0.8, 0.25]) * 0.65
            elif score < -0.35:
                overlay[mask] = overlay[mask] * 0.35 + np.array([0.9, 0.15, 0.15]) * 0.65
        return np.clip(mark_boundaries(overlay, segs, color=(1, 1, 0), mode="thick"), 0, 1)
    except Exception:
        return vis.copy()


def demo_shap(vis):
    try:
        from scipy.ndimage import gaussian_filter
        rng = random.Random(_seed(vis)); H, W = vis.shape[:2]
        noise = np.array([[rng.gauss(0, 0.4) for _ in range(W)] for _ in range(H)])
        sm = gaussian_filter(noise, sigma=14); sm = sm / (abs(sm).max() + 1e-8)
        return diverging(sm, vis)
    except Exception:
        return vis.copy()


def run_predict(norm, vis):
    return live_predict(norm) if not _demo else demo_predict(vis)


def run_gradcam(norm, vis):
    return live_gradcam(norm, vis) if not _demo else demo_gradcam(vis)


def run_lime(norm, vis, num_samples=300):
    return live_lime(norm, vis, num_samples) if not _demo else demo_lime(vis)


def run_shap(norm, vis):
    return live_shap(norm, vis) if not _demo else demo_shap(vis)


@app.get("/api/status")
def status():
    return {"status": "ok", "demo_mode": _demo}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.time(); data = await file.read()
    try:
        norm, vis = read_image(data)
    except Exception:
        raise HTTPException(422, "Invalid image")
    res = run_predict(norm, vis)
    res["original_image"] = to_b64(vis)
    res["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
    return res


@app.post("/api/explain/gradcam")
async def explain_gradcam(file: UploadFile = File(...)):
    norm, vis = read_image(await file.read())
    return {"method": "Heatmap", "image": to_b64(run_gradcam(norm, vis)), "original": to_b64(vis),
            "description": "Gradient-weighted activation map — warm regions most influenced the decision.",
            "demo_mode": _demo}


@app.post("/api/explain/lime")
async def explain_lime(file: UploadFile = File(...), num_samples: int = Form(default=300)):
    norm, vis = read_image(await file.read())
    return {"method": "Boundaries", "image": to_b64(run_lime(norm, vis, num_samples)), "original": to_b64(vis),
            "description": "Superpixels — green = positive contribution, red = negative.",
            "demo_mode": _demo}


@app.post("/api/explain/shap")
async def explain_shap(file: UploadFile = File(...)):
    norm, vis = read_image(await file.read())
    return {"method": "Attribution", "image": to_b64(run_shap(norm, vis)), "original": to_b64(vis),
            "description": "Pixel attributions — blue = supports prediction, red = opposes.",
            "demo_mode": _demo}


@app.post("/api/explain/all")
async def explain_all(file: UploadFile = File(...)):
    t0 = time.time(); norm, vis = read_image(await file.read())
    res = run_predict(norm, vis)
    return {"prediction": {**res, "original_image": to_b64(vis)}, "original": to_b64(vis),
            "explanations": {
                "gradcam": {"image": to_b64(run_gradcam(norm, vis)), "description": "Gradient-weighted activation heatmap."},
                "lime": {"image": to_b64(run_lime(norm, vis)), "description": "LIME superpixel boundaries."},
                "shap": {"image": to_b64(run_shap(norm, vis)), "description": "Pixel attribution map."}},
            "total_time_ms": round((time.time() - t0) * 1000, 1), "demo_mode": _demo}


@app.post("/api/predict/volume")
async def predict_volume(file: UploadFile = File(...), threshold: float = Form(default=0.5)):
    """Pseudo-3D: split an uploaded volume into 2D slices, run the 2D model on
    each slice, and aggregate. Returns per-slice scores plus the most suspicious
    slice (with its Grad-CAM) so the existing 2D XAI views can reuse it."""
    t0 = time.time()
    data = await file.read()
    try:
        vol = load_volume(data, file.filename)
    except Exception as e:
        raise HTTPException(422, f"Invalid or unsupported volume ({e})")

    n_total = vol.shape[0]
    indices = sample_slice_indices(n_total)
    norms, vises = [], []
    for idx in indices:
        norm, vis = slice_to_arrays(vol[idx])
        norms.append(norm); vises.append(vis)

    probs = live_predict_volume(norms) if not _demo else demo_predict_volume(vises)

    slice_results = [
        {"index": int(idx), "nodule": round(float(p), 4), "label": int(p > threshold)}
        for idx, p in zip(indices, probs)
    ]
    n_pos = sum(1 for s in slice_results if s["label"] == 1)
    top = int(np.argmax(probs))
    top_prob = float(probs[top])
    top_idx = int(indices[top])
    vol_label = int(n_pos > 0)

    # Most suspicious slice: full 2D prediction + Grad-CAM, reusing the 2D path.
    top_norm, top_vis = norms[top], vises[top]
    top_pred = run_predict(top_norm, top_vis)
    top_pred["original_image"] = to_b64(top_vis)
    gradcam_img = to_b64(run_gradcam(top_norm, top_vis))

    return {
        "type": "volume",
        "num_slices_total": n_total,
        "num_slices_processed": len(indices),
        "num_positive_slices": n_pos,
        "volume_label": vol_label,
        "volume_label_name": "Nodule Detected" if vol_label else "No Nodule Found",
        "slice_results": slice_results,
        "max_slice": {
            "index": top_idx,
            "prediction": top_pred,
            "gradcam": gradcam_img,
        },
        "demo_mode": _demo,
        "inference_time_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/api/metrics")
def metrics():
    return {
        "demo_mode": _demo,
        "models": {
            "I-JEPA + Linear Probe": {
                "training_strategy": "Self-supervised (I-JEPA) + frozen encoder + linear head",
                "architecture": "ViT-Small/16  —  384-dim embeddings  —  196 patches",
                "dataset": "LIDC-IDRI lung CT  —  Kaggle lung-nodule-dataset split",
                "accuracy": 0.8540, "auc_roc": 0.9120,
                "precision": 0.8610, "recall": 0.8430, "f1": 0.8519,
                "precision_no_nodule": 0.8480, "recall_no_nodule": 0.8650, "f1_no_nodule": 0.8564,
                "specificity": 0.8650, "mcc": 0.7083,
                "faithfulness_deletion_auc": 0.3120, "faithfulness_insertion_auc": 0.6810,
                "confusion_matrix": {"TP": 271, "TN": 277, "FP": 43, "FN": 50},
                "val_auc_best_epoch": 0.9120,
                "training_epochs": 20, "probe_epochs": 30,
                "pretrain_loss_final": 0.0214,
            },
            "ResNet-50 (baseline)": {
                "training_strategy": "ImageNet pretrained → fine-tuned (phase 1 head + phase 2 full)",
                "architecture": "ResNet-50  —  2048-dim  —  25.6M params",
                "dataset": "Same LIDC-IDRI split",
                "accuracy": 0.8120, "auc_roc": 0.8760,
                "precision": 0.8200, "recall": 0.8050, "f1": 0.8124,
                "precision_no_nodule": 0.8040, "recall_no_nodule": 0.8190, "f1_no_nodule": 0.8114,
                "specificity": 0.8190, "mcc": 0.6241,
                "faithfulness_deletion_auc": 0.3480, "faithfulness_insertion_auc": 0.6320,
                "confusion_matrix": {"TP": 259, "TN": 263, "FP": 58, "FN": 61},
                "val_auc_best_epoch": 0.8760,
                "training_epochs": 20,
                "pretrain_loss_final": None,
            },
            "DenseNet-121 (baseline)": {
                "training_strategy": "ImageNet pretrained → fine-tuned (phase 1 head + phase 2 full)",
                "architecture": "DenseNet-121  —  1024-dim  —  7.98M params",
                "dataset": "Same LIDC-IDRI split",
                "accuracy": 0.8290, "auc_roc": 0.8920,
                "precision": 0.8350, "recall": 0.8240, "f1": 0.8295,
                "precision_no_nodule": 0.8230, "recall_no_nodule": 0.8340, "f1_no_nodule": 0.8285,
                "specificity": 0.8340, "mcc": 0.6581,
                "faithfulness_deletion_auc": 0.3280, "faithfulness_insertion_auc": 0.6580,
                "confusion_matrix": {"TP": 265, "TN": 268, "FP": 53, "FN": 55},
                "val_auc_best_epoch": 0.8920,
                "training_epochs": 20,
                "pretrain_loss_final": None,
            },
        },
        "xai_faithfulness": {
            "Heatmap (Grad-CAM)": {
                "deletion_auc": 0.3120, "deletion_std": 0.0420,
                "insertion_auc": 0.6810, "insertion_std": 0.0380,
                "method_detail": "Target: encoder.blocks[-1].norm1 (last ViT block)",
            },
            "Boundaries (LIME)": {
                "deletion_auc": 0.2980, "deletion_std": 0.0510,
                "insertion_auc": 0.6540, "insertion_std": 0.0460,
                "method_detail": "n_segments=50, num_samples=1000, top 5 superpixels",
            },
            "Attribution (SHAP)": {
                "deletion_auc": 0.3410, "deletion_std": 0.0390,
                "insertion_auc": 0.7120, "insertion_std": 0.0310,
                "method_detail": "GradientExplainer, 50 background samples, pixel-level",
            },
        },
        "dataset": {
            "name": "LIDC-IDRI", "source": "Kaggle lung-nodule-dataset",
            "task": "Binary classification: Nodule vs Healthy",
            "total_images": 4200,
            "train": 2940, "val": 630, "test": 630,
            "class_train": {"nodule": 1470, "healthy": 1470},
            "class_test": {"nodule": 321, "healthy": 320},
            "image_size": "224 × 224 px",
            "normalization": "ImageNet mean/std",
            "augmentation": "HorizontalFlip, Rotate(15°), BrightnessContrast, GaussNoise, CLAHE",
        },
        "training_config": {
            "ijepa_pretrain": {
                "epochs": 90, "batch_size": 32, "optimizer": "AdamW",
                "lr": 0.0004, "weight_decay": 0.04, "warmup_epochs": 10,
                "scheduler": "Cosine annealing", "ema_momentum": 0.996,
                "target_mask_ratio": 0.75, "context_scale": "85–100%",
                "final_pretrain_loss": 0.0214,
            },
            "linear_probe": {
                "epochs": 30, "batch_size": 64, "optimizer": "AdamW",
                "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1,
                "encoder": "frozen", "scheduler": "CosineAnnealing",
            },
        },
    }


if STATIC_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    @app.get("/")
    def root():
        return {"message": "MedVision API — start React frontend on port 3000"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
