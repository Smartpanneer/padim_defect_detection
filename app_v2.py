#!/usr/bin/env python3
"""
Royal Enfield - Fuel Tank Defect Detection Server
app_ign.py logic runs INSIDE this server as run_inference().
raspi_vision_controller.py socket protocol is integrated directly.
Run: python server.py   →   http://localhost:8080
"""
from __future__ import annotations

import os, sys, time, json, shutil, struct, threading, traceback
import socket as _socket
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.covariance import EmpiricalCovariance
from ultralytics import YOLO
from PIL import Image as PILImage

from flask import Flask, jsonify, request, send_file, send_from_directory, Response
from flask_socketio import SocketIO, emit

# ══ Fixed seeds — MUST come before any torch/numpy random operation ═══
# Without this, torch.randperm gives a different channel order every run.
# Training channels ≠ inference channels → hundreds of false positives.
torch.manual_seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
WATCH_FOLDER     = BASE_DIR / "watch_folder"
PROCESSED_DEFECT = BASE_DIR / "processed_defect"
PROCESSED_CLEAN  = BASE_DIR / "processed_clean"
SETTINGS_FILE    = BASE_DIR / "settings.json"
TRAIN_FOLDER     = BASE_DIR / "train_normal"
YOLO_MODEL_PATH  = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
STATIC_DIR       = BASE_DIR / "static"
PADIM_MODEL_PATH = BASE_DIR / "padim_model.npz"   # saved after first training run

for _d in [WATCH_FOLDER, PROCESSED_DEFECT, PROCESSED_CLEAN, STATIC_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Default settings ──────────────────────────────────────────
DEFAULT_SETTINGS = {
    "camera_ip": "192.168.50.3", "camera_port": 5000,
    "camera_url": "http://192.168.50.3/image.jpg",
    # threshold_mode:
    #   "adaptive" (recommended) — threshold = mean + K*std of inside-tank Mahalanobis.
    #               "threshold" is K (sigma multiplier).  Start at 2.5.
    #               Lower K → more sensitive (catches faint defects, more FP risk).
    #               Higher K → more strict (fewer FP, may miss subtle defects).
    #   "ratio"    (legacy)     — normalise by p98, compare ratio to TH.
    #               "threshold" is TH.  Range was 0.75–1.15.
    "threshold_mode": "adaptive",
    # threshold = K in (mean + K*std).
    # 2.0 = very sensitive (catches faint defects, higher FP risk on curved surfaces)
    # 2.5 = balanced  ← recommended starting point for missing defects
    # 3.0 = strict    (fewer FP, may miss subtle marks)
    "threshold": 2.5,
    "yolo_conf": 0.8,

    # ── PaDiM sensitivity controls ────────────────────────────────────
    # padim_enhance: apply CLAHE to each tile BEFORE ResNet feature extraction.
    #   WARNING: if you enable this, you MUST delete padim_model.npz and retrain.
    #   Training and inference MUST use the same preprocessing.
    #   Set True ONLY after retraining with enhanced images.
    "padim_enhance": False,
    "padim_enhance_clip": 3.0,   # CLAHE clip limit for PaDiM tiles (2–4)
    "padim_enhance_tile": 8,     # CLAHE tile grid size for PaDiM tiles

    # padim_feature_res: internal feature map resolution (default 64).
    # Increasing to 96 or 128 gives finer spatial localisation of small defects.
    # WARNING: changing this requires DELETE + RETRAIN of padim_model.npz.
    "padim_feature_res": 96,

    # ── FIX 1: Layer selection for feature extraction ─────────────────
    # "layers": which ResNet50 layers to fuse for PaDiM features.
    # "1+2"   → layer1 (256ch) + layer2 (512ch) = 768ch total → 448 sampled
    #           BEST for thin scratches: earlier layers keep spatial detail
    #           (stride-4 and stride-8 receptive fields vs stride-16 for layer3)
    # "1+2+3" → all three layers (default, legacy behaviour, 1792ch total)
    # WARNING: changing this requires DELETE + RETRAIN of padim_model.npz.
    "padim_layers": "1+2",

    # ── FIX 2: Heatmap thresholding pipeline ─────────────────────────
    # refine_blur_k: Gaussian blur kernel before binary threshold (must be odd)
    "refine_blur_k": 5,
    # refine_morph_open: morphology OPEN kernel after threshold (removes specks)
    "refine_morph_open": 3,
    # refine_morph_close: morphology CLOSE kernel to join scratch fragments
    "refine_morph_close": 9,

    # ── FIX 3: Scratch-aware connected component filtering ────────────
    # scratch_min_area: smaller than this = noise dot (px)
    "scratch_min_area": 80,
    # scratch_max_area: larger than this = lighting gradient blob (px)
    "scratch_max_area": 15000,
    # scratch_min_asp: scratch must be at least this elongated (length/width)
    "scratch_min_asp": 1.2,
    # dot_max_area: round dots (dents/pits) can be small and round — separate gate
    "dot_max_area": 3000,

    # ── FIX 4: 2-stage bbox refinement ───────────────────────────────
    # refine_bbox_enabled: run tight Sobel+Canny re-fit on each rough bbox
    "refine_bbox_enabled": True,
    # refine_pad: pixels of padding around rough bbox before re-analysis
    "refine_pad": 12,
    # refine_canny_low / high: Canny thresholds for edge detection
    "refine_canny_low": 20,
    "refine_canny_high": 60,

    # gate_peak_margin: Gate 7 accepts blobs whose peak >= effective_th * margin.
    # Lowering from 0.80 to 0.70 catches weaker anomaly peaks.
    "gate_peak_margin": 0.70,

    # min_defect_area: minimum connected-component pixel area.
    "min_defect_area": 200,
    # max_defect_area: maximum area — gradient blobs are larger than real defects.
    "max_defect_area": 12000,
    # min_roi_std: Sobel gradient std inside bbox. Faint marks score 4–8.
    # Set to 3 to ensure real defects are never rejected by this gate.
    "min_roi_std": 3,
    # max_aspect_ratio: long thin gradient bands score high asp.
    "max_aspect_ratio": 6.0,
    # use_gradient_contrast: Gate 9 uses Sobel gradient std (True) or pixel std (False)
    "use_gradient_contrast": True,
    "s_max": 55, "v_min": 90, "v_max": 255, "tank_min_area_frac": 0.25,
    "opener_pad_left": 120, "opener_pad_right": 180,
    "opener_pad_top": 100,  "opener_pad_bottom": 150,
    "hough_pad_left": 220,  "hough_pad_right": 220,
    "hough_pad_top":  220,  "hough_pad_bottom": 220,
    "hough_min_radius": 80, "hough_max_radius": 350,
    "hough_max_centre_frac": 0.45,
    # pipe_excl_frac: left-edge pipe column exclusion (12 % of width)
    "pipe_excl_frac": 0.12,
    # right_excl_frac: metallic rim exclusion
    "right_excl_frac": 0.13,
    # top_excl_frac: the physical tank seam/rim along the top edge ALWAYS scores
    # hot for both PaDiM and white-mark detector. 0.15 = exclude top 15% of image.
    # If you still see rim false positives, raise to 0.18 or 0.20.
    "top_excl_frac": 0.15,
    # bottom_excl_frac: lower curved boundary exclusion
    "bottom_excl_frac": 0.08,
    # erode_tank_mask_px: additional erosion of tank mask before analysis (pixels).
    # Increasing this pushes ALL exclusion zones inward from the green boundary.
    # Default 15px. Raise to 25-35 if rim/edge FPs persist after top_excl_frac.
    "erode_tank_mask_px": 20,
    "opener_class_id": 0,
    "scratch_min_length": 80,
    "scratch_aspect_ratio": 3.0,
    "scratch_peak_scale": 0.75,
    # ── White / light surface mark detector (yellow boxes) ────────────
    # Uses CLAHE + dual-background subtraction to detect faint thin marks.
    "white_mark_enabled":  True,
    "wm_delta":            25,    # CLAHE-enhanced deviation threshold (25 = good default)
    "wm_clahe_clip":       6.0,   # CLAHE clip limit (higher = more aggressive enhancement)
    "wm_clahe_tile":       16,    # CLAHE tile grid size in pixels
    "wm_blur_radius":      81,    # isotropic background blur radius (must be odd)
    "wm_horz_blur":        201,   # horizontal-only blur to kill banding (must be odd)
    "wm_min_area":         150,   # min blob area in pixels
    "wm_max_area":         40000, # max blob area
    "wm_min_asp":          1.0,   # min aspect ratio
    "wm_max_asp":          8.0,   # max aspect ratio

    # ── Industrial Uniformity Fixes ───────────────────────────────────
    # FIX 1 — Min-max normalise heatmap before threshold.
    # Prevents one bright region from dominating the dynamic range,
    # ensuring faint defects on the opposite side still get fair comparison.
    "heatmap_normalize": True,

    # FIX 2 — Percentile-based threshold (replaces fixed K*std).
    # threshold_percentile: top X% of inside-tank scores are flagged.
    # e.g. 97 = top 3% are anomalous. Range 95–99 (lower = more sensitive).
    # Only used when threshold_mode = "percentile".
    "threshold_mode": "percentile",
    "threshold_percentile": 97.5,

    # FIX 3 — 4-zone per-surface percentile detection.
    # Splits safe tank area into Top/Bottom/Left/Right zones and applies
    # an independent percentile threshold per zone. This guarantees every
    # zone gets an equal chance to contribute defects — one bright zone
    # cannot suppress detection in another zone.
    "zone_detect_enabled": True,
    "zone_percentile":     97.5,   # percentile per zone (95–99)
    "zone_min_area":       80,     # minimum blob area inside a zone (px)

    # FIX 4 — No top-N cap on blobs.
    # Previously blobs could be capped by max_defect_area acting as a
    # de-facto top-N filter. Set max_defect_area very high to disable.
    # (Already tunable; explicit flag added for clarity.)
    "zone_max_boxes":      999,    # effectively unlimited per zone

    # FIX 5 — Area-only connected component filter.
    # Keep all blobs above minimum area, regardless of score ranking.
    # Score is only used at Gate 7; area is the primary survival criterion.
    "zone_area_only_filter": True,
}

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

# ── Flask / SocketIO ──────────────────────────────────────────
app      = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["SECRET_KEY"] = "re-secret-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global state ──────────────────────────────────────────────
settings          = load_settings()
camera_socket     = None
camera_lock       = threading.Lock()
camera_connected  = False
padim_trained     = False
yolo_loaded       = False
processed_images  = []

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
backbone      = None
_layer_buf    = []
padim_mean    = None
padim_inv_cov = None
_transform    = None
yolo_model    = None

# ── Live frame buffer (populated by live stream thread) ───────
_live_frame_lock  = threading.Lock()
_live_frame_bytes = None   # latest raw JPEG bytes from camera
_live_frame_ts    = 0.0    # time.time() of last frame

# ── PaDiM feature extraction ──────────────────────────────────
def _feat_hook(module, inp, out):
    _layer_buf.append(out)

# Fixed channel indices — set once in init_padim with seed=42 and reused
# for every training AND inference call.  Changing this means retraining.
_padim_ch_idx = None

# Layer mode — set from settings["padim_layers"] in init_padim.
# "1+2" = layer1+layer2 only (finer spatial resolution, best for thin scratches)
# "1+2+3" = all three layers (legacy default, wider receptive field)
_padim_layer_mode = "1+2"

# CLAHE enhancer for PaDiM tiles — created once, reused every call
# Only active when settings["padim_enhance"] = True AND model was retrained with enhancement
_padim_clahe = None

def _extract_features(image_path: str, img_bgr=None, feat_res: int = 64,
                       enhance: bool = False, clahe_clip: float = 3.0, clahe_tile: int = 8):
    """ResNet50 feature extraction for PaDiM.

    IMPORTANT — enhancement consistency rule:
      If enhance=True, training AND inference MUST both use enhancement.
      Mismatch = systematic false positives everywhere.
      Default: enhance=False (safe, matches existing padim_model.npz).

    Args:
        image_path : path to image (used only if img_bgr is None)
        img_bgr    : pre-loaded BGR image (tile crops are passed here)
        feat_res   : spatial resolution of output feature map (default 64).
                     Higher = finer localisation of small defects.
                     Changing this requires full retrain.
        enhance    : apply CLAHE before feature extraction
        clahe_clip : CLAHE clip limit (2–4 recommended)
        clahe_tile : CLAHE tile grid size
    """
    global _layer_buf, _padim_clahe
    _layer_buf = []
    img = img_bgr if img_bgr is not None else cv2.imread(image_path)

    # Optional CLAHE enhancement — makes faint surface marks visible to ResNet
    # Only use after retraining with the same enhancement applied
    if enhance:
        if _padim_clahe is None or getattr(_padim_clahe, '_clip', None) != clahe_clip:
            _padim_clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                            tileGridSize=(clahe_tile, clahe_tile))
            _padim_clahe._clip = clahe_clip
        # Apply CLAHE per channel in LAB space (only L channel)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = _padim_clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PILImage.fromarray(img)
    t   = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _ = backbone(t)

    # ── FIX 1: Layer selection — use layer1+layer2 for finer spatial resolution ──
    # layer1 (stride-4) and layer2 (stride-8) preserve more spatial detail than
    # layer3 (stride-16), making thin scratches localisable at pixel level.
    # _padim_layer_mode is set in init_padim() from settings["padim_layers"].
    # MUST match what was used during training — changing requires full retrain.
    _layers_to_use = _padim_layer_mode   # global set by init_padim
    if _layers_to_use == "1+2":
        f1 = F.interpolate(_layer_buf[0], size=(feat_res, feat_res), mode="bilinear", align_corners=False)
        f2 = F.interpolate(_layer_buf[1], size=(feat_res, feat_res), mode="bilinear", align_corners=False)
        feat = torch.cat([f1, f2], dim=1)            # (1, 768, feat_res, feat_res)
    else:
        # Default 1+2+3 — all three layers (legacy)
        f1 = F.interpolate(_layer_buf[0], size=(feat_res, feat_res), mode="bilinear", align_corners=False)
        f2 = F.interpolate(_layer_buf[1], size=(feat_res, feat_res), mode="bilinear", align_corners=False)
        f3 = F.interpolate(_layer_buf[2], size=(feat_res, feat_res), mode="bilinear", align_corners=False)
        feat = torch.cat([f1, f2, f3], dim=1)        # (1, 1792, feat_res, feat_res)
    if _padim_ch_idx is not None:
        feat = feat[:, _padim_ch_idx, :, :]
    return feat.squeeze().permute(1,2,0).cpu().numpy()


def init_padim():
    """First run: train on train_normal/ and save padim_model.npz  (~2 min).
    Later runs: load padim_model.npz from disk                     (~1 sec).
    Force retrain: DELETE padim_model.npz or POST /api/padim/retrain.
    """
    global backbone, _transform, padim_mean, padim_inv_cov, padim_trained, _padim_ch_idx, _padim_layer_mode
    try:
        print("[PaDiM] Loading ResNet50 backbone...")
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.eval().to(DEVICE)
        backbone.layer1.register_forward_hook(_feat_hook)
        backbone.layer2.register_forward_hook(_feat_hook)
        backbone.layer3.register_forward_hook(_feat_hook)
        _transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # ── FIX 1: Layer mode from settings ────────────────────────────
        _padim_layer_mode = str(settings.get("padim_layers", "1+2"))
        print(f"[PaDiM] Layer mode: {_padim_layer_mode}")

        # ── Fixed channel selection ────────────────────────────────────
        # Re-seed immediately before randperm so the same 448 channels
        # are always selected regardless of what ran before.
        torch.manual_seed(42)
        if _padim_layer_mode == "1+2":
            total_ch = 256 + 512               # layer1 + layer2 only = 768
        else:
            total_ch = 256 + 512 + 1024        # layer1 + layer2 + layer3 = 1792
        _padim_ch_idx = torch.randperm(total_ch)[:448]
        print(f"[PaDiM] total_ch={total_ch}  Channels first5={_padim_ch_idx[:5].tolist()} (must be same every run)")

        # ── Load saved model ───────────────────────────────────────────
        if PADIM_MODEL_PATH.exists():
            print(f"[PaDiM] Loading saved model from {PADIM_MODEL_PATH} ...")
            d = np.load(str(PADIM_MODEL_PATH))
            padim_mean    = d["mean"]
            padim_inv_cov = d["inv_cov"]
            padim_trained = True
            print("[PaDiM] Model loaded — no training needed.  "
                  "(Delete padim_model.npz to force retrain)")
            return

        # ── Train from scratch ─────────────────────────────────────────
        if not TRAIN_FOLDER.exists():
            print(f"[PaDiM] WARNING: {TRAIN_FOLDER} not found."); return
        files = sorted([p for p in TRAIN_FOLDER.iterdir()
                        if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")])
        if not files:
            print("[PaDiM] WARNING: No training images found."); return

        # Check if CLAHE enhancement is enabled for training
        train_enhance    = bool(settings.get("padim_enhance",       False))
        train_enh_clip   = float(settings.get("padim_enhance_clip", 3.0))
        train_enh_tile   = int(settings.get("padim_enhance_tile",   8))
        train_feat_res   = int(settings.get("padim_feature_res",    64))
        print(f"[PaDiM] Training on {len(files)} images (runs once, then saved)...")
        print(f"[PaDiM] enhance={train_enhance}  feat_res={train_feat_res}  "
              f"clahe_clip={train_enh_clip if train_enhance else 'N/A'}")
        all_f = []
        for p in files:
            ft = _extract_features(str(p),
                                   feat_res=train_feat_res,
                                   enhance=train_enhance,
                                   clahe_clip=train_enh_clip,
                                   clahe_tile=train_enh_tile)
            all_f.append(ft.reshape(-1, ft.shape[2]))
        all_f = np.vstack(all_f)
        print(f"[PaDiM] Feature matrix shape={all_f.shape}  "
              f"mean={all_f.mean():.4f}  std={all_f.std():.4f}")

        cov        = EmpiricalCovariance(assume_centered=False).fit(all_f)
        padim_mean = cov.location_
        raw_cov    = cov.covariance_
        shrunk_cov = raw_cov + (np.trace(raw_cov) * 1e-4) * np.eye(raw_cov.shape[0])
        padim_inv_cov = np.linalg.pinv(shrunk_cov)

        np.savez_compressed(str(PADIM_MODEL_PATH),
                            mean=padim_mean, inv_cov=padim_inv_cov)
        print(f"[PaDiM] Model saved → {PADIM_MODEL_PATH}")
        padim_trained = True
        print("[PaDiM] Training complete.")
    except Exception:
        print("[PaDiM] FAILED:"); traceback.print_exc()

def init_yolo():
    global yolo_model, yolo_loaded
    try:
        if not YOLO_MODEL_PATH.exists():
            print(f"[YOLO] WARNING: model not found at {YOLO_MODEL_PATH}"); return
        yolo_model  = YOLO(str(YOLO_MODEL_PATH))
        yolo_loaded = True
        print("[YOLO] Loaded.")
    except Exception:
        print("[YOLO] FAILED:"); traceback.print_exc()

# ── Inference (app_ign.py algorithm) ─────────────────────────
def run_inference(image_path):
    if not padim_trained:
        raise RuntimeError("PaDiM not trained. Ensure train_normal/ has images.")

    cfg = settings
    TH   = float(cfg["threshold"]);       YCONF = float(cfg["yolo_conf"])
    MDA  = int(cfg["min_defect_area"]);   OCI   = int(cfg["opener_class_id"])
    MXA  = int(cfg.get("max_defect_area",  12000))   # Gate 8: max area (gradient filter)
    MSTD = float(cfg.get("min_roi_std",    4.0))     # Gate 9: min local texture std
    MASP = float(cfg.get("max_aspect_ratio", 5.0))   # Gate 10: max aspect ratio
    USE_GRAD = bool(cfg.get("use_gradient_contrast", True))  # Gate 9 mode
    SMAX = int(cfg["s_max"]);             VMIN  = int(cfg["v_min"])
    VMAX = int(cfg["v_max"]);             TMIN  = float(cfg["tank_min_area_frac"])
    OPL  = int(cfg["opener_pad_left"]);   OPR   = int(cfg["opener_pad_right"])
    OPT  = int(cfg["opener_pad_top"]);    OPB   = int(cfg["opener_pad_bottom"])
    HPL  = int(cfg["hough_pad_left"]);    HPR   = int(cfg["hough_pad_right"])
    HPT  = int(cfg["hough_pad_top"]);     HPB   = int(cfg["hough_pad_bottom"])
    HMN  = int(cfg["hough_min_radius"]);  HMX   = int(cfg["hough_max_radius"])
    HCF  = float(cfg["hough_max_centre_frac"])
    PXF  = float(cfg["pipe_excl_frac"])
    # FIX: read right-edge exclusion fraction (defaults to 0.05 if not in settings)
    RXF  = float(cfg.get("right_excl_frac",   0.05))
    # Top-edge seam exclusion — the physical rim/seam along the top ALWAYS scores hot.
    # Raised default to 0.15 (15% of image height) to cover the rim reliably.
    TXF  = float(cfg.get("top_excl_frac",     0.15))
    # Bottom-edge exclusion (curved boundary gives false positives)
    BXF  = float(cfg.get("bottom_excl_frac",  0.08))
    # Additional tank mask erosion — pushes analysis zone inward from green boundary
    ERODE_PX = int(cfg.get("erode_tank_mask_px", 20))

    orig = cv2.imread(image_path)
    if orig is None:
        raise ValueError(f"Cannot read: {image_path}")
    out  = orig.copy()
    h, w = orig.shape[:2]
    diag = np.sqrt(h**2 + w**2)

    # ══════════════════════════════════════════════════════════════════
    # PaDiM TILED DEEP-ZOOM ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    # Strategy:
    #   1. Build tank mask FIRST (before PaDiM) so tiles that fall
    #      entirely outside the tank are skipped entirely.
    #   2. Divide the image into a grid of 6 tiles (2 rows × 3 cols)
    #      with 20% overlap. Each tile is upscaled to 512×512 so the
    #      model sees surface detail at "microscope zoom".
    #   3. Also run one global full-image pass to catch large defects.
    #   4. Fuse tile maps back via pixel-wise maximum.
    #   5. Normalise using 98th-percentile of INSIDE-TANK scores only,
    #      so edge/boundary scores never set the reference ceiling.
    # ══════════════════════════════════════════════════════════════════

    def _mahal(ft_arr):
        """Mahalanobis distance map from a (H,W,C) feature array."""
        HH, WW, CC = ft_arr.shape
        flat = ft_arr.reshape(-1, CC)
        diff = flat - padim_mean
        return np.sqrt(np.sum((diff @ padim_inv_cov) * diff, axis=1)).reshape(HH, WW)

    # ── Build a quick coarse tank mask for tile skipping ──────────────
    # (Full precise mask is computed in STEP 1 below; this rough version
    #  just prevents running PaDiM on pure background tiles.)
    hsv_q  = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    cmq    = ((hsv_q[:,:,1] <= SMAX) &
              (hsv_q[:,:,2] >= VMIN) &
              (hsv_q[:,:,2] <= VMAX)).astype(np.uint8)
    cmq    = cv2.morphologyEx(cmq, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(55,55)))

    # ── PaDiM sensitivity settings — must be defined BEFORE tiling ───────
    PADIM_ENHANCE    = bool(cfg.get("padim_enhance",       False))
    PADIM_ENH_CLIP   = float(cfg.get("padim_enhance_clip", 3.0))
    PADIM_ENH_TILE   = int(cfg.get("padim_enhance_tile",   8))
    FEAT_RES         = int(cfg.get("padim_feature_res",    64))
    GATE_PEAK_MARGIN = float(cfg.get("gate_peak_margin",   0.70))

    if PADIM_ENHANCE:
        print(f"[PaDiM] ⚠ CLAHE enhancement ACTIVE (clip={PADIM_ENH_CLIP} tile={PADIM_ENH_TILE}). "
              f"Ensure model was retrained with enhancement!")

    def _ef(img_bgr=None):
        """Shorthand: call _extract_features with current run settings."""
        return _extract_features(image_path, img_bgr=img_bgr,
                                  feat_res=FEAT_RES,
                                  enhance=PADIM_ENHANCE,
                                  clahe_clip=PADIM_ENH_CLIP,
                                  clahe_tile=PADIM_ENH_TILE)

    def _run_tiled_pass(rows, cols, ovl, upscale_size, label):
        """Generic tiled PaDiM pass. Returns fused anomaly map (h,w)."""
        th_ = int(h / (rows * (1 - ovl) + ovl))
        tw_ = int(w / (cols * (1 - ovl) + ovl))
        sh_ = max(1, int(th_ * (1 - ovl)))
        sw_ = max(1, int(tw_ * (1 - ovl)))
        acc_ = np.zeros((h, w), np.float32)
        wgt_ = np.zeros((h, w), np.float32)
        n = 0
        for tr in range(rows):
            for tc in range(cols):
                ty1 = tr * sh_;  tx1 = tc * sw_
                ty2 = min(ty1 + th_, h);  tx2 = min(tx1 + tw_, w)
                if ty2 <= ty1 or tx2 <= tx1: continue
                cov = cmq[ty1:ty2, tx1:tx2].mean()
                if cov < 0.08:
                    continue
                tile_img = orig[ty1:ty2, tx1:tx2]
                tile_up  = cv2.resize(tile_img, (upscale_size, upscale_size),
                                      interpolation=cv2.INTER_CUBIC)
                araw_t   = _mahal(_ef(img_bgr=tile_up))
                araw_t   = cv2.resize(araw_t.astype(np.float32),
                                      (tx2-tx1, ty2-ty1), interpolation=cv2.INTER_LINEAR)
                acc_[ty1:ty2, tx1:tx2] += araw_t
                wgt_[ty1:ty2, tx1:tx2] += 1.0
                n += 1
                print(f"  [{label} {tr},{tc}] y:{ty1}-{ty2} x:{tx1}-{tx2} "
                      f"peak={araw_t.max():.2f} cov={cov:.0%}")
        result = np.divide(acc_, wgt_, out=np.zeros_like(acc_), where=wgt_ > 0)
        print(f"[{label}] {n} tiles done. map peak={result.max():.3f}")
        return result

    # ── Level 0: Global pass — full image at 256×256 ─────────────────
    print("[DeepZoom L0] Global full-image pass...")
    ft_g = _ef()
    araw = cv2.resize(_mahal(ft_g).astype(np.float32), (w, h),
                      interpolation=cv2.INTER_LINEAR)

    # ── Level 1: Coarse grid — 3×4 tiles, 512px upscale ─────────────
    print("[DeepZoom L1] 3x4 coarse grid at 512px...")
    araw_l1 = _run_tiled_pass(3, 4, 0.20, 512, "L1")
    araw = np.maximum(araw, araw_l1)

    # ── Level 2: Fine grid — 5×7 micro-tiles, 768px upscale ─────────
    # Each tile covers ~1/5 height × ~1/7 width, upscaled to 768px.
    # Gives ~6-8x magnification — enough to see faint paint marks.
    print("[DeepZoom L2] 5x7 fine grid at 768px (deep zoom)...")
    araw_l2 = _run_tiled_pass(5, 7, 0.25, 768, "L2")
    araw = np.maximum(araw, araw_l2)

    # ── Level 3: Hotspot micro-zoom — 1024px on top anomaly regions ──
    # Find top-N highest-scoring regions from L2, crop tightly at 200px,
    # re-run at 1024px. This is the microscope pass for faintest marks.
    print("[DeepZoom L3] Hotspot micro-zoom at 1024px...")
    HOTSPOT_N  = 8
    HOTSPOT_SZ = 200
    l2_smooth  = cv2.GaussianBlur(araw_l2, (31, 31), 0)
    l2_tank    = l2_smooth * cmq.astype(np.float32)
    hs_acc = np.zeros((h, w), np.float32)
    hs_wgt = np.zeros((h, w), np.float32)
    for _ in range(HOTSPOT_N):
        hy, hx = np.unravel_index(np.argmax(l2_tank), l2_tank.shape)
        hy1 = max(0, hy - HOTSPOT_SZ // 2)
        hx1 = max(0, hx - HOTSPOT_SZ // 2)
        hy2 = min(h, hy1 + HOTSPOT_SZ)
        hx2 = min(w, hx1 + HOTSPOT_SZ)
        if hy2 <= hy1 or hx2 <= hx1:
            break
        cov = cmq[hy1:hy2, hx1:hx2].mean()
        if cov >= 0.15:
            hs_img = orig[hy1:hy2, hx1:hx2]
            hs_up  = cv2.resize(hs_img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            hs_map = _mahal(_ef(img_bgr=hs_up))
            hs_map = cv2.resize(hs_map.astype(np.float32),
                                (hx2-hx1, hy2-hy1), interpolation=cv2.INTER_LINEAR)
            hs_acc[hy1:hy2, hx1:hx2] += hs_map
            hs_wgt[hy1:hy2, hx1:hx2] += 1.0
            print(f"  [L3 hotspot] centre=({hx},{hy}) peak={hs_map.max():.2f} cov={cov:.0%}")
        # Suppress this region so next iteration finds a different hotspot
        l2_tank[max(0,hy-HOTSPOT_SZ):min(h,hy+HOTSPOT_SZ),
                max(0,hx-HOTSPOT_SZ):min(w,hx+HOTSPOT_SZ)] = 0

    araw_l3 = np.divide(hs_acc, hs_wgt, out=np.zeros_like(hs_acc), where=hs_wgt > 0)
    araw = np.maximum(araw, araw_l3)

    print(f"[PaDiM] Deep-zoom complete (L0+L1+L2+L3).  "
          f"Fused map: min={araw.min():.3f} max={araw.max():.3f} "
          f"mean={araw.mean():.3f} p98={np.percentile(araw,98):.3f}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — Tank mask
    # Strategy: multi-pass HSV + GrabCut refinement so the green line
    # always hugs the real outer tank boundary even when edges are dark.
    # ══════════════════════════════════════════════════════════════════
    hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

    # Pass 1 – broad HSV range captures the full grey tank body including
    #           slightly darker edges (v_min low) and bright highlights.
    # FIX: v_max is now read from settings (default 255) so the bright
    #      near-white top of the tank is no longer excluded.
    cm = ((hsv[:,:,1] <= SMAX) &
          (hsv[:,:,2] >= VMIN) &
          (hsv[:,:,2] <= VMAX)).astype(np.uint8)

    # FIX: closing kernel increased from (45,45) to (55,55) so the bright
    #      top portion of the tank is bridged into the main body contour.
    cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55)))
    # Open removes small background blobs
    cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    # ── Candidate selection ───────────────────────────────────────────
    # The tank must be: (a) large, (b) centred, (c) roughly convex.
    # This eliminates the background wall even when it shares the grey tone.
    img_cx, img_cy = w / 2.0, h / 2.0
    ctrs, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1
    for c in ctrs:
        area = cv2.contourArea(c)
        # Must cover at least TMIN fraction of the frame
        if area < h * w * TMIN:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        ccx = M["m10"] / M["m00"]
        ccy = M["m01"] / M["m00"]
        # Centroid within 40 % of image size from centre
        if abs(ccx - img_cx) > 0.40 * w or abs(ccy - img_cy) > 0.40 * h:
            continue
        # Prefer the largest centred blob
        if area > best_score:
            best_score = area
            best = c

    tmask = np.zeros((h, w), dtype=np.uint8)

    if best is not None:
        # Convex hull = guaranteed no-jagged, no-outside-tank outline
        hull = cv2.convexHull(best)

        # Fill the hull into tmask
        cv2.drawContours(tmask, [hull], -1, 1, thickness=cv2.FILLED)

        # ── Optional GrabCut refinement ───────────────────────────────
        # Shrink the hull slightly to create a definite-foreground seed,
        # then let GrabCut snap to the real tank edge.
        try:
            gc_mask = np.zeros((h, w), dtype=np.uint8)
            # Sure background = outside the hull bounding box with margin
            bx, by, bw2, bh2 = cv2.boundingRect(hull)
            margin = max(15, int(min(bw2, bh2) * 0.04))
            sure_fg = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(sure_fg,
                             [cv2.erode(hull,
                                        cv2.getStructuringElement(
                                            cv2.MORPH_ELLIPSE, (margin*2, margin*2)))],
                             -1, 255, thickness=cv2.FILLED)
            gc_mask[tmask == 0] = cv2.GC_BGD          # definite background
            gc_mask[tmask == 1] = cv2.GC_PR_FGD       # probable foreground
            gc_mask[sure_fg == 255] = cv2.GC_FGD      # definite foreground
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(orig, gc_mask, None,
                        bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            gc_result = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0
            ).astype(np.uint8)
            # Only accept GrabCut result if it is larger (more complete)
            if gc_result.sum() > tmask.sum() * 0.8:
                tmask = gc_result
                # Re-find contour on the refined mask
                gc_ctrs, _ = cv2.findContours(tmask, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                if gc_ctrs:
                    best = max(gc_ctrs, key=cv2.contourArea)
                    hull = cv2.convexHull(best)
        except Exception:
            pass  # GrabCut failed — keep hull-based mask

        # Fill tmask from final hull (in case GrabCut updated it)
        tmask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmask, [hull], -1, 1, thickness=cv2.FILLED)

        # ── Draw the GREEN outline on the OUTPUT image ────────────────
        # Use the convex hull directly — it is always on the real boundary.
        cv2.drawContours(out, [hull], -1, (0, 200, 0), 3)

        # Erode tmask inward so defect analysis stays well inside the tank.
        # ERODE_PX is configurable — default 20px pushes boundary inward enough
        # to avoid seam/rim false positives even when camera is close to the tank.
        ek = max(3, ERODE_PX)
        tmask = cv2.erode(tmask,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek)))
    else:
        # Fallback: no tank found — use the central 80 % of the image
        pad_h, pad_w = int(h * 0.10), int(w * 0.10)
        tmask[pad_h:h-pad_h, pad_w:w-pad_w] = 1

    # ── Edge exclusions — strip all boundary zones that produce false positives ──
    # Left: pipe column
    tmask[:, :int(w * PXF)] = 0
    # Right: metallic rim
    tmask[:, int(w * (1.0 - RXF)):] = 0
    # Top: tank seam/ridge — the physical rim along the top edge of the tank
    #      always has a strong texture gradient that PaDiM flags as anomalous.
    tmask[:int(h * TXF), :] = 0
    # Bottom: lower boundary corners where the tank curves away from camera
    tmask[int(h * (1.0 - BXF)):, :] = 0

    # STEP 2 — opener
    omask = np.zeros((h,w), dtype=np.uint8)
    found = False; det = False

    if yolo_model is not None:
        res = yolo_model.predict(source=image_path, conf=YCONF, save=False, verbose=False)
        for r in res:
            if r.boxes is None: continue
            for box,cls,sc in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()):
                if int(cls) != OCI: continue
                det = True
                x1,y1,x2,y2 = box.astype(int)
                x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
                lbl = f"opener {sc:.2f}"
                (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(out,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.rectangle(out,(x1,y1-th-6),(x1+tw,y1),(255,0,0),-1)
                cv2.putText(out,lbl,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
                omask[max(0,y1-OPT):min(h,y2+OPB), max(0,x1-OPL):min(w,x2+OPR)] = 1
                found = True

    if not found:
        gc = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        _,dm = cv2.threshold(gc, 60, 255, cv2.THRESH_BINARY_INV)
        dm = cv2.bitwise_and(dm, dm, mask=tmask)
        dm = cv2.morphologyEx(dm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
        dm = cv2.morphologyEx(dm, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        dct,_ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bc,ba = None,0
        for cnt in dct:
            a = cv2.contourArea(cnt)
            if a<3000: continue
            bx,by,bw,bh = cv2.boundingRect(cnt)
            if np.sqrt((bx+bw/2-w/2)**2+(by+bh/2-h/2)**2) > diag*HCF: continue
            if max(bw,bh)/(min(bw,bh)+1) > 2.5: continue
            if a>ba: ba=a; bc=cnt
        if bc is not None:
            bx,by,bw,bh = cv2.boundingRect(bc); EX=220
            x1p,y1p = max(0,bx-EX),max(0,by-EX)
            x2p,y2p = min(w,bx+bw+EX),min(h,by+bh+EX)
            omask[y1p:y2p, x1p:x2p]=1
            cv2.rectangle(out,(x1p,y1p),(x2p,y2p),(255,0,0),2)
            found=det=True

    if not found:
        bl = cv2.GaussianBlur(cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY),(9,9),2)
        ccs = cv2.HoughCircles(bl, cv2.HOUGH_GRADIENT, 1, min(h,w)//4,
                                param1=80, param2=40, minRadius=HMN, maxRadius=HMX)
        if ccs is not None:
            for cx,cy,r in np.round(ccs[0]).astype(int):
                if np.sqrt((cx-w/2)**2+(cy-h/2)**2)>diag*HCF: continue
                if cy>=h or cx>=w or tmask[cy,cx]==0: continue
                x1p,y1p=max(0,cx-r-HPL),max(0,cy-r-HPT)
                x2p,y2p=min(w,cx+r+HPR),min(h,cy+r+HPB)
                omask[y1p:y2p,x1p:x2p]=1
                cv2.rectangle(out,(x1p,y1p),(x2p,y2p),(255,0,0),2)
                found=det=True; break

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 — Anomaly map normalisation + thresholding
    #
    # MODES (threshold_mode setting):
    #   "percentile" (recommended) — threshold = Nth percentile of inside-tank
    #                scores. Every image gets a threshold relative to its own
    #                distribution → equal detection chance on all zones.
    #   "adaptive"   — threshold = mean + K*std (legacy, K = "threshold" setting)
    #   "ratio"      — normalise by p98, compare to TH (legacy)
    #
    # FIX 1: heatmap_normalize=True → min-max normalise BEFORE any threshold.
    #   This prevents one bright region dominating the dynamic range and
    #   ensures faint defects elsewhere still cross the threshold.
    # FIX 2: threshold_mode="percentile" → top X% flagged regardless of absolute
    #   score. Even a surface with very uniform scores will flag the true outliers.
    # ══════════════════════════════════════════════════════════════════
    THRESHOLD_MODE  = str(cfg.get("threshold_mode",       "percentile")).lower()
    HMAP_NORMALIZE  = bool(cfg.get("heatmap_normalize",   True))
    TH_PERCENTILE   = float(cfg.get("threshold_percentile", 97.5))
    K = TH   # fallback K for adaptive mode

    combined = np.clip(
        tmask.astype(np.int32) - omask.astype(np.int32), 0, 1
    ).astype(np.float32)   # safe-zone mask (h, w)

    # Mild smooth first — kills single-pixel salt noise before statistics
    araw_s = cv2.GaussianBlur(araw, (7, 7), 0)

    # ── FIX 1: Min-max normalise heatmap inside tank mask ─────────────
    # After normalisation every pixel value is in [0, 1] relative to this
    # image's own range. One hot-spot can no longer push the threshold so
    # high that scratches elsewhere become invisible.
    inside_mask = combined > 0.5
    inside_vals = araw_s[inside_mask]

    if inside_vals.size > 0:
        inside_mean = float(np.mean(inside_vals))
        inside_std  = float(np.std(inside_vals))
        inside_p98  = float(np.percentile(inside_vals, 98))
        amin        = float(inside_vals.min())
        amax        = float(inside_vals.max())
    else:
        inside_mean = float(araw_s.mean())
        inside_std  = float(araw_s.std())
        inside_p98  = float(araw_s.max())
        amin        = float(araw_s.min())
        amax        = float(araw_s.max())

    if HMAP_NORMALIZE and (amax - amin) > 1e-6:
        araw_norm = np.clip((araw_s - amin) / (amax - amin), 0.0, 1.0).astype(np.float32)
        print(f"[PaDiM] FIX1 min-max normalised: amin={amin:.3f} amax={amax:.3f}")
    else:
        araw_norm = np.clip(araw_s / max(inside_p98, 1e-6), 0.0, 1.0).astype(np.float32)

    # ── FIX 2: Percentile-based threshold ─────────────────────────────
    # Compute threshold from the normalised inside-tank distribution.
    # Top (100 - TH_PERCENTILE)% of scores are flagged as anomalous.
    # This is image-adaptive AND scale-invariant — works equally well
    # regardless of overall surface brightness or contrast.
    norm_inside = araw_norm[inside_mask]

    if THRESHOLD_MODE == "percentile":
        if norm_inside.size > 0:
            effective_th = float(np.percentile(norm_inside, TH_PERCENTILE))
        else:
            effective_th = 0.97
        defect_mask_input = araw_norm * combined
        print(f"[PaDiM] mode=percentile  p{TH_PERCENTILE:.1f}={effective_th:.4f}  "
              f"inside: mean={inside_mean:.3f} std={inside_std:.3f} p98={inside_p98:.3f}")

    elif THRESHOLD_MODE == "adaptive":
        norm_mean = float(norm_inside.mean()) if norm_inside.size > 0 else 0.5
        norm_std  = float(norm_inside.std())  if norm_inside.size > 0 else 0.1
        effective_th = norm_mean + K * norm_std
        defect_mask_input = araw_norm * combined
        print(f"[PaDiM] mode=adaptive  th(mean+{K}*std)={effective_th:.4f}  "
              f"norm_mean={norm_mean:.3f} norm_std={norm_std:.3f}")

    else:
        # ratio mode (legacy) — threshold is a raw fraction
        effective_th = TH
        defect_mask_input = araw_norm * combined
        print(f"[PaDiM] mode=ratio  th={effective_th:.3f}")

    # Keep a [0,1] copy for heatmap rendering
    ar = (araw_norm * combined).astype(np.float32)

    print(f"[PaDiM] effective_threshold={effective_th:.4f}  "
          f"inside_p98_norm={float(np.percentile(norm_inside, 98)) if norm_inside.size>0 else 0:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 — Defect bounding boxes
    # ══════════════════════════════════════════════════════════════════
    # Improved heatmap → binary mask pipeline
    REFINE_BLK  = max(3, int(cfg.get("refine_blur_k",       5)) | 1)
    REFINE_OPEN = max(1, int(cfg.get("refine_morph_open",   3)))
    REFINE_CLOS = max(1, int(cfg.get("refine_morph_close",  9)))

    dm_smooth = cv2.GaussianBlur(defect_mask_input.astype(np.float32),
                                 (REFINE_BLK, REFINE_BLK), 0)

    bm = (dm_smooth > effective_th).astype(np.uint8)
    bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (REFINE_OPEN, REFINE_OPEN)))
    bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (REFINE_CLOS, REFINE_CLOS)))

    excl_top    = int(h * TXF)
    excl_bottom = int(h * (1.0 - BXF))
    excl_left   = int(w * PXF)
    excl_right  = int(w * (1.0 - max(RXF, 0.10)))  # minimum 10 % right exclusion

    # Scratch-aware CC filter params
    SCR_MIN_A   = int(cfg.get("scratch_min_area",   80))
    SCR_MAX_A   = int(cfg.get("scratch_max_area",   15000))
    SCR_MIN_ASP = float(cfg.get("scratch_min_asp",  1.2))
    DOT_MAX_A   = int(cfg.get("dot_max_area",       3000))

    # 2-stage bbox refinement params
    REFINE_EN   = bool(cfg.get("refine_bbox_enabled", True))
    REFINE_PAD  = int(cfg.get("refine_pad",           12))
    REFINE_CL   = int(cfg.get("refine_canny_low",     20))
    REFINE_CH   = int(cfg.get("refine_canny_high",    60))

    # FIX 3+4+5 zone detection params
    ZONE_EN     = bool(cfg.get("zone_detect_enabled",  True))
    ZONE_PCT    = float(cfg.get("zone_percentile",      97.5))
    ZONE_MINA   = int(cfg.get("zone_min_area",          80))

    gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── FIX 3: Build 4-zone mask (Top / Bottom / Left / Right halves) ────
    # Each zone gets its own independent percentile threshold so that one
    # bright zone cannot suppress detection in a dimmer zone.
    # Zones overlap at the midpoint — a defect near centre is caught by both.
    # ─────────────────────────────────────────────────────────────────────
    mid_y = (excl_top + excl_bottom) // 2
    mid_x = (excl_left + excl_right) // 2

    zone_masks = {}
    _safe = combined > 0.5
    _top_half  = np.zeros((h, w), bool); _top_half[excl_top:mid_y,   excl_left:excl_right] = True
    _bot_half  = np.zeros((h, w), bool); _bot_half[mid_y:excl_bottom, excl_left:excl_right] = True
    _lft_half  = np.zeros((h, w), bool); _lft_half[excl_top:excl_bottom, excl_left:mid_x]  = True
    _rgt_half  = np.zeros((h, w), bool); _rgt_half[excl_top:excl_bottom, mid_x:excl_right] = True
    zone_masks = {
        "TOP": _safe & _top_half,
        "BOT": _safe & _bot_half,
        "LFT": _safe & _lft_half,
        "RGT": _safe & _rgt_half,
    }

    # Build per-zone binary anomaly maps using zone-local percentile threshold
    # FIX 4: No top-N cap — all blobs above min area are evaluated
    # FIX 5: Area-only filter — area is the survival criterion; score is Gate 7 only
    zone_bm_union = np.zeros((h, w), np.uint8)

    for zname, zmask in zone_masks.items():
        zvals = defect_mask_input[zmask]
        if zvals.size < 50:
            continue
        z_th = float(np.percentile(zvals, ZONE_PCT))
        z_bm = ((defect_mask_input > z_th) & zmask).astype(np.uint8)
        z_bm = cv2.morphologyEx(z_bm, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REFINE_OPEN, REFINE_OPEN)))
        z_bm = cv2.morphologyEx(z_bm, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REFINE_CLOS, REFINE_CLOS)))
        zone_bm_union = cv2.bitwise_or(zone_bm_union, z_bm)
        n_pix = int(z_bm.sum())
        print(f"  [Zone {zname}] th_p{ZONE_PCT}={z_th:.4f}  active_px={n_pix}")

    # Merge: global threshold map OR zone union — catches both strong global
    # defects and subtle zone-local defects that the global threshold misses
    if ZONE_EN:
        bm = cv2.bitwise_or(bm, zone_bm_union)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bm, connectivity=8)
    dcnt = 0; dboxes = []

    # Gate 9 contrast map: gradient magnitude (Sobel) catches faint surface marks
    # better than raw pixel std because it responds to local edges/texture changes
    # rather than global brightness variation.
    if USE_GRAD:
        sx = cv2.Sobel(gray_orig, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray_orig, cv2.CV_32F, 0, 1, ksize=3)
        contrast_map = np.sqrt(sx**2 + sy**2)   # gradient magnitude per pixel
        contrast_label = "grad_mag_std"
    else:
        contrast_map = gray_orig
        contrast_label = "pixel_std"

    for lab in range(1, num_labels):
        area_cnt = int(stats[lab, cv2.CC_STAT_AREA])
        x  = int(stats[lab, cv2.CC_STAT_LEFT]);   y  = int(stats[lab, cv2.CC_STAT_TOP])
        wb = int(stats[lab, cv2.CC_STAT_WIDTH]);  hb = int(stats[lab, cv2.CC_STAT_HEIGHT])
        x2b = min(x + wb, w - 1);                 y2b = min(y + hb, h - 1)
        if y2b <= y or x2b <= x: continue

        roi_comb  = combined[y:y2b, x:x2b]
        roi_ar    = ar[y:y2b, x:x2b]
        roi_omask = omask[y:y2b, x:x2b]
        if roi_comb.size == 0: continue

        # ── Gate 1 — minimum area (FIX 3): scratch-aware lower bound ──────
        # Use scratch_min_area (80px default) — smaller than the legacy MDA
        # so thin scratch segments aren't pruned before they can be joined.
        # Legacy MDA still applies as a hard floor.
        min_area_eff = min(MDA, SCR_MIN_A)
        if area_cnt < min_area_eff:
            print(f"  [Gate1-FAIL] lab={lab} area={area_cnt} < min_area={min_area_eff}")
            continue

        # ── Gate 2 — inside tank: ≥85 % of box pixels must be inside the tank mask ──
        if roi_comb.mean() < 0.85:
            print(f"  [Gate2-FAIL] lab={lab} tank_frac={roi_comb.mean():.2f}")
            continue

        # ── Gate 3 — not on opener or fuel-cap exclusion zone ──
        if roi_omask.max() > 0:
            print(f"  [Gate3-FAIL] lab={lab} overlaps opener mask")
            continue

        # ── Gate 4 — edge exclusion zones (shadow/rim always score high) ──
        if y < excl_top  or y2b > excl_bottom:
            print(f"  [Gate4-FAIL] lab={lab} y={y}-{y2b} edge top/bot")
            continue
        if x < excl_left or x2b > excl_right:
            print(f"  [Gate4-FAIL] lab={lab} x={x}-{x2b} edge left/right")
            continue

        # ── Gate 5 — centroid must land inside the safe tank area ──
        ccy = max(0, min((y + y2b) // 2, h - 1))
        ccx = max(0, min((x + x2b) // 2, w - 1))
        if combined[ccy, ccx] < 0.5:
            print(f"  [Gate5-FAIL] lab={lab} centroid outside safe zone")
            continue

        # ── Gate 6 — FIX 3: Scratch-aware shape gate ────────────────────────
        # Two classes of real defects:
        #   A. Scratches/marks: elongated (asp >= SCR_MIN_ASP), can be small area
        #   B. Pits/dents: roughly round (asp < SCR_MIN_ASP), must be >= dot_max_area threshold
        # Reject: tiny round blobs that match neither class (dust / JPEG noise)
        asp = float(max(wb, hb)) / (float(min(wb, hb)) + 1e-5)
        is_elongated = asp >= SCR_MIN_ASP
        is_large_dot = (asp < SCR_MIN_ASP) and (area_cnt >= 500)
        if not is_elongated and not is_large_dot:
            print(f"  [Gate6-FAIL] lab={lab} tiny round blob asp={asp:.2f} area={area_cnt} (not scratch, not large pit)")
            continue
        # Also reject blobs that are far too large to be a scratch (lighting gradient)
        if area_cnt > SCR_MAX_A:
            print(f"  [Gate6-FAIL] lab={lab} area={area_cnt} > SCR_MAX_A={SCR_MAX_A} (gradient blob)")
            continue

        # ── Gate 7 — peak anomaly score must exceed threshold × margin ──
        # GATE_PEAK_MARGIN default 0.70 (was 0.80) — lower catches weaker defect peaks
        roi_dm = defect_mask_input[y:y2b, x:x2b]
        if roi_dm.max() < effective_th * GATE_PEAK_MARGIN:
            print(f"  [Gate7-FAIL] lab={lab} peak={roi_dm.max():.2f} < {GATE_PEAK_MARGIN}*th={effective_th*GATE_PEAK_MARGIN:.2f}")
            continue

        # ── Gate 8 — maximum area: large blobs are lighting gradients, NOT defects ──
        # Real surface defects are small and localised.
        # Gradient/shadow regions cover thousands of pixels smoothly.
        if area_cnt > MXA:
            print(f"  [Gate8-FAIL] lab={lab} area={area_cnt} > MXA={MXA} (gradient blob)")
            continue

        # ── Gate 9 — local texture contrast: gradients are smooth, defects are sharp ──
        # Measure std of gradient magnitude inside the bounding box.
        # A smooth lighting gradient has near-zero gradient variation;
        # a scratch/dent/circular mark has elevated gradient edges.
        roi_gray = contrast_map[y:y2b, x:x2b]
        roi_std  = float(roi_gray.std()) if roi_gray.size > 0 else 0.0
        if roi_std < MSTD:
            print(f"  [Gate9-FAIL] lab={lab} {contrast_label}={roi_std:.2f} < MSTD={MSTD} (smooth gradient)")
            continue

        # ── Gate 10 — maximum aspect ratio: extreme stripe/bar = not a real defect ──
        # Scratches can be elongated (asp up to 4); a lighting stripe is asp 6–20+.
        if asp > MASP:
            print(f"  [Gate10-FAIL] lab={lab} asp={asp:.2f} > MASP={MASP} (stripe artifact)")
            continue

        roi_dm_local = defect_mask_input[y:y2b, x:x2b]
        print(f"  [DEFECT] lab={lab} area={area_cnt} asp={asp:.2f} "
              f"{contrast_label}={roi_std:.2f} peak={roi_dm_local.max():.1f} th={effective_th:.1f}")

        # ── FIX 4: 2-stage bbox refinement ───────────────────────────────
        # Stage 1 bbox (rx,ry,rx2,ry2) = the rough PaDiM connected-component box.
        # Stage 2: crop that region (with padding), apply Sobel+Canny edge map,
        # find the tightest enclosing rectangle of the actual edge pixels.
        # This gives pixel-accurate bounding boxes instead of blob-inflated ones.
        rx, ry, rx2, ry2 = x, y, x2b, y2b   # start with rough bbox

        if REFINE_EN and (rx2 - rx) > 4 and (ry2 - ry) > 4:
            try:
                # Crop with padding (clamped to image bounds)
                px1 = max(0, rx - REFINE_PAD);  py1 = max(0, ry - REFINE_PAD)
                px2 = min(w, rx2 + REFINE_PAD); py2 = min(h, ry2 + REFINE_PAD)
                crop_g = cv2.cvtColor(orig[py1:py2, px1:px2], cv2.COLOR_BGR2GRAY)

                # Illumination-normalise crop so scratch edges are visible
                k101 = max(3, min(crop_g.shape[0]//2*2-1, 51))  # safe odd kernel
                bg_c = cv2.GaussianBlur(crop_g.astype(np.float32), (k101, k101), 0)
                norm_c = np.clip(crop_g.astype(np.float32) - bg_c * 0.8 + 128,
                                 0, 255).astype(np.uint8)

                # Sobel edge magnitude
                sx_c = cv2.Sobel(norm_c, cv2.CV_32F, 1, 0, ksize=3)
                sy_c = cv2.Sobel(norm_c, cv2.CV_32F, 0, 1, ksize=3)
                edge_c = np.sqrt(sx_c**2 + sy_c**2)
                edge_c = cv2.normalize(edge_c, None, 0, 255,
                                       cv2.NORM_MINMAX).astype(np.uint8)

                # Canny on top of Sobel for clean edge pixels
                canny_c = cv2.Canny(edge_c, REFINE_CL, REFINE_CH)

                # Also include the PaDiM heatmap blob within this crop as a guide
                heat_crop = (defect_mask_input[py1:py2, px1:px2] > effective_th * 0.7)
                heat_crop = (heat_crop * 255).astype(np.uint8)

                # Fuse: edge pixels OR anomaly pixels
                fused = cv2.bitwise_or(canny_c, heat_crop)
                fused = cv2.dilate(fused,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

                # Find tight bounding rect of non-zero pixels in fused map
                nz = cv2.findNonZero(fused)
                if nz is not None and len(nz) >= 4:
                    fx, fy, fw_r, fh_r = cv2.boundingRect(nz)
                    # Convert back to full-image coordinates
                    rfx1 = px1 + fx
                    rfy1 = py1 + fy
                    rfx2 = min(w - 1, rfx1 + fw_r)
                    rfy2 = min(h - 1, rfy1 + fh_r)
                    # Only accept refined box if it is smaller (tighter) than rough
                    if (rfx2 - rfx1) > 2 and (rfy2 - rfy1) > 2:
                        area_refined = (rfx2 - rfx1) * (rfy2 - rfy1)
                        area_rough   = (rx2 - rx) * (ry2 - ry)
                        # Accept refinement if it shrinks the box or extends it
                        # only modestly (< 20% expansion per side)
                        ok = (rfx1 >= rx - REFINE_PAD and rfy1 >= ry - REFINE_PAD and
                              rfx2 <= rx2 + REFINE_PAD and rfy2 <= ry2 + REFINE_PAD)
                        if ok:
                            rx, ry, rx2, ry2 = rfx1, rfy1, rfx2, rfy2
                            print(f"    [Refine] bbox tightened: "
                                  f"({x},{y})-({x2b},{y2b}) → ({rx},{ry})-({rx2},{ry2})")
            except Exception as _ref_e:
                print(f"    [Refine] skipped ({_ref_e})")

        cv2.rectangle(out, (rx, ry), (rx2, ry2), (0, 0, 255), 2)
        cv2.putText(out, f"{roi_dm_local.max():.1f}/{effective_th:.1f} c={roi_std:.0f}",
                    (rx+2, ry+13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        dcnt += 1
        dboxes.append({"x": int(rx), "y": int(ry), "x2": int(rx2), "y2": int(ry2),
                       "peak": round(float(roi_dm_local.max()), 2),
                       "roi_std": round(roi_std, 2),
                       "area": area_cnt})

    # ══════════════════════════════════════════════════════════════════
    # STEP 5 — White / light-coloured surface mark detector (CLAHE edition)
    # ──────────────────────────────────────────────────────────────────
    # These marks are very faint thin bright lines (only ~1-5 gray levels
    # above background) — completely invisible without contrast enhancement.
    #
    # Validated pipeline (from pixel analysis of real tank images):
    #   1. CLAHE (clipLimit=6, tileGrid=16x16) amplifies local contrast so
    #      thin white scratch/circle lines become clearly visible.
    #   2. Dual-background subtraction removes concentric banding rings:
    #        bg_iso  = GaussianBlur(81x81)     — isotropic background
    #        bg_horz = GaussianBlur(1x201)     — captures horizontal bands
    #        background = max(bg_iso, bg_horz) — always subtract smoothest
    #   3. dev = clip(enhanced - background, 0) — positive = brighter than bg
    #   4. Threshold at WM_DELTA on dev map inside safe-zone mask.
    #   5. Morphology + connected-component geometry gates.
    #   Result: yellow bounding boxes on the output image.
    #
    # Settings (tunable via /api/settings):
    #   white_mark_enabled  : True/False  (default True)
    #   wm_delta            : deviation threshold after CLAHE (default 25)
    #   wm_clahe_clip       : CLAHE clip limit (default 6.0)
    #   wm_clahe_tile       : CLAHE tile grid size (default 16)
    #   wm_blur_radius      : isotropic background blur radius (default 81)
    #   wm_horz_blur        : horizontal background blur height (default 201)
    #   wm_min_area         : minimum blob area px (default 150)
    #   wm_max_area         : maximum blob area px (default 40000)
    #   wm_min_asp / wm_max_asp : aspect ratio range (default 1.0 – 8.0)
    # ══════════════════════════════════════════════════════════════════
    WM_EN        = bool(cfg.get("white_mark_enabled",   True))
    WM_DELTA     = float(cfg.get("wm_delta",            25.0))
    WM_CLAHE_CL  = float(cfg.get("wm_clahe_clip",       6.0))
    WM_CLAHE_T   = int(cfg.get("wm_clahe_tile",         16))
    WM_BLUR      = int(cfg.get("wm_blur_radius",        81))
    WM_HORZ      = int(cfg.get("wm_horz_blur",          201))
    WM_MINA      = int(cfg.get("wm_min_area",           150))
    WM_MAXA      = int(cfg.get("wm_max_area",           40000))
    WM_MIN_ASP   = float(cfg.get("wm_min_asp",          1.0))
    WM_MAX_ASP   = float(cfg.get("wm_max_asp",          8.0))

    wm_boxes = []
    wm_cnt   = 0

    if WM_EN:
        # ── 1. CLAHE contrast enhancement ─────────────────────────────
        gray_wm  = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        clahe_wm = cv2.createCLAHE(clipLimit=WM_CLAHE_CL,
                                    tileGridSize=(WM_CLAHE_T, WM_CLAHE_T))
        enh_wm   = clahe_wm.apply(gray_wm).astype(np.float32)

        # ── 2. Dual-background subtraction ────────────────────────────
        iso_k    = max(3, WM_BLUR | 1)
        bg_iso   = cv2.GaussianBlur(enh_wm, (iso_k, iso_k), 0)
        horz_k   = max(3, WM_HORZ | 1)
        bg_horz  = cv2.GaussianBlur(enh_wm, (1, horz_k), 0)
        # max() background kills horizontal banding without losing mark signal
        bg_comb  = np.maximum(bg_iso, bg_horz)
        wm_dev   = np.clip(enh_wm - bg_comb, 0, 255).astype(np.float32)

        # ── 3. Safe-zone mask (same edge exclusions as PaDiM gates) ───
        wm_safe = combined.copy()
        wm_safe[:excl_top,  :]   = 0
        wm_safe[excl_bottom:, :] = 0
        wm_safe[:, :excl_left]   = 0
        wm_safe[:, excl_right:]  = 0

        # ── 4. Threshold + morphological clean-up ─────────────────────
        wm_bin = ((wm_dev > WM_DELTA) & (wm_safe > 0.5)).astype(np.uint8)
        wm_bin = cv2.morphologyEx(wm_bin, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        wm_bin = cv2.morphologyEx(wm_bin, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # ── 5. Connected components + geometry gates ───────────────────
        wm_labels, _, wm_stats, _ = cv2.connectedComponentsWithStats(
            wm_bin, connectivity=8)

        print(f"[WhiteMark] {wm_labels-1} raw blobs  "
              f"(delta={WM_DELTA} clahe_clip={WM_CLAHE_CL})")

        for wlab in range(1, wm_labels):
            wa   = int(wm_stats[wlab, cv2.CC_STAT_AREA])
            wx   = int(wm_stats[wlab, cv2.CC_STAT_LEFT])
            wy   = int(wm_stats[wlab, cv2.CC_STAT_TOP])
            wwb  = int(wm_stats[wlab, cv2.CC_STAT_WIDTH])
            whb  = int(wm_stats[wlab, cv2.CC_STAT_HEIGHT])
            wx2  = min(wx + wwb, w - 1)
            wy2  = min(wy + whb, h - 1)
            if wy2 <= wy or wx2 <= wx: continue

            # Gate: area range
            if wa < WM_MINA or wa > WM_MAXA: continue
            # Gate: not on opener
            if omask[max(0,(wy+wy2)//2), max(0,(wx+wx2)//2)] > 0: continue
            # Gate: aspect ratio
            wasp = float(max(wwb, whb)) / (float(min(wwb, whb)) + 1e-5)
            if wasp < WM_MIN_ASP or wasp > WM_MAX_ASP: continue
            # Gate: peak dev must be clearly above threshold
            roi_dev  = wm_dev[wy:wy2, wx:wx2]
            peak_dev = float(roi_dev.max()) if roi_dev.size > 0 else 0.0
            if peak_dev < WM_DELTA * 0.85: continue
            # Gate: density — rejects large sparse blobs (diffuse banding leakage)
            box_area = (wy2-wy) * (wx2-wx)
            density  = wa / max(box_area, 1)
            if density < 0.005: continue

            print(f"  [WHITE-MARK] wlab={wlab} area={wa} asp={wasp:.2f} "
                  f"peak_dev={peak_dev:.1f} density={density:.3f}")

            cv2.rectangle(out, (wx, wy), (wx2, wy2), (0, 220, 255), 2)
            cv2.putText(out, f"mark {peak_dev:.0f}",
                        (wx+2, wy+14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 220, 255), 1, cv2.LINE_AA)
            wm_cnt += 1
            wm_boxes.append({"x": int(wx), "y": int(wy), "x2": int(wx2), "y2": int(wy2),
                              "type": "white_mark", "peak_dev": round(peak_dev,2), "area": wa})

        print(f"[WhiteMark] {wm_cnt} marks after geometry filter")

    # ── Legend on output image ────────────────────────────────────────
    cv2.putText(out, "RED=PaDiM defect  YELLOW=surface mark",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)

    has = dcnt > 0 or wm_cnt > 0
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fn  = f"result_{ts}.jpg"
    dst = PROCESSED_DEFECT if has else PROCESSED_CLEAN
    cv2.imwrite(str(dst/fn), out)
    hm  = (ar*255).astype("uint8"); hc=cv2.applyColorMap(hm, cv2.COLORMAP_JET); hc[combined==0]=0
    cv2.imwrite(str(dst/f"heat_{ts}.jpg"), hc)

    return {"filename":fn, "folder":"processed_defect" if has else "processed_clean",
            "has_defects":has,
            "defect_count":dcnt, "defect_boxes":dboxes,
            "white_mark_count":wm_cnt, "white_mark_boxes":wm_boxes,
            "opener_detected":det, "timestamp":datetime.now().isoformat(),
            "padim_stats": {
                "mode": THRESHOLD_MODE,
                "inside_mean": round(inside_mean, 3),
                "inside_std":  round(inside_std,  3),
                "inside_p98":  round(inside_p98,  3),
                "fused_max":   round(float(araw.max()), 3),
                "effective_th":round(effective_th, 3),
            }}

# ── Camera (raspi_vision_controller.py protocol) ─────────────
def _recvall(sock, n):
    buf = bytearray()
    while len(buf)<n:
        p = sock.recv(n-len(buf))
        if not p: return None
        buf.extend(p)
    return bytes(buf)

def _send_cmd(sock, cmd):
    d=json.dumps(cmd).encode(); sock.sendall(struct.pack(">I",len(d))+d)

def _recv_resp(sock):
    r=_recvall(sock,4)
    if not r: return None
    return json.loads(_recvall(sock,struct.unpack(">I",r)[0]).decode())

def camera_connect():
    global camera_socket, camera_connected
    with camera_lock:
        try:
            s=_socket.socket(_socket.AF_INET,_socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((settings["camera_ip"],int(settings["camera_port"])))
            s.settimeout(None); _recv_resp(s)
            camera_socket=s; camera_connected=True
            socketio.emit("camera_status",{"connected":True})
            print(f"[Camera] Connected to {settings['camera_ip']}:{settings['camera_port']}")
            return True
        except Exception as e:
            camera_socket=None; camera_connected=False
            socketio.emit("camera_status",{"connected":False})
            print(f"[Camera] Connect failed: {e}"); return False

def camera_disconnect():
    global camera_socket, camera_connected
    with camera_lock:
        if camera_socket:
            try: camera_socket.close()
            except: pass
            camera_socket=None
        camera_connected=False
    socketio.emit("camera_status",{"connected":False})

def camera_capture():
    global camera_socket, camera_connected
    with camera_lock:
        if not camera_connected or camera_socket is None: return None
        try:
            _send_cmd(camera_socket,{"action":"capture"})
            resp=_recv_resp(camera_socket)
            if not resp or resp.get("status")!="success": return None
            imglen=struct.unpack(">I",_recvall(camera_socket,4))[0]
            data=_recvall(camera_socket,imglen)
            if not data: return None
            ts=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path=WATCH_FOLDER/f"img_{ts}.jpg"; path.write_bytes(data)
            return str(path)
        except Exception as e:
            print(f"[Camera] Capture error: {e}")
            camera_socket=None; camera_connected=False
            socketio.emit("camera_status",{"connected":False}); return None

def camera_grab_frame():
    """Grab a single JPEG frame from the camera socket (non-destructive, no file write)."""
    global camera_socket, camera_connected
    with camera_lock:
        if not camera_connected or camera_socket is None:
            return None
        try:
            _send_cmd(camera_socket, {"action": "capture"})
            resp = _recv_resp(camera_socket)
            if not resp or resp.get("status") != "success":
                return None
            imglen = struct.unpack(">I", _recvall(camera_socket, 4))[0]
            data   = _recvall(camera_socket, imglen)
            return data if data else None
        except Exception as e:
            print(f"[Camera] Frame grab error: {e}")
            camera_socket   = None
            camera_connected = False
            socketio.emit("camera_status", {"connected": False})
            return None

def _live_stream_loop():
    """Background thread: continuously pulls frames from the camera and stores
    the latest JPEG in _live_frame_bytes so /api/camera/stream can serve them."""
    global _live_frame_bytes, _live_frame_ts
    print("[LiveStream] Thread started.")
    while True:
        if camera_connected:
            data = camera_grab_frame()
            if data:
                with _live_frame_lock:
                    _live_frame_bytes = data
                    _live_frame_ts    = time.time()
                time.sleep(0.08)   # ~12 fps max
                continue
        time.sleep(0.5)   # not connected – poll slowly

# ── Watch folder thread ───────────────────────────────────────
_done_set = set()

def _watch_loop():
    print("[Watch] Thread started.")
    while True:
        try:
            for fp in sorted(WATCH_FOLDER.iterdir()):
                if fp.suffix.lower() not in (".jpg",".jpeg",".png"): continue
                if fp.name in _done_set: continue
                _done_set.add(fp.name)
                socketio.emit("processing_status",{"active":True})
                print(f"[Watch] Processing {fp.name}")
                result=None
                try:
                    result=run_inference(str(fp))
                    processed_images.insert(0,result)
                    if len(processed_images)>200: processed_images.pop()
                    socketio.emit("new_result",result)
                    if result["has_defects"]:
                        socketio.emit("defect_alert",result)
                    print(f"[Watch] Done — defects:{result['defect_count']}")
                except Exception:
                    traceback.print_exc()
                finally:
                    socketio.emit("processing_status",{"active":False})
                    dst = PROCESSED_DEFECT if (result and result["has_defects"]) else PROCESSED_CLEAN
                    try: shutil.move(str(fp), str(dst/f"raw_{fp.name}"))
                    except: pass
        except Exception: pass
        time.sleep(1)

# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def r_index(): return send_file(BASE_DIR/"index.html")

@app.route("/api/status")
def r_status():
    return jsonify({"camera_connected":camera_connected,"padim_trained":padim_trained,
                    "yolo_loaded":yolo_loaded,"processed_images":processed_images[:50],
                    "settings":settings})

@app.route("/api/padim/retrain", methods=["POST"])
def r_padim_retrain():
    """Delete the saved model and force a full retrain from train_normal/."""
    global padim_trained, padim_mean, padim_inv_cov
    if PADIM_MODEL_PATH.exists():
        PADIM_MODEL_PATH.unlink()
        print("[PaDiM] Saved model deleted — retraining...")
    padim_trained = False; padim_mean = None; padim_inv_cov = None
    threading.Thread(target=init_padim, daemon=True).start()
    return jsonify({"success": True, "message": "Retraining started in background"})

@app.route("/api/padim/diagnose", methods=["POST"])
def r_padim_diagnose():
    """Diagnostic endpoint — returns raw PaDiM scores WITHOUT any gate filtering.

    POST a multipart form with an image file, or pass {"filename": "..."} JSON
    pointing to a file already in watch_folder/processed_defect/processed_clean.

    Returns full score statistics so you can tune threshold and gates:
      inside_mean, inside_std, inside_p98, fused_max, effective_th,
      raw_blobs (list of all connected components BEFORE gates with their
      area, peak, asp, roi_std — so you can see what each gate would reject).
    """
    if not padim_trained:
        return jsonify({"error": "PaDiM not trained"}), 400

    # Accept uploaded file or filename reference
    img_path = None
    if "file" in request.files:
        f = request.files["file"]
        tmp = WATCH_FOLDER / f"diag_{f.filename}"
        f.save(str(tmp))
        img_path = str(tmp)
    elif request.is_json and "filename" in request.json:
        for folder in [WATCH_FOLDER, PROCESSED_DEFECT, PROCESSED_CLEAN]:
            candidate = folder / request.json["filename"]
            if candidate.exists():
                img_path = str(candidate)
                break

    if not img_path:
        return jsonify({"error": "No image provided"}), 400

    try:
        cfg = settings
        PADIM_ENHANCE  = bool(cfg.get("padim_enhance",       False))
        PADIM_ENH_CLIP = float(cfg.get("padim_enhance_clip", 3.0))
        PADIM_ENH_TILE = int(cfg.get("padim_enhance_tile",   8))
        FEAT_RES       = int(cfg.get("padim_feature_res",    64))
        K              = float(cfg["threshold"])

        orig = cv2.imread(img_path)
        h, w = orig.shape[:2]

        def _ef_d(img_bgr=None):
            return _extract_features(img_path, img_bgr=img_bgr,
                                      feat_res=FEAT_RES,
                                      enhance=PADIM_ENHANCE,
                                      clahe_clip=PADIM_ENH_CLIP,
                                      clahe_tile=PADIM_ENH_TILE)

        def _mahal_d(ft_arr):
            HH,WW,CC = ft_arr.shape
            flat = ft_arr.reshape(-1,CC)
            diff = flat - padim_mean
            return np.sqrt(np.sum((diff @ padim_inv_cov)*diff,axis=1)).reshape(HH,WW)

        # Global + one coarse tile pass for speed
        ft_g  = _ef_d()
        araw  = cv2.resize(_mahal_d(ft_g).astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)

        # Quick 3×4 tile pass
        th_ = h//3; tw_ = w//4
        for tr in range(3):
            for tc in range(4):
                ty1=tr*th_; ty2=min(ty1+th_,h); tx1=tc*tw_; tx2=min(tx1+tw_,w)
                if ty2<=ty1 or tx2<=tx1: continue
                tile_up = cv2.resize(orig[ty1:ty2,tx1:tx2],(512,512),interpolation=cv2.INTER_CUBIC)
                at = _mahal_d(_ef_d(img_bgr=tile_up))
                at = cv2.resize(at.astype(np.float32),(tx2-tx1,ty2-ty1),interpolation=cv2.INTER_LINEAR)
                araw[ty1:ty2,tx1:tx2] = np.maximum(araw[ty1:ty2,tx1:tx2], at)

        araw_s = cv2.GaussianBlur(araw,(7,7),0)
        inside = araw_s[araw_s > 0]
        if inside.size == 0: inside = araw_s.ravel()
        im = float(np.mean(inside)); ist = float(np.std(inside))
        ip98 = float(np.percentile(inside,98))
        adap_th = im + K*ist

        # Raw blobs (no gates)
        bm = (araw_s > adap_th).astype(np.uint8)
        bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        nl,_,st,_ = cv2.connectedComponentsWithStats(bm, connectivity=8)

        # Sobel for contrast scores
        gray_d = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sx = cv2.Sobel(gray_d, cv2.CV_32F,1,0,ksize=3)
        sy = cv2.Sobel(gray_d, cv2.CV_32F,0,1,ksize=3)
        grad_map = np.sqrt(sx**2+sy**2)

        blobs = []
        for lab in range(1,nl):
            area = int(st[lab,cv2.CC_STAT_AREA])
            bx=int(st[lab,cv2.CC_STAT_LEFT]); by=int(st[lab,cv2.CC_STAT_TOP])
            bw=int(st[lab,cv2.CC_STAT_WIDTH]); bh=int(st[lab,cv2.CC_STAT_HEIGHT])
            bx2=min(bx+bw,w-1); by2=min(by+bh,h-1)
            asp = float(max(bw,bh))/(float(min(bw,bh))+1e-5)
            peak = float(araw_s[by:by2,bx:bx2].max()) if by2>by and bx2>bx else 0.0
            roi_std = float(grad_map[by:by2,bx:bx2].std()) if by2>by and bx2>bx else 0.0
            blobs.append({"lab":lab,"area":area,"x":bx,"y":by,"x2":bx2,"y2":by2,
                          "asp":round(asp,2),"peak":round(peak,2),"roi_std":round(roi_std,2),
                          "peak_ratio":round(peak/adap_th,3) if adap_th>0 else 0})

        # Sort by peak descending
        blobs.sort(key=lambda b: b["peak"], reverse=True)

        return jsonify({
            "inside_mean":   round(im,3),
            "inside_std":    round(ist,3),
            "inside_p98":    round(ip98,3),
            "fused_max":     round(float(araw.max()),3),
            "effective_th":  round(adap_th,3),
            "K":             K,
            "raw_blob_count":nl-1,
            "raw_blobs":     blobs[:30],   # top 30 by peak score
            "tip": (
                "peak_ratio > 1.0 means blob exceeds threshold. "
                "Look at area/asp/roi_std to see which gate filters it. "
                "Real defects: area 200-8000, asp < 5, roi_std > 3"
            )
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/camera/connect",  methods=["POST"])
def r_cam_connect():    return jsonify({"success":camera_connect()})

@app.route("/api/camera/disconnect", methods=["POST"])
def r_cam_disconnect(): camera_disconnect(); return jsonify({"success":True})

@app.route("/api/camera/capture",  methods=["POST"])
def r_cam_capture():
    p=camera_capture()
    return jsonify({"success":bool(p),"filename":Path(p).name if p else None,
                    "message":None if p else "Capture failed or camera not connected"})

@app.route("/api/camera/stream")
def r_cam_stream():
    """Return the most recent live frame as a single JPEG.
    Priority: live camera buffer → watch_folder newest → processed results → placeholder."""
    # 1. Live frame from camera (freshest, within last 5 s)
    with _live_frame_lock:
        if _live_frame_bytes and (time.time() - _live_frame_ts) < 5.0:
            data = bytes(_live_frame_bytes)
            resp = Response(data, mimetype="image/jpeg")
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"]        = "no-cache"
            return resp

    # 2. Newest file in watch_folder (just captured, not yet processed)
    wf_imgs = sorted(WATCH_FOLDER.glob("*.jpg"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if wf_imgs:
        resp = send_file(str(wf_imgs[0]), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # 3. Latest result image from either processed folder
    for pat, folder in [("result_*.jpg", PROCESSED_DEFECT),
                        ("result_*.jpg", PROCESSED_CLEAN)]:
        imgs = sorted(folder.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        if imgs:
            resp = send_file(str(imgs[0]), mimetype="image/jpeg")
            resp.headers["Cache-Control"] = "no-store"
            return resp

    # 4. Grey placeholder
    ph = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.putText(ph, "Waiting for camera", (130, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 90, 90), 2)
    _, buf = cv2.imencode(".jpg", ph)
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/camera/live")
def r_cam_live():
    """MJPEG multipart stream — use as <img src='/api/camera/live'> for true live video."""
    def _generate():
        boundary = b"--frame\r\n"
        while True:
            # Grab from live buffer
            with _live_frame_lock:
                frame_data = bytes(_live_frame_bytes) if _live_frame_bytes else None
                fresh = frame_data and (time.time() - _live_frame_ts) < 5.0

            if not fresh or not frame_data:
                # Build a "waiting" placeholder frame
                ph = np.full((480, 640, 3), 40, dtype=np.uint8)
                msg = "Camera Connected – Waiting…" if camera_connected else "Camera Offline"
                cv2.putText(ph, msg, (80 if camera_connected else 160, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 2)
                _, buf = cv2.imencode(".jpg", ph)
                frame_data = buf.tobytes()

            yield (boundary +
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   frame_data +
                   b"\r\n")
            time.sleep(0.08)  # ~12 fps

    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache",
                 "X-Accel-Buffering": "no"}
    )

@app.route("/api/settings", methods=["GET"])
def r_get_settings(): return jsonify(settings)

@app.route("/api/settings", methods=["POST"])
def r_save_settings():
    global settings; settings.update(request.json); save_settings(settings)
    return jsonify({"success":True})

@app.route("/api/images/<folder>/<filename>")
def r_image(folder,filename):
    d={"processed_defect":PROCESSED_DEFECT,"processed_clean":PROCESSED_CLEAN}.get(folder)
    if d and (d/filename).exists(): return send_from_directory(str(d),filename)
    return ("Not found",404)

# ── SocketIO events ───────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("camera_status",{"connected":camera_connected})
    emit("system_status",{"padim_trained":padim_trained,"yolo_loaded":yolo_loaded})

@socketio.on("camera_connect")
def on_cam_conn():    emit("camera_status",{"connected":camera_connect()})

@socketio.on("camera_disconnect")
def on_cam_disc():    camera_disconnect()

@socketio.on("capture")
def on_capture():
    p=camera_capture()
    emit("capture_ack",{"success":bool(p),"filename":Path(p).name if p else None})

# ── Startup ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*52)
    print("  Royal Enfield Defect Detection Server")
    print(f"  Device : {DEVICE}   Base: {BASE_DIR}")
    print("="*52)
    init_yolo()
    init_padim()
    threading.Thread(target=_watch_loop,       daemon=True).start()
    threading.Thread(target=_live_stream_loop, daemon=True).start()
    print("\n  → Open http://localhost:8080\n")
    socketio.run(app, host="0.0.0.0", port=8080, debug=False, allow_unsafe_werkzeug=True)