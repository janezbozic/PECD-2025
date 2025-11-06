#!/usr/bin/env python3
# leaf_recognition.py
"""
Leaf segmentation using a pre-trained U²-Net (u2netp.onnx).

Exports:
    segment_leaves(image_path: str, save_overlay: bool = False) -> list[np.ndarray]
        Returns per-leaf binary masks (uint8 {0,255}), sorted by area desc.

Defines:
    MODEL_FILE   = "u2netp.onnx"
    OUT_OVERLAY  = "leaf_overlay.jpg"
    OUT_MASKS    = "leaf_masks.png"
    MAX_SIDE     = 1200
    MIN_LEAF_PX  = 1500
"""

import os
import cv2
import numpy as np
import onnxruntime as ort

# ---- Constants (owned by this module) ----
MODEL_FILE = "u2netp.onnx"
OUT_OVERLAY = "leaf_overlay.jpg"
OUT_MASKS   = "leaf_masks.png"
MAX_SIDE    = 1200
MIN_LEAF_PX = 1500

# ---- Internal helpers ----
def _read_and_resize(path: str, max_side: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img

def _u2net_saliency_mask(img_bgr: np.ndarray, sess: ort.InferenceSession) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,320,320)

    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: x})[0]          # expect (1,1,320,320)
    sal = out[0, 0]
    sal = (255 * (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)).astype(np.uint8)
    sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(sal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def _split_components(bin_mask: np.ndarray, min_leaf_px: int) -> list[np.ndarray]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    masks: list[np.ndarray] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_leaf_px:
            m = np.zeros_like(bin_mask, np.uint8)
            m[labels == idx] = 255
            masks.append(m)
    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    return masks

def _draw_overlay(img: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Contour-only overlay; labels 'Leaf i'; returns (overlay_bgr, idx_uint16)."""
    out = img.copy()
    idx = np.zeros(img.shape[:2], np.uint16)
    for i, m in enumerate(masks, 1):
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(cnts[0])
            cv2.putText(out, f"Leaf {i}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        idx[m == 255] = i
    return out, idx

# ---- Public API ----
def segment_leaves(
    image_path: str,
    *,
    model_path: str = MODEL_FILE,
    max_side: int = MAX_SIDE,
    min_leaf_px: int = MIN_LEAF_PX,
    kernel_size: int = 5,
    close_iters: int = 2,
    open_iters: int = 1,
    save_overlay: bool = False,   # keep False so main.py can draw health lines
) -> list[np.ndarray]:
    """
    Saliency-based leaf segmentation -> list of binary masks (uint8 {0,255}).
    If save_overlay=True, writes contour-only overlay and 16-bit index map.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    img = _read_and_resize(image_path, max_side)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    raw = _u2net_saliency_mask(img, sess)

    k = np.ones((kernel_size, kernel_size), np.uint8)
    clean = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k, iterations=close_iters)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  k, iterations=open_iters)

    leaf_masks = _split_components(clean, min_leaf_px)

    if save_overlay:
        overlay, idx = _draw_overlay(img, leaf_masks)
        cv2.imwrite(OUT_OVERLAY, overlay)
        cv2.imwrite(OUT_MASKS, idx)  # uint16 PNG
        print(f"[leaf_recognition] Saved {OUT_OVERLAY} and {OUT_MASKS}")

    return leaf_masks
