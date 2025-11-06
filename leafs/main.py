# main.py
"""
Driver that:
  - calls leaf_recognition.segment_leaves(image_path)
  - computes per-leaf color fractions and a simple health label
  - draws an overlay WITH health lines
  - writes outputs using OUT_OVERLAY/OUT_MASKS from leaf_recognition
"""

import sys
import cv2
import numpy as np
from leaf_recognition import segment_leaves, OUT_OVERLAY, OUT_MASKS, MODEL_FILE

IMAGE_FILE = "leaf_brownish.jpg"

# ---- Health utilities (remain in main.py) ----
def color_fracs(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    inside = (mask == 255)
    if np.count_nonzero(inside) == 0:
        return dict(green=0.0, yellow=0.0, brown=0.0)
    H, S, V = cv2.split(hsv)
    h, s, v = H[inside], S[inside], V[inside]
    total = float(h.size)
    green  = np.count_nonzero((h >= 35) & (h <= 85) & (s >= 30) & (v >= 30)) / total
    yellow = np.count_nonzero((h >= 20) & (h < 35)  & (s >= 20) & (v >= 40)) / total
    brown  = np.count_nonzero((h >= 5)  & (h < 25)  & (v <= 200))            / total
    return dict(green=green, yellow=yellow, brown=brown)

def health_from_colors(fr: dict) -> str:
    g = fr["green"]; yb = fr["yellow"] + fr["brown"]
    if g > yb + 0.05:  return "healthy-ish"
    if yb > g + 0.05:  return "stressed-ish"
    return "borderline"

def draw_overlay_with_lines(img: np.ndarray, masks: list[np.ndarray], lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Overlay with contours + 'Leaf i' + legend lines; returns (overlay_bgr, idx_uint16)."""
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

    # Legend
    x0, y0 = 10, 22
    box_w = 780  # wider to avoid clipping
    box_h = 24 * len(lines) + 28
    cv2.rectangle(out, (x0 - 8, y0 - 20), (x0 - 8 + box_w, y0 - 20 + box_h), (0, 0, 0), -1)
    y = y0
    for s in lines:
        cv2.putText(out, s, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 24
    return out, idx

def main():
    # Get masks (no overlay from module; we add health first)
    try:
        leaf_masks = segment_leaves(IMAGE_FILE, model_path=MODEL_FILE, save_overlay=False)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # Load the (possibly resized inside module) image again for display and color stats
    img = cv2.imread(IMAGE_FILE)
    if img is None:
        print(f"Cannot read {IMAGE_FILE}", file=sys.stderr)
        sys.exit(1)

    if not leaf_masks:
        print("No leaf-like regions detected.")
        cv2.imwrite(OUT_OVERLAY, img)
        cv2.imwrite(OUT_MASKS, np.zeros(img.shape[:2], np.uint16))  # empty 16-bit index map
        print(f"Saved: {OUT_OVERLAY}, {OUT_MASKS}")
        return

    # Per-leaf health lines
    lines = []
    for i, m in enumerate(leaf_masks, 1):
        fr = color_fracs(img, m)
        hv = health_from_colors(fr)
        lines.append(
            f"Leaf {i}: G {fr['green']*100:.1f}% | Y {fr['yellow']*100:.1f}% | "
            f"B {fr['brown']*100:.1f}% | Health: {hv}"
        )

    # Draw final overlay (with lines) and save using module constants
    overlay, idx = draw_overlay_with_lines(img, leaf_masks, lines)
    cv2.imwrite(OUT_OVERLAY, overlay)
    cv2.imwrite(OUT_MASKS, idx)  # uint16 PNG keeps indices intact

    print("Per-leaf summary:")
    for s in lines:
        print(" - " + s)
    print(f"Saved: {OUT_OVERLAY}, {OUT_MASKS}")

if __name__ == "__main__":
    main()
