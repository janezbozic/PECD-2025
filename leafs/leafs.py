import os, sys, cv2, numpy as np
import onnxruntime as ort

# ---- Constants ----
IMAGE_FILE = "leaf_brownish.jpg"
MODEL_FILE = "u2netp.onnx"
OUT_OVERLAY = "leaf_overlay.jpg"
OUT_MASKS = "leaf_masks.png"
MAX_SIDE = 1200
MIN_LEAF_PX = 1500

# ---- Functions ----
def read_and_resize(path, max_side=MAX_SIDE):
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: cannot read {path}", file=sys.stderr); sys.exit(1)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

def u2net_mask(img_bgr, model_path):
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))[None, ...]  # (1,3,320,320)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    sal = sess.run(None, {inp: x})[0][0,0]
    sal = (255*(sal - sal.min())/(sal.max()-sal.min()+1e-8)).astype(np.uint8)
    sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(sal, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return mask

def split_leaf_masks(bin_mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    masks = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= MIN_LEAF_PX:
            m = np.zeros_like(bin_mask, np.uint8)
            m[labels == idx] = 255
            masks.append(m)
    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    return masks

def color_fracs(img_bgr, mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    inside = (mask == 255)
    if np.count_nonzero(inside) == 0:
        return dict(green=0.0, yellow=0.0, brown=0.0)
    H,S,V = cv2.split(hsv)
    h, s, v = H[inside], S[inside], V[inside]
    total = float(h.size)
    green  = np.count_nonzero((h>=35)&(h<=85)&(s>=30)&(v>=30)) / total
    yellow = np.count_nonzero((h>=20)&(h<35)&(s>=20)&(v>=40)) / total
    brown  = np.count_nonzero((h>=5)&(h<25)&(v<=200)) / total
    return dict(green=green, yellow=yellow, brown=brown)

def health_from_colors(fr):
    g = fr["green"]; yb = fr["yellow"] + fr["brown"]
    if g > yb + 0.05:  return "healthy-ish"
    if yb > g + 0.05:  return "stressed-ish"
    return "borderline"

def draw_overlay(img, masks, lines):
    out = img.copy()
    idx = np.zeros(img.shape[:2], np.uint16)
    for i, m in enumerate(masks, 1):
        cnts, _ = cv2.findContours((m>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(out, cnts, -1, (0,0,255), 2)
            x,y,w,h = cv2.boundingRect(cnts[0])
            cv2.putText(out, f"Leaf {i}", (x, max(20,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        idx[m==255] = i
    # legend box
    x0,y0 = 10,22
    cv2.rectangle(out, (x0-8,y0-20), (x0+620, y0-20+22*len(lines)+24), (0,0,0), -1)
    y = y0
    for s in lines:
        cv2.putText(out, s, (x0,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y+=22
    return out, idx

def main():
    if not os.path.isfile(IMAGE_FILE) or not os.path.isfile(MODEL_FILE):
        print("ERROR: put leaf.jpg and u2netp.onnx next to this script.", file=sys.stderr)
        sys.exit(1)

    img = read_and_resize(IMAGE_FILE)
    raw_mask = u2net_mask(img, MODEL_FILE)

    k = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k, iterations=2)
    clean = cv2.morphologyEx(clean,  cv2.MORPH_OPEN,  k, iterations=1)
    leaf_masks = split_leaf_masks(clean)

    if not leaf_masks:
        print("No leaf-like regions detected.")
        cv2.imwrite(OUT_OVERLAY, img)
        cv2.imwrite(OUT_MASKS, np.zeros(img.shape[:2], np.uint8))
        return

    lines = []
    for i, m in enumerate(leaf_masks, 1):
        fr = color_fracs(img, m)
        hv = health_from_colors(fr)
        lines.append(f"Leaf {i}: G {fr['green']*100:.1f}% | Y {fr['yellow']*100:.1f}% | B {fr['brown']*100:.1f}% | Health: {hv}")

    overlay, idx = draw_overlay(img, leaf_masks, lines)
    cv2.imwrite(OUT_OVERLAY, overlay)
    cv2.imwrite(OUT_MASKS, idx.astype(np.uint8))

    print("Per-leaf summary:")
    for s in lines: print(" - " + s)
    print(f"Saved: {OUT_OVERLAY}, {OUT_MASKS}")

if __name__ == "__main__":
    main()
