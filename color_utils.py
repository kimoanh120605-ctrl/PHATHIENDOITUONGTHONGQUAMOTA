"""Kiểm tra màu thật (HSV) trên ảnh cắt của đối tượng để loại kết quả sai màu."""
import cv2
import numpy as np

COLORS = ["black", "white", "gray", "red", "orange", "yellow", "green",
          "blue", "purple", "pink", "brown"]
# vùng lấy mẫu (x0, x1, y0, y1) theo tỉ lệ ảnh cắt
REGIONS = {"body": (0.10, 0.90, 0.15, 0.90),
           "upper": (0.20, 0.80, 0.15, 0.55),
           "lower": (0.20, 0.80, 0.55, 0.90)}
ITEM_REGION = {"car": "body", "truck": "body", "bus": "body", "motorcycle": "body",
               "bicycle": "body", "dress": "body", "shirt": "upper", "jacket": "upper",
               "t-shirt": "upper", "pants": "lower"}


def _fractions(bgr):
    h, w = bgr.shape[:2]
    sc = 64 / max(h, w, 1)
    if sc < 1:
        bgr = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = [hsv[..., i].astype(int) for i in range(3)]
    low_s = S < 45
    chrom = (V >= 50) & (S >= 45)
    reddish = chrom & ((H < 22) | (H >= 170))
    masks = {
        "black": (V < 70) | ((S >= 45) & (V < 50)),
        "dgray": low_s & (V >= 70) & (V < 110),       # xám đậm (xe đen bị phản chiếu)
        "lgray": low_s & (V >= 110) & (V < 165),      # xám sáng (xe trắng trong bóng)
        "white": low_s & (V >= 165),
        "red": reddish & ((H < 8) | (H >= 170)) & (V >= 130),
        "orange": reddish & (H >= 8) & (H < 22) & (V >= 130),
        "brown": reddish & (V < 130),
        "yellow": chrom & (H >= 22) & (H < 35),
        "green": chrom & (H >= 35) & (H < 85),
        "blue": chrom & (H >= 85) & (H < 130),
        "purple": chrom & (H >= 130) & (H < 150),
        "pink": chrom & (H >= 150) & (H < 170),
    }
    n = float(H.size)
    return {k: float(m.sum()) / n for k, m in masks.items()}


def soft_scores(fr):
    """Điểm mềm cho từng màu: xám đậm tính một phần là đen, xám sáng một phần là trắng."""
    sc = {k: fr[k] for k in ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown")}
    sc["black"] = fr["black"] + 0.7 * fr["dgray"]
    sc["white"] = fr["white"] + 0.7 * fr["lgray"]
    sc["gray"] = fr["dgray"] + fr["lgray"]
    return sc


def analyze(path):
    """Trả về {vùng: {màu: tỉ lệ}} hoặc None nếu không đọc được ảnh."""
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    out = {}
    for name, (x0, x1, y0, y1) in REGIONS.items():
        crop = img[int(h * y0):max(int(h * y1), int(h * y0) + 1),
                   int(w * x0):max(int(w * x1), int(w * x0) + 1)]
        out[name] = _fractions(crop)
    return out


def color_match(fracs, attrs):
    """attrs: [(màu, vật)]. Đạt khi màu được tìm là màu nổi bật nhất của vật (không cần ngưỡng).
    Trả về (đạt?, điểm màu 0..1, có áp dụng không)."""
    scores, applied = [], False
    for color, item in attrs:
        region = ITEM_REGION.get(item)
        if region is None or fracs is None:
            continue
        applied = True
        col = "gray" if color in ("grey", "silver") else color
        sc = soft_scores(fracs[region])
        q = sc.get(col, 0.0)
        if q < 0.12 or q < 0.85 * max(sc.values()):
            return False, q, True
        scores.append(q)
    if not applied:
        return True, 0.0, False
    return True, sum(scores) / len(scores), True