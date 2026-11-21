"""Phát hiện + theo dõi + trích embedding + lập chỉ mục FAISS.

Dùng dòng lệnh:  python index_video.py --video input.mp4 --out data
Hoặc import:     from index_video import run_index
"""
import argparse, json, os
from collections import defaultdict

import cv2
import faiss
import numpy as np
from PIL import Image
from ultralytics import YOLO

from common import load_clip, embed_images

# COCO: person, bicycle, car, motorcycle, bus, truck
CLASSES = [0, 1, 2, 3, 5, 7]


def run_index(video, out, weights="yolo11n.pt", stride=2, conf=0.3, min_len=5,
              max_crops=12, min_area=24 * 48, progress=None):
    """Trả về số tracklet đã lập chỉ mục. progress(p: 0..1, msg: str) là callback tùy chọn."""
    report = progress or (lambda p, m: print(m))
    os.makedirs(os.path.join(out, "thumbs"), exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()

    detector = YOLO(weights)
    names = detector.names

    boxes = defaultdict(list)                 # id -> [[frame,x1,y1,x2,y2], ...]
    cls_votes = defaultdict(lambda: defaultdict(int))
    crops = defaultdict(list)                 # id -> [(area, PIL)]
    next_crop = defaultdict(int)
    interval = defaultdict(lambda: stride * 3)

    stream = detector.track(
        source=video, stream=True, persist=True, tracker="bytetrack.yaml",
        classes=CLASSES, conf=conf, vid_stride=stride, verbose=False,
    )
    for i, r in enumerate(stream):
        frame_idx = i * stride
        if i % 20 == 0:
            report(0.9 * min(1.0, frame_idx / total),
                   f"Theo dõi đối tượng: khung {frame_idx}/{total} ({len(boxes)} track)")
        if r.boxes is None or r.boxes.id is None:
            continue
        frame = r.orig_img
        h, w = frame.shape[:2]
        for xyxy, tid, c in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.id.int().cpu().tolist(),
                                r.boxes.cls.int().cpu().tolist()):
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            boxes[tid].append([frame_idx, x1, y1, x2, y2])
            cls_votes[tid][c] += 1
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area and frame_idx >= next_crop[tid]:
                crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                crops[tid].append((area, Image.fromarray(crop)))
                next_crop[tid] = frame_idx + interval[tid]
                if len(crops[tid]) > max_crops:      # giữ phân bố đều theo thời gian
                    crops[tid] = crops[tid][::2]
                    interval[tid] *= 2

    report(0.92, "Đang nạp CLIP và trích đặc trưng...")
    model, proc, device = load_clip()
    tracks, vectors = [], []
    for tid, bx in boxes.items():
        if len(bx) < min_len or not crops[tid]:
            continue
        imgs = [im for _, im in crops[tid]]
        emb = embed_images(model, proc, imgs, device)
        v = emb.mean(axis=0)
        v /= np.linalg.norm(v) + 1e-9                  # embedding đại diện tracklet
        best = max(crops[tid], key=lambda t: t[0])[1]
        best.save(os.path.join(out, "thumbs", f"{tid}.jpg"))
        cls_id = max(cls_votes[tid], key=cls_votes[tid].get)
        tracks.append({
            "track_id": tid, "cls": names[cls_id],
            "start_frame": bx[0][0], "end_frame": bx[-1][0],
            "start_sec": bx[0][0] / fps, "end_sec": bx[-1][0] / fps,
            "boxes": bx,
        })
        vectors.append(v)

    if not vectors:
        raise RuntimeError("Không phát hiện được đối tượng nào đủ lớn/đủ lâu trong video.")

    index = faiss.IndexFlatIP(len(vectors[0]))         # cosine vì đã chuẩn hóa
    index.add(np.stack(vectors).astype("float32"))
    faiss.write_index(index, os.path.join(out, "index.faiss"))
    with open(os.path.join(out, "tracks.json"), "w", encoding="utf-8") as f:
        json.dump({"video": os.path.abspath(video), "fps": fps, "tracks": tracks}, f)
    report(1.0, f"Xong: {len(tracks)} tracklet")
    return len(tracks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-crops", type=int, default=12)
    ap.add_argument("--min-area", type=int, default=24 * 48)
    a = ap.parse_args()
    n = run_index(a.video, a.out, a.weights, a.stride, a.conf, a.min_len, a.max_crops, a.min_area)
    print(f"Xong: {n} tracklet đã được lập chỉ mục -> {a.out}/")


if __name__ == "__main__":
    main()