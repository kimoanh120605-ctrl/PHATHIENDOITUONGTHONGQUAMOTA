"""Bước 2: Tìm kiếm bằng mô tả ngôn ngữ tự nhiên, xuất clip có vẽ khung.

Chạy:  python search.py --data data --query "người đàn ông áo đỏ đeo balo đen" --topk 5
Đặt ANTHROPIC_API_KEY để tự dịch/phân tích câu tiếng Việt (CLIP hiểu tiếng Anh tốt nhất).
"""
import argparse, bisect, json, os

import cv2
import faiss

from common import load_clip, embed_text
from query_rules import rule_parse
from clips import export_clip

VALID_CLS = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def parse_query(q: str) -> dict:
    """Dùng LLM tách truy vấn -> {text_en, cls}. Không có API key thì dùng nguyên văn."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return rule_parse(q)  # không có API key: dịch bằng luật
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-5-5", max_tokens=300,
        system=("Dịch mô tả đối tượng sang tiếng Anh ngắn gọn cho CLIP và xác định lớp đối tượng. "
                'Chỉ trả JSON: {"text_en": "...", "cls": "person|bicycle|car|motorcycle|bus|truck|null"}'),
        messages=[{"role": "user", "content": q}],
    )
    txt = msg.content[0].text.strip().strip("`").removeprefix("json").strip()
    try:
        d = json.loads(txt)
        d["cls"] = d.get("cls") if d.get("cls") in VALID_CLS else None
        return d
    except Exception:
        return rule_parse(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--no-clip", action="store_true", help="không xuất clip")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.data, "tracks.json"), encoding="utf-8"))
    tracks, fps, video = meta["tracks"], meta["fps"], meta["video"]
    index = faiss.read_index(os.path.join(a.data, "index.faiss"))

    q = parse_query(a.query)
    print("Truy vấn hiểu là:", q)
    model, proc, device = load_clip()
    vec = embed_text(model, proc, [f"a photo of {q['text_en']}"], device).astype("float32")

    scores, ids = index.search(vec, len(tracks))           # xếp hạng toàn bộ rồi lọc theo lớp
    results = []
    for s, i in zip(scores[0], ids[0]):
        t = tracks[i]
        if q["cls"] and t["cls"] != q["cls"]:
            continue
        results.append((float(s), t))
        if len(results) == a.topk:
            break

    out_dir = os.path.join(a.data, "results"); os.makedirs(out_dir, exist_ok=True)
    for rank, (s, t) in enumerate(results, 1):
        print(f"#{rank}  track {t['track_id']} ({t['cls']})  điểm={s:.3f}  "
              f"{t['start_sec']:.1f}s -> {t['end_sec']:.1f}s")
        if not a.no_clip:
            p = os.path.join(out_dir, f"rank{rank}_track{t['track_id']}.mp4")
            export_clip(video, t, fps, p, label=f"#{rank} {s:.2f}",
                        proxy=os.path.join(a.data, "proxy.mp4"))
            print("    clip:", p)


if __name__ == "__main__":
    main()