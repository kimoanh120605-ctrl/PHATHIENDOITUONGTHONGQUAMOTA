"""Giao diện Streamlit: chọn video -> lập chỉ mục -> nhập mô tả -> xem clip ngay.

Chạy:  python -m streamlit run app.py
"""
import json, os, shutil, subprocess

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tìm kiếm đối tượng trong video", layout="wide")
st.title("🔎 Tìm kiếm & theo dõi đối tượng trong video")
# Thư viện nặng (torch, faiss, ...) được import muộn để giao diện hiện ngay.


@st.cache_resource(show_spinner="Đang nạp CLIP...")
def get_clip():
    from common import load_clip
    return load_clip()


@st.cache_resource(show_spinner="Đang nạp chỉ mục...")
def get_index(data_dir):
    import faiss
    meta = json.load(open(os.path.join(data_dir, "tracks.json"), encoding="utf-8"))
    return meta, faiss.read_index(os.path.join(data_dir, "index.faiss"))


@st.cache_data(show_spinner=False)
def parse_cached(q):
    from search import parse_query
    return parse_query(q)


@st.cache_data(show_spinner=False)
def colors_of(data_dir, track_id, _stamp):
    from color_utils import analyze
    return analyze(os.path.join(data_dir, "thumbs", f"{track_id}.jpg"))


def make_clip(data_dir, video, track, fps, pad):
    """Xuất clip có vẽ khung; chuyển sang H.264 (nếu có ffmpeg) để trình duyệt phát được."""
    from search import export_clip
    out_dir = os.path.join(data_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"track{track['track_id']}_pad{pad}.mp4")
    if os.path.exists(final):
        return final
    raw = final.replace(".mp4", "_raw.mp4")
    export_clip(video, track, fps, raw, pad_sec=pad, label=f"track {track['track_id']}")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", final], check=True)
        os.remove(raw)
    else:
        os.replace(raw, final)
    return final


def fmt(sec):
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def select(r):
    st.session_state.selected = r


def show_video(path):
    try:
        st.video(path, autoplay=True, muted=True, loop=True)
    except TypeError:       # Streamlit bản cũ không có autoplay
        st.video(path)


ROOT = "data"


def list_indexed():
    found = []
    if os.path.exists(os.path.join(ROOT, "index.faiss")):
        found.append(ROOT)
    if os.path.isdir(ROOT):
        for d in sorted(os.listdir(ROOT)):
            p = os.path.join(ROOT, d)
            if os.path.exists(os.path.join(p, "index.faiss")):
                found.append(p)
    return found


# ---------------- Sidebar ----------------
st.sidebar.header("Cấu hình")
st.sidebar.subheader("1. Chọn video")
up = st.sidebar.file_uploader("Bấm để chọn video từ máy tính",
                              type=["mp4", "avi", "mov", "mkv", "webm"])
if up is not None:
    stride = st.sidebar.slider("Bỏ qua khung (lớn hơn = index nhanh hơn)", 1, 10, 3)
    if st.sidebar.button("⚙️ Lập chỉ mục video này", type="primary"):
        out = os.path.join(ROOT, os.path.splitext(up.name)[0])
        os.makedirs(out, exist_ok=True)
        vpath = os.path.join(out, up.name)
        with open(vpath, "wb") as f:
            f.write(up.getbuffer())
        bar = st.progress(0.0, text="Đang bắt đầu...")
        try:
            from index_video import run_index
            n = run_index(vpath, out, stride=stride,
                          progress=lambda p, m: bar.progress(min(max(p, 0.0), 1.0), text=m))
            get_index.clear()
            colors_of.clear()
            st.session_state["data_sel"] = out
            st.sidebar.success(f"Đã lập chỉ mục {n} đối tượng.")
        except Exception as e:
            st.sidebar.error(f"Lỗi khi lập chỉ mục: {e}")
        bar.empty()

st.sidebar.subheader("2. Video đã lập chỉ mục")
_options = list_indexed()
data_dir = (st.sidebar.selectbox("Chọn video", _options, key="data_sel",
                                 format_func=lambda p: os.path.basename(p) if p != ROOT else ROOT)
            if _options else ROOT)

st.sidebar.subheader("3. Tùy chọn tìm kiếm")
topk = st.sidebar.slider("Số kết quả (Top-K)", 1, 20, 6)
pad = st.sidebar.slider("Đệm trước/sau clip (giây)", 0.0, 5.0, 1.0, 0.5)
min_score = st.sidebar.slider("Điểm tối thiểu", 0.0, 0.5, 0.0, 0.01,
                              help="Điểm cosine CLIP thường nằm trong khoảng 0.15–0.35")
cls_choice = st.sidebar.selectbox("Lớp đối tượng", ["Tự động", "Tất cả", "person", "car",
                                  "motorcycle", "bus", "truck", "bicycle"])
use_color = st.sidebar.checkbox("Lọc theo màu thật (HSV)", True,
                                help="Loại kết quả sai màu, ví dụ tìm xe trắng thì bỏ xe đen")

if not os.path.exists(os.path.join(data_dir, "index.faiss")):
    st.info("👈 Chưa có video nào được lập chỉ mục. Hãy chọn video ở thanh bên trái rồi bấm "
            "**Lập chỉ mục video này**.")
    st.stop()

meta, index = get_index(data_dir)
tracks, fps, video = meta["tracks"], meta["fps"], meta["video"]
st.caption(f"Video: `{os.path.basename(video)}` · {len(tracks)} tracklet · {fps:.1f} fps")

# ---------------- Truy vấn ----------------
query = st.text_input("Mô tả đối tượng cần tìm",
                      placeholder="xe ô tô trắng / người đàn ông áo đỏ đeo balo đen")
if not query:
    st.info("Nhập mô tả để bắt đầu. Video gốc bên dưới để bạn tham khảo.")
    if os.path.exists(video):
        st.video(video)
    st.stop()

from common import embed_text
from query_rules import extract_attrs
from color_utils import color_match

q = parse_cached(query)
cls = q["cls"] if cls_choice == "Tự động" else (None if cls_choice == "Tất cả" else cls_choice)
attrs = extract_attrs(q["text_en"])

model, proc, device = get_clip()
vec = embed_text(model, proc, [f"a photo of {q['text_en']}"], device).astype("float32")
scores, ids = index.search(vec, len(tracks))

stamp = os.path.getmtime(os.path.join(data_dir, "tracks.json"))
cands, color_on = [], False
for s, i in zip(scores[0], ids[0]):
    t = tracks[i]
    if (cls and t["cls"] != cls) or s < min_score:
        continue
    ok, frac, applied = True, 0.0, False
    if use_color and attrs:
        ok, frac, applied = color_match(colors_of(data_dir, t["track_id"], stamp), attrs)
        color_on = color_on or applied
    cands.append((float(s) + 0.15 * frac, float(s), t, ok, frac))

good = [c for c in cands if c[3]]
relaxed = color_on and not good
if relaxed:       # không có đối tượng nào đúng màu -> hiển thị gần đúng nhất và báo rõ
    good = sorted(cands, key=lambda c: -c[4])
rejected = len(cands) - len(good) if not relaxed else 0
results = [(f, s, t) for f, s, t, _, _ in sorted(good, key=lambda c: -c[0] if not relaxed else -c[4])][:topk]

with st.expander("Truy vấn được hiểu như thế nào?"):
    st.json({**q, "lọc_lớp": cls, "màu_vật": attrs, "đã_loại_sai_màu": rejected})

if not results:
    st.warning("Không có kết quả phù hợp. Thử tắt lọc màu hoặc đổi lớp đối tượng.")
    st.stop()
if relaxed:
    st.warning("Không có đối tượng nào đúng màu bạn tìm trong video. Đang hiển thị các kết quả gần đúng nhất.")

key = (query, data_dir, cls_choice, use_color)
if st.session_state.get("last_key") != key:      # truy vấn mới -> tự chọn kết quả tốt nhất
    st.session_state.last_key = key
    st.session_state.selected = 0
sel = min(st.session_state.get("selected", 0), len(results) - 1)

# ---------------- Clip phát ngay dưới ô nhập ----------------
_, s_clip, t = results[sel]
st.subheader(f"▶️ Kết quả #{sel + 1}: track {t['track_id']} ({t['cls']}) · "
             f"{fmt(t['start_sec'])} → {fmt(t['end_sec'])}")
left, right = st.columns([3, 2])
with left:
    with st.spinner("Đang xuất clip..."):
        path = make_clip(data_dir, video, t, fps, pad)
    show_video(path)
    with open(path, "rb") as f:
        st.download_button("⬇️ Tải clip", f, file_name=os.path.basename(path), mime="video/mp4")
with right:
    st.markdown("**Quỹ đạo (tâm khung theo thời gian)**")
    traj = pd.DataFrame(t["boxes"], columns=["frame", "x1", "y1", "x2", "y2"])
    traj["Giây"] = traj["frame"] / fps
    traj["x"] = (traj["x1"] + traj["x2"]) / 2
    traj["y"] = (traj["y1"] + traj["y2"]) / 2
    st.altair_chart(
        alt.Chart(traj).mark_line(point=True).encode(
            x=alt.X("x:Q", title="x (px)"),
            y=alt.Y("y:Q", title="y (px)", scale=alt.Scale(reverse=True)),
            order="frame:Q", tooltip=["Giây", "x", "y"]).properties(height=280),
        use_container_width=True)
    st.caption(f"{len(traj)} lần quan sát · điểm CLIP {s_clip:.3f}")

# ---------------- Danh sách kết quả ----------------
st.subheader("Các kết quả tìm được")
for row in range(0, len(results), 4):
    cols = st.columns(4)
    for c, r in zip(cols, range(row, min(row + 4, len(results)))):
        _, s0, t0 = results[r]
        with c:
            thumb = os.path.join(data_dir, "thumbs", f"{t0['track_id']}.jpg")
            if os.path.exists(thumb):
                st.image(thumb, use_container_width=True)
            st.markdown(f"**#{r + 1}** · track {t0['track_id']} · `{t0['cls']}`  \n"
                        f"điểm **{s0:.3f}** · ⏱ {fmt(t0['start_sec'])} → {fmt(t0['end_sec'])}")
            st.button("▶️ Đang xem" if r == sel else "Xem clip", key=f"btn{r}",
                      on_click=select, args=(r,), use_container_width=True,
                      type="primary" if r == sel else "secondary")

# ---------------- Timeline ----------------
st.subheader("Timeline")
df = pd.DataFrame([{
    "Kết quả": f"#{r} · track {t1['track_id']}", "Bắt đầu (s)": t1["start_sec"],
    "Kết thúc (s)": t1["end_sec"], "Điểm": round(s1, 3), "Lớp": t1["cls"],
} for r, (_, s1, t1) in enumerate(results, 1)])
st.altair_chart(
    alt.Chart(df).mark_bar(height=18).encode(
        x=alt.X("Bắt đầu (s):Q", title="Thời gian trong video (giây)"), x2="Kết thúc (s):Q",
        y=alt.Y("Kết quả:N", sort=None, title=None),
        color=alt.Color("Điểm:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=list(df.columns)).properties(height=max(120, 32 * len(df))),
    use_container_width=True)