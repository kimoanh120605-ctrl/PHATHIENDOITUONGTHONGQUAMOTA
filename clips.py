"""Xuất clip nhanh: dùng bản proxy 720p, mã hóa H.264, ghi file an toàn.
Có đường dự phòng bằng OpenCV khi ffmpeg thiếu/lỗi; lỗi ffmpeg được ghi vào ffmpeg_error.txt."""
import bisect
import functools
import os
import shutil
import subprocess
import tempfile

import cv2


def _dims(path):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


@functools.lru_cache(maxsize=1)
def _h264_args():
    """Chọn bộ mã hóa H.264 mà ffmpeg trên máy này có. None nếu không dùng được ffmpeg."""
    if not shutil.which("ffmpeg"):
        return None
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return None
    if "libx264" in out:
        return ["-vcodec", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p"]
    if "libopenh264" in out:
        return ["-vcodec", "libopenh264", "-b:v", "3M", "-pix_fmt", "yuv420p"]
    if "h264_mf" in out:
        return ["-vcodec", "h264_mf", "-b:v", "3M", "-pix_fmt", "yuv420p"]
    return None


def _log(folder, text):
    try:
        with open(os.path.join(folder, "ffmpeg_error.txt"), "a", encoding="utf-8") as f:
            f.write(text + "\n---\n")
    except OSError:
        pass


def ensure_proxy(video, proxy_path, max_side=720):
    """Tạo (một lần) bản video nhỏ để xem nhanh. Trả về đường dẫn video nên dùng."""
    if os.path.exists(proxy_path):
        return proxy_path
    w, h = _dims(video)
    if max(w, h) <= max_side or w == 0:
        return video
    s = max_side / max(w, h)
    nw, nh = max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)
    tmp = proxy_path[:-4] + ".part.mp4"
    folder = os.path.dirname(proxy_path) or "."
    try:
        enc = _h264_args()
        ok = False
        if enc:
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-an",
                                "-vf", f"scale={nw}:{nh}", *enc, "-movflags", "+faststart", tmp],
                               capture_output=True, text=True)
            ok = r.returncode == 0 and os.path.exists(tmp)
            if not ok:
                _log(folder, "ensure_proxy: " + (r.stderr or "")[-1500:])
        if not ok:                                   # dự phòng bằng OpenCV
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (nw, nh))
            while True:
                good, img = cap.read()
                if not good:
                    break
                wr.write(cv2.resize(img, (nw, nh)))
            cap.release(); wr.release()
        os.replace(tmp, proxy_path)
        return proxy_path
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def export_clip(video, track, fps, out_path, pad_sec=1.0, label="", proxy=None, max_len_sec=30):
    ow, oh = _dims(video)
    src = video
    if proxy and os.path.exists(proxy):
        pw, ph = _dims(proxy)
        if pw and (pw >= ph) == (ow >= oh):          # cùng chiều ngang/dọc mới dùng proxy
            src = proxy
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2
    sx, sy = w / ow, h / oh

    start = max(0, int(track["start_frame"] - pad_sec * fps))
    end = min(int(track["end_frame"] + pad_sec * fps), start + int(max_len_sec * fps))
    frames = [b[0] for b in track["boxes"]]
    tmp = out_path[:-4] + ".part.mp4"
    folder = os.path.dirname(out_path) or "."

    def render(write):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for f in range(start, end + 1):
            ok, img = cap.read()
            if not ok:
                break
            img = img[:h, :w]
            if track["start_frame"] <= f <= track["end_frame"]:
                k = max(0, bisect.bisect_right(frames, f) - 1)
                _, x1, y1, x2, y2 = track["boxes"][k]
                p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
                cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
                cv2.putText(img, label, (p1[0], max(15, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            write(img)

    try:
        done = False
        enc = _h264_args()
        if enc:                                      # ghi thẳng H.264 một lượt
            errf = tempfile.TemporaryFile()
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", *enc, "-movflags", "+faststart", tmp],
                stdin=subprocess.PIPE, stderr=errf)
            try:
                render(lambda img: proc.stdin.write(img.tobytes()))
                proc.stdin.close()
                proc.wait()
                done = proc.returncode == 0 and os.path.exists(tmp)
            except (BrokenPipeError, OSError):
                done = False
            finally:
                if proc.poll() is None:
                    proc.kill()
            if not done:
                errf.seek(0)
                _log(folder, "export_clip: " + errf.read().decode(errors="ignore")[-1500:])
            errf.close()
        if not done:                                 # dự phòng bằng OpenCV (mp4v)
            if os.path.exists(tmp):
                os.remove(tmp)
            wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            render(wr.write)
            wr.release()
        os.replace(tmp, out_path)                    # chỉ đổi tên khi đã xong -> không có file hỏng
    finally:
        cap.release()
        if os.path.exists(tmp):
            os.remove(tmp)