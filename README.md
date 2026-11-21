# Tìm kiếm & theo dõi đối tượng trong video bằng mô tả

```bash
pip install -r requirements.txt
python index_video.py --video input.mp4 --out data        # index 1 lần
python search.py --data data --query "người đàn ông áo đỏ đeo balo đen" --topk 5
```
Kết quả: danh sách track (thời điểm, điểm tương đồng) và clip có vẽ khung trong `data/results/`.

Gợi ý nâng cấp: đổi `CLIP_NAME` sang `openai/clip-vit-large-patch14`; thêm Re-ID (OSNet);
dùng SAM 2 / Grounding DINO để bám theo; thêm bộ lọc thuộc tính màu và VLM xác minh top-K.

## Giao diện Streamlit
```bash
streamlit run app.py
```
Nhập mô tả, xem timeline các khoảnh khắc khớp, chọn kết quả để xem clip có vẽ khung, tải clip về
và xem quỹ đạo di chuyển. Nên cài `ffmpeg` để clip phát được trong trình duyệt (chuyển sang H.264).
