"""Dịch truy vấn tiếng Việt -> tiếng Anh bằng luật (không cần API key) và suy ra lớp đối tượng."""
import re

COLORS = {"xanh dương": "blue", "xanh lam": "blue", "xanh da trời": "blue", "xanh lá": "green",
          "xanh": "blue", "trắng": "white", "đen": "black", "đỏ": "red", "vàng": "yellow",
          "cam": "orange", "xám": "gray", "ghi": "gray", "bạc": "silver", "nâu": "brown",
          "hồng": "pink", "tím": "purple"}
THINGS = {  # đồ vật / phương tiện có thể đi kèm màu: "áo đỏ" -> "red shirt"
    "xe ô tô": "car", "xe hơi": "car", "xe con": "car", "ô tô": "car", "ôtô": "car",
    "xe tải": "truck", "xe buýt": "bus", "xe bus": "bus", "xe khách": "bus",
    "xe máy": "motorcycle", "xe mô tô": "motorcycle", "mô tô": "motorcycle",
    "xe đạp": "bicycle", "mũ bảo hiểm": "helmet", "ba lô": "backpack", "balo": "backpack",
    "áo khoác": "jacket", "áo thun": "t-shirt", "áo": "shirt", "quần": "pants",
    "váy": "dress", "túi": "bag", "mũ": "hat", "nón": "hat",
}
OTHERS = {"người đàn ông": "man", "người phụ nữ": "woman", "đàn ông": "man", "phụ nữ": "woman",
          "bé trai": "boy", "bé gái": "girl", "trẻ em": "child", "người": "person",
          "đeo": "carrying", "mang": "carrying", "mặc": "wearing", "đội": "wearing",
          "cầm": "holding", "chở": "carrying"}

VEHICLE_CLS = ["car", "truck", "bus", "motorcycle", "bicycle"]
PERSON_WORDS = {"man", "woman", "boy", "girl", "child", "person"}


def _alt(keys):
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


_COL = _alt(COLORS)
_THING = _alt(THINGS)
_ALL = {**COLORS, **THINGS, **OTHERS}


def rule_parse(q: str) -> dict:
    s = " " + q.lower().strip() + " "
    # "áo đỏ" / "xe ô tô màu trắng" -> "red shirt" / "white car"
    s = re.sub(rf"(?<!\w)({_THING})(?:\s+màu)?\s+({_COL})(?!\w)",
               lambda m: f" {COLORS[m.group(2)]} {THINGS[m.group(1)]} ", s)
    # các từ còn lại
    s = re.sub(rf"(?<!\w)({_alt(_ALL)})(?!\w)", lambda m: f" {_ALL[m.group(1)]} ", s)
    # bỏ từ tiếng Việt chưa dịch được (còn dấu/ký tự ngoài a-z)
    words = [w for w in s.split() if re.fullmatch(r"[a-z0-9\-]+", w)]
    text = " ".join(words)
    if not text:
        return {"text_en": q, "cls": None}
    cls = next((c for c in VEHICLE_CLS if c in words), None)
    if cls is None and PERSON_WORDS & set(words):
        cls = "person"
    return {"text_en": text, "cls": cls}


_EN_COLORS = "blue|green|white|black|red|yellow|orange|gray|grey|silver|brown|pink|purple"
_EN_ITEMS = "car|truck|bus|motorcycle|bicycle|shirt|jacket|t-shirt|pants|dress"


def extract_attrs(text_en: str):
    """Lấy các cặp (màu, vật) từ câu tiếng Anh, ví dụ 'white car' -> [('white','car')]."""
    return re.findall(rf"\b({_EN_COLORS})\s+({_EN_ITEMS})\b", text_en.lower())