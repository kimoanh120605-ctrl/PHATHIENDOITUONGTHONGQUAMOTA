"""Tiện ích dùng chung: nạp CLIP, mã hóa ảnh/văn bản."""
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor

CLIP_NAME = "openai/clip-vit-base-patch32"  # có thể đổi sang ViT-L/14 để chính xác hơn


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(device=None):
    device = device or get_device()
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAME)
    if device == "cuda":
        model = model.half()
    return model, proc, device


@torch.no_grad()
def embed_images(model, proc, pil_images, device, batch=64):
    out = []
    for i in range(0, len(pil_images), batch):
        inp = proc(images=pil_images[i:i + batch], return_tensors="pt").to(device)
        if device == "cuda":
            inp["pixel_values"] = inp["pixel_values"].half()
        # Không dùng get_image_features vì kiểu trả về khác nhau giữa các bản transformers
        pooled = model.vision_model(pixel_values=inp["pixel_values"]).pooler_output
        f = model.visual_projection(pooled).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def embed_text(model, proc, texts, device):
    inp = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    pooled = model.text_model(input_ids=inp["input_ids"],
                              attention_mask=inp["attention_mask"]).pooler_output
    f = model.text_projection(pooled).float()
    return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()