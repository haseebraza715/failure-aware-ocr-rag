from __future__ import annotations

from functools import lru_cache
from pathlib import Path


GOT_OCR_MODEL = "stepfun-ai/GOT-OCR-2.0-hf"


@lru_cache(maxsize=1)
def _load_got_ocr(model_name: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto")
    return model.eval(), processor


def extract_got_ocr(image_path: Path, model_name: str = GOT_OCR_MODEL) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"GOT-OCR input image is missing: {image_path}")
    model, processor = _load_got_ocr(model_name)
    inputs = processor(str(image_path), return_tensors="pt").to(model.device)
    generated = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=4096,
    )
    prompt_length = inputs["input_ids"].shape[1]
    return processor.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
