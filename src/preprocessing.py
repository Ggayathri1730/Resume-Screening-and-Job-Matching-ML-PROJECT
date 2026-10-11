import re

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9+#.\-/ ]+", " ", text)
    return text.strip()

def normalize_text(text: str) -> str:
    return clean_text(text).lower()
