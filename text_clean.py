import re

_HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "like and subscribe",
    "please subscribe",
    "subscribe to my channel",
    "see you next time",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_hallucination(text: str) -> bool:
    t = normalize(text).lower().rstrip(".")
    if not t or t in {".", ",", "...", "…", "listening", "listening..."}:
        return True
    return t in _HALLUCINATION_PHRASES


def collapse_repetition(text: str) -> str:
    """'This is better. This is better. ...' -> 'This is better.'"""
    text = normalize(text)
    if not text:
        return ""

    parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    if not parts:
        return text

    if len(set(p.lower() for p in parts)) == 1:
        return parts[0] + "."

    out = [parts[0]]
    for part in parts[1:]:
        if part.lower() != out[-1].lower():
            out.append(part)
    return ". ".join(out) + "."


def clean_transcript(text: str) -> str:
    text = collapse_repetition(normalize(text))
    if is_hallucination(text):
        return ""
    return text
