import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
KEYS_FILE = CONFIG_DIR / "keys.env"

DEFAULT_SETTINGS = {
    "sample_rate": 16000,
    "record_rate": 48000,
    "chunk_ms": 30,
    "silence_threshold": 0.004,
    "silence_duration_sec": 0.7,
    "min_speech_duration_sec": 0.25,
    "max_record_sec": 25,
    "pre_roll_chunks": 6,
    "whisper_model": "base",
}

LANGUAGE_OPTIONS = {
    "auto": ["en-US", "ur-PK"],
    "english": ["en-US"],
    "urdu": ["ur-PK", "en-US"],
}


def load_env():
    if KEYS_FILE.exists():
        from dotenv import load_dotenv
        load_dotenv(KEYS_FILE)


def get_whisper_model() -> str:
    load_env()
    return os.getenv("WHISPER_MODEL", DEFAULT_SETTINGS["whisper_model"])
