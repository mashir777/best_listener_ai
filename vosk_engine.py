import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from vosk import Model, SetLogLevel

from app.config import PROJECT_ROOT

SetLogLevel(-1)

MODELS_DIR = PROJECT_ROOT / "models"
PRIMARY_MODEL = MODELS_DIR / "vosk-model-en-us-0.22-lgraph"
PRIMARY_URL = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
FALLBACK_MODEL = MODELS_DIR / "vosk-model-small-en-us-0.15"
FALLBACK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

_MODEL: Optional[Model] = None
_MODEL_LOCK = threading.Lock()


def ensure_vosk_model(on_status: Optional[Callable[[str], None]] = None) -> Path:
    if PRIMARY_MODEL.exists():
        return PRIMARY_MODEL

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not PRIMARY_MODEL.exists():
        zip_path = MODELS_DIR / "vosk-model-en-us-0.22-lgraph.zip"
        if on_status:
            on_status("Downloading speech model (one time)...")
        try:
            urllib.request.urlretrieve(PRIMARY_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(MODELS_DIR)
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass

    if PRIMARY_MODEL.exists():
        return PRIMARY_MODEL
    if FALLBACK_MODEL.exists():
        return FALLBACK_MODEL

    zip_path = MODELS_DIR / "vosk-model-small.zip"
    if on_status:
        on_status("Downloading speech model...")
    urllib.request.urlretrieve(FALLBACK_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODELS_DIR)
    zip_path.unlink(missing_ok=True)
    return PRIMARY_MODEL if PRIMARY_MODEL.exists() else FALLBACK_MODEL


def get_shared_model() -> Model:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = Model(str(ensure_vosk_model()))
        return _MODEL


def preload_vosk(on_status: Optional[Callable[[str], None]] = None) -> bool:
    try:
        get_shared_model()
        return True
    except Exception as e:
        print(f"Vosk preload: {e}")
        return False
