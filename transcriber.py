# import re
# import threading
# import time
# from typing import Callable, Optional

# import numpy as np
# import speech_recognition as sr
# from faster_whisper import WhisperModel

# from app.audio_utils import numpy_to_audio_data, prepare_for_stt
# from app.config import LANGUAGE_OPTIONS, get_whisper_model
# from app.text_clean import clean_transcript

# _WHISPER: Optional[WhisperModel] = None
# _WHISPER_LOCK = threading.Lock()
# _WHISPER_READY = threading.Event()

# # Only block obvious invented phrases — not real short words like "go", "you", "the".
# _HALLUCINATION_PHRASES = {
#     "thank you for watching",
#     "thanks for watching",
#     "like and subscribe",
#     "please subscribe",
#     "subscribe to my channel",
#     "see you next time",
#     "see you in the next video",
# }


# def _normalize_text(text: str) -> str:
#     return re.sub(r"\s+", " ", text.strip())


# def _is_hallucination(text: str) -> bool:
#     t = _normalize_text(text).lower().rstrip(".")
#     if not t or t in {".", ",", "...", "…"}:
#         return True
#     return t in _HALLUCINATION_PHRASES


# def _words_per_second_ok(text: str, duration_sec: float) -> bool:
#     words = len(text.split())
#     if words == 0:
#         return False
#     return words <= max(2, duration_sec * 5.0 + 1.0)


# def _audio_has_speech(audio: np.ndarray, min_rms: float = 0.008) -> bool:
#     rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
#     return rms >= min_rms


# def preload_whisper(on_status: Optional[Callable[[str], None]] = None) -> bool:
#     global _WHISPER
#     if _WHISPER_READY.is_set():
#         return True
#     with _WHISPER_LOCK:
#         if _WHISPER is not None:
#             _WHISPER_READY.set()
#             return True
#         try:
#             if on_status:
#                 on_status("Loading speech model...")
#             _WHISPER = WhisperModel(
#                 get_whisper_model(),
#                 device="cpu",
#                 compute_type="int8",
#             )
#             dummy = np.zeros(16000, dtype=np.float32)
#             _WHISPER.transcribe(dummy, beam_size=1, vad_filter=False)
#             _WHISPER_READY.set()
#             return True
#         except Exception as e:
#             print(f"Whisper preload: {e}")
#             return False


# class FastTranscriber:
#     """Writes what was actually said — Google first, Whisper backup."""

#     def __init__(
#         self,
#         on_text: Callable[[str, float], None],
#         on_error: Callable[[str], None],
#         on_done: Optional[Callable[[], None]] = None,
#         language_mode: str = "auto",
#     ):
#         self.on_text = on_text
#         self.on_error = on_error
#         self.on_done = on_done
#         self.language_mode = language_mode
#         self._recognizer = sr.Recognizer()
#         self._ready = threading.Event()

#     @property
#     def is_ready(self) -> bool:
#         return self._ready.is_set()

#     def set_language_mode(self, mode: str):
#         self.language_mode = mode

#     def preload(self):
#         threading.Thread(target=self._load, daemon=True).start()

#     def _load(self):
#         preload_whisper()
#         self._ready.set()

#     def transcribe_async(self, audio: np.ndarray, sample_rate: int):
#         threading.Thread(
#             target=self._transcribe,
#             args=(audio, sample_rate),
#             daemon=True,
#         ).start()

#     def _transcribe(self, audio: np.ndarray, sample_rate: int):
#         try:
#             text, elapsed = self._transcribe_accurate(audio, sample_rate)
#             text = clean_transcript(text)
#             if text:
#                 self.on_text(text, elapsed)
#         except Exception as e:
#             self.on_error(f"Error: {e}")
#         finally:
#             if self.on_done:
#                 self.on_done()

#     def _google_try(self, audio_data: sr.AudioData) -> str:
#         langs = LANGUAGE_OPTIONS.get(self.language_mode, LANGUAGE_OPTIONS["auto"])
#         for lang in langs:
#             try:
#                 text = self._recognizer.recognize_google(audio_data, language=lang).strip()
#                 if text:
#                     return text
#             except sr.UnknownValueError:
#                 continue
#             except sr.RequestError:
#                 break
#         return ""

#     def _whisper_lang(self) -> Optional[str]:
#         if self.language_mode == "english":
#             return "en"
#         if self.language_mode == "urdu":
#             return "ur"
#         return None

#     def _whisper_try(self, audio: np.ndarray) -> str:
#         if not _WHISPER_READY.wait(timeout=30.0):
#             return ""

#         with _WHISPER_LOCK:
#             if _WHISPER is None:
#                 return ""
#             segments, _ = _WHISPER.transcribe(
#                 audio,
#                 language=self._whisper_lang(),
#                 beam_size=3,
#                 vad_filter=False,
#                 best_of=1,
#                 temperature=0.0,
#                 word_timestamps=False,
#                 condition_on_previous_text=False,
#                 no_speech_threshold=0.65,
#                 log_prob_threshold=-0.8,
#                 compression_ratio_threshold=2.4,
#             )
#             parts: list[str] = []
#             for seg in segments:
#                 if seg.no_speech_prob > 0.65:
#                     continue
#                 if seg.avg_logprob < -1.0:
#                     continue
#                 piece = seg.text.strip()
#                 if piece:
#                     parts.append(piece)

#         text = _normalize_text(" ".join(parts))
#         if _is_hallucination(text):
#             return ""
#         return text

#     def _pick_best(self, google: str, whisper: str, duration_sec: float) -> str:
#         google = _normalize_text(google)
#         whisper = _normalize_text(whisper)

#         if _is_hallucination(google):
#             google = ""
#         if _is_hallucination(whisper):
#             whisper = ""

#         if not google and not whisper:
#             return ""

#         if google and not whisper:
#             return google if _words_per_second_ok(google, duration_sec) else ""

#         if whisper and not google:
#             return whisper if _words_per_second_ok(whisper, duration_sec) else ""

#         # Both returned text — prefer shorter faithful result to avoid invented extra words.
#         if duration_sec < 4.0:
#             shorter = google if len(google) <= len(whisper) else whisper
#             longer = whisper if shorter == google else google
#             if len(longer) > len(shorter) * 1.8:
#                 return shorter if _words_per_second_ok(shorter, duration_sec) else longer
#             if google.lower() == whisper.lower():
#                 return google
#             return shorter if _words_per_second_ok(shorter, duration_sec) else longer

#         return google if _words_per_second_ok(google, duration_sec) else whisper

#     def _transcribe_accurate(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
#         t0 = time.perf_counter()
#         audio, stt_rate = prepare_for_stt(audio, sample_rate)
#         duration_sec = len(audio) / stt_rate

#         if duration_sec < 0.1 or not _audio_has_speech(audio):
#             return "", time.perf_counter() - t0

#         audio_data = numpy_to_audio_data(audio, stt_rate)

#         # Google first — fast and accurate for clear speech.
#         google = self._google_try(audio_data)
#         if google and not _is_hallucination(google) and _words_per_second_ok(google, duration_sec):
#             whisper = self._whisper_try(audio)
#             if not whisper or google.lower() in whisper.lower() or whisper.lower() in google.lower():
#                 return google, time.perf_counter() - t0
#             return self._pick_best(google, whisper, duration_sec), time.perf_counter() - t0

#         whisper = self._whisper_try(audio)
#         text = self._pick_best(google, whisper, duration_sec)
#         return text, time.perf_counter() - t0


import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel

from app.audio_utils import numpy_to_audio_data, prepare_for_stt
from app.config import LANGUAGE_OPTIONS, get_whisper_model
from app.text_clean import clean_transcript

_WHISPER: Optional[WhisperModel] = None
_WHISPER_LOCK = threading.Lock()
_WHISPER_READY = threading.Event()

_HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "like and subscribe",
    "please subscribe",
    "subscribe to my channel",
    "see you next time",
    "see you in the next video",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _is_hallucination(text: str) -> bool:
    t = _normalize_text(text).lower().rstrip(".")
    if not t or t in {".", ",", "...", "…"}:
        return True
    return t in _HALLUCINATION_PHRASES


def _words_per_second_ok(text: str, duration_sec: float) -> bool:
    words = len(text.split())
    if words == 0:
        return False
    return words <= max(2, duration_sec * 5.0 + 1.0)


def _audio_has_speech(audio: np.ndarray, min_rms: float = 0.008) -> bool:
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    return rms >= min_rms


def preload_whisper(on_status: Optional[Callable[[str], None]] = None) -> bool:
    global _WHISPER
    if _WHISPER_READY.is_set():
        return True
    with _WHISPER_LOCK:
        if _WHISPER is not None:
            _WHISPER_READY.set()
            return True
        try:
            if on_status:
                on_status("Loading speech model...")
            _WHISPER = WhisperModel(
                get_whisper_model(),
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                inter_op_threads=2
            )
            dummy = np.zeros(16000, dtype=np.float32)
            _WHISPER.transcribe(dummy, beam_size=1, vad_filter=True)
            _WHISPER_READY.set()
            return True
        except Exception as e:
            print(f"Whisper preload: {e}")
            return False


class FastTranscriber:
    """Writes what was actually said — Runs Google & Whisper in parallel for zero delay."""

    def __init__(
        self,
        on_text: Callable[[str, float], None],
        on_error: Callable[[str], None],
        on_done: Optional[Callable[[], None]] = None,
        language_mode: str = "auto",
    ):
        self.on_text = on_text
        self.on_error = on_error
        self.on_done = on_done
        self.language_mode = language_mode
        self._recognizer = sr.Recognizer()
        self._ready = threading.Event()
        # Thread pool executor for lightning fast parallel runs
        self._executor = ThreadPoolExecutor(max_workers=2)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def set_language_mode(self, mode: str):
        self.language_mode = mode

    def preload(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        preload_whisper()
        self._ready.set()

    def transcribe_async(self, audio: np.ndarray, sample_rate: int):
        threading.Thread(
            target=self._transcribe,
            args=(audio, sample_rate),
            daemon=True,
        ).start()

    def _transcribe(self, audio: np.ndarray, sample_rate: int):
        try:
            text, elapsed = self._transcribe_accurate(audio, sample_rate)
            text = clean_transcript(text)
            if text:
                self.on_text(text, elapsed)
        except Exception as e:
            self.on_error(f"Error: {e}")
        finally:
            if self.on_done:
                self.on_done()

    def _google_try(self, audio_data: sr.AudioData) -> str:
        langs = LANGUAGE_OPTIONS.get(self.language_mode, LANGUAGE_OPTIONS["auto"])
        for lang in langs:
            try:
                # OPTIMIZATION: Sub-second timeout (0.8s) so network lag doesn't freeze the UI
                text = self._recognizer.recognize_google(audio_data, language=lang, timeout=0.8).strip()
                if text:
                    return text
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                continue
            except sr.RequestError:
                break
        return ""

    def _whisper_lang(self) -> Optional[str]:
        if self.language_mode == "english":
            return "en"
        if self.language_mode == "urdu":
            return "ur"
        return None

    def _whisper_try(self, audio: np.ndarray) -> str:
        if not _WHISPER_READY.wait(timeout=2.0):
            return ""

        with _WHISPER_LOCK:
            if _WHISPER is None:
                return ""
            
            segments, _ = _WHISPER.transcribe(
                audio,
                language=self._whisper_lang(),
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_speech_duration_ms=150),
                best_of=1,
                temperature=0.0,
                word_timestamps=False,
                condition_on_previous_text=False,
            )
            parts: list[str] = []
            for seg in segments:
                piece = seg.text.strip()
                if piece:
                    parts.append(piece)

        text = _normalize_text(" ".join(parts))
        if _is_hallucination(text):
            return ""
        return text

    def _pick_best(self, google: str, whisper: str, duration_sec: float) -> str:
        google = _normalize_text(google)
        whisper = _normalize_text(whisper)

        if _is_hallucination(google):
            google = ""
        if _is_hallucination(whisper):
            whisper = ""

        if not google and not whisper:
            return ""

        if google and not whisper:
            return google if _words_per_second_ok(google, duration_sec) else ""

        if whisper and not google:
            return whisper if _words_per_second_ok(whisper, duration_sec) else ""

        if duration_sec < 4.0:
            shorter = google if len(google) <= len(whisper) else whisper
            longer = whisper if shorter == google else google
            if len(longer) > len(shorter) * 1.8:
                return shorter if _words_per_second_ok(shorter, duration_sec) else longer
            if google.lower() == whisper.lower():
                return google
            return shorter if _words_per_second_ok(shorter, duration_sec) else longer

        return google if _words_per_second_ok(google, duration_sec) else whisper

    def _transcribe_accurate(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        t0 = time.perf_counter()
        audio, stt_rate = prepare_for_stt(audio, sample_rate)
        duration_sec = len(audio) / stt_rate

        if duration_sec < 0.1 or not _audio_has_speech(audio):
            return "", time.perf_counter() - t0

        audio_data = numpy_to_audio_data(audio, stt_rate)

        # PARALLEL EXECUTOR: Dono engines ko ek sath run karo (Race Mode)
        # Jo pehle khatam hoga aur valid text dega, system use process karega.
        future_google = self._executor.submit(self._google_try, audio_data)
        future_whisper = self._executor.submit(self._whisper_try, audio)

        google_res = ""
        whisper_res = ""
        
        for future in as_completed([future_google, future_whisper]):
            if future == future_google:
                google_res = future.result()
                # Agar google fast response de de aur text valid ho, toh whisper ka mazeed wait mat karo
                if google_res and not _is_hallucination(google_res) and _words_per_second_ok(google_res, duration_sec):
                    return google_res, time.perf_counter() - t0
            else:
                whisper_res = future.result()

        text = self._pick_best(google_res, whisper_res, duration_sec)
        return text, time.perf_counter() - t0