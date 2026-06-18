"""Google pass — Vosk jo suna, uski spelling saaf karta hai."""

import speech_recognition as sr

from app.audio_utils import numpy_to_audio_data, prepare_for_stt
from app.config import LANGUAGE_OPTIONS

_recognizer = sr.Recognizer()


def refine_heard(
    audio,
    sample_rate: int,
    heard: str,
    language_mode: str = "auto",
) -> str:
    heard = " ".join(heard.split()).strip()
    if not heard:
        return ""

    audio, rate = prepare_for_stt(audio, sample_rate)
    if len(audio) < int(rate * 0.08):
        return heard

    langs = LANGUAGE_OPTIONS.get(language_mode, LANGUAGE_OPTIONS["auto"])
    data = numpy_to_audio_data(audio, rate)

    for lang in langs:
        try:
            text = _recognizer.recognize_google(data, language=lang).strip()
            if text:
                return text
        except sr.UnknownValueError:
            continue
        except sr.RequestError:
            break

    return heard
