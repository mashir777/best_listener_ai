import numpy as np
import speech_recognition as sr


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate or len(audio) == 0:
        return audio.astype(np.float32)
    new_len = int(len(audio) * to_rate / from_rate)
    if new_len < 1:
        return audio.astype(np.float32)
    x_old = np.arange(len(audio), dtype=np.float32)
    x_new = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def enhance_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize mic audio so STT hears words clearly, not too quiet or clipped."""
    audio = audio.astype(np.float32)
    audio = audio - np.mean(audio)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    if peak < 1e-6 and rms < 1e-6:
        return audio
    target_rms = 0.12
    gain = min(target_rms / max(rms, 1e-6), 12.0)
    if peak * gain > 0.98:
        gain = 0.95 / peak
    return np.clip(audio * gain, -1.0, 1.0)


def prepare_for_stt(audio: np.ndarray, sample_rate: int, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    audio = enhance_audio(audio)
    if sample_rate != target_rate:
        audio = resample_audio(audio, sample_rate, target_rate)
        sample_rate = target_rate
    return audio, sample_rate


def numpy_to_audio_data(audio: np.ndarray, sample_rate: int) -> sr.AudioData:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return sr.AudioData(pcm, sample_rate, 2)
