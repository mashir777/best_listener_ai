import json
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer

from app.audio_utils import resample_audio
from app.devices import parse_device_id
from app.refine import refine_heard
from app.vosk_engine import get_shared_model, preload_vosk

VOSK_RATE = 16000
BLOCK_MS = 20


def _to_pcm(samples: np.ndarray) -> bytes:
    f = samples.astype(np.float32)
    if float(np.max(np.abs(f))) > 1.5:
        f = f / 32768.0
    f -= np.mean(f)
    peak = float(np.max(np.abs(f)))
    if peak > 1e-6:
        f = f / peak * 0.92
    return (f * 32767).astype(np.int16).tobytes()


class LiveListener:
    """Jo sunay — wahi likho. Live Vosk + clear Google final."""

    def __init__(
        self,
        on_partial: Callable[[str], None],
        on_final: Callable[[str, float], None],
        on_error: Callable[[str], None],
        device_id: str = "mic:default",
        language_mode: str = "auto",
    ):
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.device_id = device_id
        self.language_mode = language_mode
        self._kind, self._device_index = parse_device_id(device_id)
        self._listening = False
        self._thread: Optional[threading.Thread] = None
        self._phrase_audio: list[np.ndarray] = []

    @property
    def is_listening(self) -> bool:
        return self._listening

    def set_language_mode(self, mode: str):
        self.language_mode = mode

    @staticmethod
    def preload(
        on_ready: Callable[[], None],
        on_error: Callable[[str], None],
        on_status: Optional[Callable[[str], None]] = None,
    ):
        def run():
            try:
                if not preload_vosk(on_status):
                    on_error("Model load failed")
                    return
                on_ready()
            except Exception as e:
                on_error(f"Error: {e}")

        threading.Thread(target=run, daemon=True).start()

    def start(self):
        if self._listening:
            return
        self._phrase_audio.clear()
        self._listening = True
        fn = self._system_loop if self._kind == "system" else self._mic_loop
        self._thread = threading.Thread(target=fn, daemon=True)
        self._thread.start()

    def stop(self):
        self._listening = False

    def _finish_phrase(self, audio: np.ndarray, vosk_text: str):
        def run():
            text = refine_heard(audio, VOSK_RATE, vosk_text, self.language_mode)
            if text and self._listening:
                self.on_partial("")
                self.on_final(text, 0.0)

        threading.Thread(target=run, daemon=True).start()

    def _feed(self, rec: KaldiRecognizer, samples: np.ndarray):
        self._phrase_audio.append(samples.astype(np.float32))

        if rec.AcceptWaveform(_to_pcm(samples)):
            vosk = json.loads(rec.Result()).get("text", "").strip()
            audio = (
                np.concatenate(self._phrase_audio)
                if self._phrase_audio
                else np.array([], dtype=np.float32)
            )
            self._phrase_audio.clear()
            if vosk:
                self._finish_phrase(audio, vosk)
        else:
            text = json.loads(rec.PartialResult()).get("partial", "").strip()
            if text:
                self.on_partial(text)

    def _flush(self, rec: KaldiRecognizer):
        vosk = json.loads(rec.FinalResult()).get("text", "").strip()
        audio = (
            np.concatenate(self._phrase_audio)
            if self._phrase_audio
            else np.array([], dtype=np.float32)
        )
        self._phrase_audio.clear()
        if vosk:
            self._finish_phrase(audio, vosk)

    def _mic_loop(self):
        try:
            model = get_shared_model()
            rec = KaldiRecognizer(model, VOSK_RATE)
            rec.SetWords(True)
            pending = np.array([], dtype=np.float32)

            if self._device_index is not None:
                dev = sd.query_devices(self._device_index)
            else:
                dev = sd.query_devices(kind="input")
            rate = int(dev["default_samplerate"])
            rate = min(max(rate, 16000), 48000)
            block = int(rate * BLOCK_MS / 1000)

            kwargs: dict = {
                "samplerate": rate,
                "channels": 1,
                "dtype": "float32",
                "blocksize": block,
                "latency": "low",
            }
            if self._device_index is not None:
                kwargs["device"] = self._device_index

            with sd.InputStream(**kwargs) as stream:
                while self._listening:
                    data, _ = stream.read(block)
                    samples = data.flatten().astype(np.float32)

                    if rate != VOSK_RATE:
                        pending = np.concatenate([pending, samples])
                        step = int(rate * BLOCK_MS / 1000)
                        while len(pending) >= step:
                            piece = pending[:step]
                            pending = pending[step:]
                            mono = resample_audio(piece, rate, VOSK_RATE)
                            self._feed(rec, mono)
                    else:
                        self._feed(rec, samples)

                self._flush(rec)
        except Exception as e:
            self.on_error(f"Mic error: {e}")

    def _system_loop(self):
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            self.on_error("Install: pip install pyaudiowpatch")
            return

        try:
            model = get_shared_model()
            rec = KaldiRecognizer(model, VOSK_RATE)
            rec.SetWords(True)
            pending = np.array([], dtype=np.float32)

            with pyaudio.PyAudio() as p:
                if self._device_index is not None:
                    info = p.get_device_info_by_index(self._device_index)
                else:
                    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                    info = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
                    if not info.get("isLoopbackDevice"):
                        for lb in p.get_loopback_device_info_generator():
                            if info["name"] in lb["name"]:
                                info = lb
                                break

                rate = int(info["defaultSampleRate"])
                ch = max(1, int(info["maxInputChannels"]))
                chunk = int(rate * BLOCK_MS / 1000)

                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=ch,
                    rate=rate,
                    frames_per_buffer=chunk,
                    input=True,
                    input_device_index=info["index"],
                )
                stream.start_stream()

                while self._listening:
                    raw = stream.read(chunk, exception_on_overflow=False)
                    samples = np.frombuffer(raw, dtype=np.float32)
                    if ch > 1:
                        samples = samples.reshape(-1, ch).mean(axis=1)

                    if rate != VOSK_RATE:
                        pending = np.concatenate([pending, samples])
                        step = int(rate * BLOCK_MS / 1000)
                        while len(pending) >= step:
                            piece = pending[:step]
                            pending = pending[step:]
                            mono = resample_audio(piece, rate, VOSK_RATE)
                            self._feed(rec, mono)
                    else:
                        self._feed(rec, samples)

                stream.stop_stream()
                stream.close()
                self._flush(rec)
        except Exception as e:
            self.on_error(f"System audio error: {e}")
