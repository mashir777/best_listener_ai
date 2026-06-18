import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from app.config import DEFAULT_SETTINGS
from app.devices import parse_device_id


class SpeechListener:
    """Captures speech quickly and accurately with adaptive mic thresholds."""

    def __init__(
        self,
        on_audio_ready: Callable[[np.ndarray, int], None],
        device_id: str = "mic:default",
        chunk_ms: int = DEFAULT_SETTINGS["chunk_ms"],
        silence_duration: float = DEFAULT_SETTINGS["silence_duration_sec"],
        min_speech_duration: float = DEFAULT_SETTINGS["min_speech_duration_sec"],
        max_record_sec: float = DEFAULT_SETTINGS["max_record_sec"],
        pre_roll_chunks: int = DEFAULT_SETTINGS["pre_roll_chunks"],
    ):
        self.on_audio_ready = on_audio_ready
        self.chunk_ms = chunk_ms
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_record_sec = max_record_sec
        self.pre_roll_chunks = pre_roll_chunks

        self._listening = False
        self._thread: Optional[threading.Thread] = None
        self._start_threshold = 0.012
        self._continue_threshold = 0.007
        self._kind, self._device_index = parse_device_id(device_id)
        self._record_rate = DEFAULT_SETTINGS["record_rate"]

    @property
    def is_listening(self) -> bool:
        return self._listening

    def _resolve_record_rate(self) -> int:
        try:
            if self._device_index is not None:
                dev = sd.query_devices(self._device_index)
            else:
                dev = sd.query_devices(kind="input")
            rate = int(dev["default_samplerate"])
            return min(max(rate, 16000), 48000)
        except Exception:
            return DEFAULT_SETTINGS["record_rate"]

    def start(self):
        if self._listening:
            return
        self._listening = True
        if self._kind == "system":
            self._thread = threading.Thread(target=self._system_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._listening = False

    def _calibrate(self, levels: list[float]):
        noise = float(np.percentile(levels, 35)) if levels else 0.005
        self._start_threshold = max(noise * 3.0, 0.006)
        self._continue_threshold = max(noise * 1.8, 0.004)

    def _chunk_rms(self, chunk: np.ndarray) -> float:
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

    def _mic_loop(self):
        self._record_rate = self._resolve_record_rate()
        chunk_samples = int(self._record_rate * self.chunk_ms / 1000)
        silence_chunks = max(1, int(self.silence_duration / (self.chunk_ms / 1000)))
        min_speech_chunks = max(1, int(self.min_speech_duration / (self.chunk_ms / 1000)))
        max_chunks = int(self.max_record_sec / (self.chunk_ms / 1000))

        kwargs = {
            "samplerate": self._record_rate,
            "channels": 1,
            "dtype": "float32",
            "blocksize": chunk_samples,
        }
        if self._device_index is not None:
            kwargs["device"] = self._device_index

        try:
            with sd.InputStream(**kwargs) as stream:
                cal = []
                for _ in range(12):
                    data, _ = stream.read(chunk_samples)
                    cal.append(self._chunk_rms(data.flatten()))
                self._calibrate(cal)

                while self._listening:
                    audio = self._capture_utterance(
                        stream, chunk_samples, silence_chunks, min_speech_chunks, max_chunks
                    )
                    if audio is not None and len(audio) > 0:
                        self.on_audio_ready(audio, self._record_rate)
        except Exception as e:
            print(f"Microphone error: {e}")

    def _capture_utterance(
        self,
        stream,
        chunk_samples: int,
        silence_chunks_needed: int,
        min_speech_chunks: int,
        max_chunks: int,
    ) -> Optional[np.ndarray]:
        frames: list[np.ndarray] = []
        ring: list[np.ndarray] = []
        silent_count = 0
        speech_active = False

        for _ in range(max_chunks):
            if not self._listening:
                return None

            data, _ = stream.read(chunk_samples)
            chunk = data.flatten().astype(np.float32)
            rms = self._chunk_rms(chunk)

            ring.append(chunk)
            if len(ring) > self.pre_roll_chunks:
                ring.pop(0)

            if rms >= self._start_threshold or (speech_active and rms >= self._continue_threshold):
                if not speech_active:
                    frames.extend(ring[:-1])
                    speech_active = True
                silent_count = 0
                frames.append(chunk)
            elif speech_active:
                frames.append(chunk)
                silent_count += 1
                if silent_count >= silence_chunks_needed:
                    if len(frames) >= min_speech_chunks:
                        return np.concatenate(frames)
                    frames.clear()
                    speech_active = False
                    silent_count = 0

        if speech_active and frames:
            return np.concatenate(frames)
        return None

    def _system_loop(self):
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            print("pyaudiowpatch not installed")
            return

        chunk_samples = int(self._record_rate * self.chunk_ms / 1000)
        silence_chunks_needed = max(1, int(self.silence_duration / (self.chunk_ms / 1000)))
        min_speech_chunks = max(1, int(self.min_speech_duration / (self.chunk_ms / 1000)))
        max_chunks = int(self.max_record_sec / (self.chunk_ms / 1000))

        try:
            with pyaudio.PyAudio() as p:
                if self._device_index is not None:
                    device_info = p.get_device_info_by_index(self._device_index)
                else:
                    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                    device_info = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
                    if not device_info.get("isLoopbackDevice"):
                        for lb in p.get_loopback_device_info_generator():
                            if device_info["name"] in lb["name"]:
                                device_info = lb
                                break

                rate = int(device_info["defaultSampleRate"])
                channels = device_info["maxInputChannels"]
                self._record_rate = rate

                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=rate,
                    frames_per_buffer=chunk_samples,
                    input=True,
                    input_device_index=device_info["index"],
                )
                stream.start_stream()

                cal = []
                for _ in range(12):
                    raw = stream.read(chunk_samples, exception_on_overflow=False)
                    chunk = np.frombuffer(raw, dtype=np.float32)
                    if channels > 1:
                        chunk = chunk.reshape(-1, channels).mean(axis=1)
                    cal.append(self._chunk_rms(chunk))
                self._calibrate(cal)

                while self._listening:
                    frames: list[np.ndarray] = []
                    ring: list[np.ndarray] = []
                    silent_count = 0
                    speech_active = False

                    for _ in range(max_chunks):
                        if not self._listening:
                            break
                        raw = stream.read(chunk_samples, exception_on_overflow=False)
                        chunk = np.frombuffer(raw, dtype=np.float32)
                        if channels > 1:
                            chunk = chunk.reshape(-1, channels).mean(axis=1)
                        rms = self._chunk_rms(chunk)

                        ring.append(chunk)
                        if len(ring) > self.pre_roll_chunks:
                            ring.pop(0)

                        if rms >= self._start_threshold or (speech_active and rms >= self._continue_threshold):
                            if not speech_active:
                                frames.extend(ring[:-1])
                                speech_active = True
                            silent_count = 0
                            frames.append(chunk)
                        elif speech_active:
                            frames.append(chunk)
                            silent_count += 1
                            if silent_count >= silence_chunks_needed:
                                if len(frames) >= min_speech_chunks:
                                    audio = np.concatenate(frames).astype(np.float32)
                                    self.on_audio_ready(audio, rate)
                                frames.clear()
                                speech_active = False
                                silent_count = 0

                stream.stop_stream()
                stream.close()
        except Exception as e:
            print(f"System audio error: {e}")
