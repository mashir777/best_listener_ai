from dataclasses import dataclass
from typing import List

import sounddevice as sd


@dataclass
class AudioDevice:
    id: str
    label: str
    kind: str  # "mic" or "system"


def list_audio_devices() -> List[AudioDevice]:
    devices: List[AudioDevice] = []

    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                name = dev["name"]
                devices.append(AudioDevice(id=f"mic:{i}", label=f"Mic: {name}", kind="mic"))
    except Exception:
        pass

    try:
        import pyaudiowpatch as pyaudio

        with pyaudio.PyAudio() as p:
            for loopback in p.get_loopback_device_info_generator():
                idx = loopback["index"]
                name = loopback["name"]
                devices.append(
                    AudioDevice(
                        id=f"system:{idx}",
                        label=f"System: {name}",
                        kind="system",
                    )
                )
    except Exception:
        pass

    if not devices:
        devices.append(AudioDevice(id="mic:default", label="Default Microphone", kind="mic"))

    return devices


def parse_device_id(device_id: str) -> tuple[str, int | None]:
    if device_id == "mic:default":
        return "mic", None
    kind, _, raw = device_id.partition(":")
    if kind == "mic":
        return "mic", int(raw) if raw.isdigit() else None
    if kind == "system":
        return "system", int(raw) if raw.isdigit() else None
    return "mic", None
