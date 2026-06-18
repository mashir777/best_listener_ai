from typing import Optional

import customtkinter as ctk

from app.devices import list_audio_devices
from app.live_listener import LiveListener

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ListenerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Listener")
        self.geometry("720x560")
        self.minsize(620, 480)

        self._listener: Optional[LiveListener] = None
        self._device_map: dict[str, str] = {}
        self._done_lines: list[str] = []
        self._live = ""
        self.device_menu: Optional[ctk.CTkOptionMenu] = None

        self._build_ui()
        self._init_backend()

    def _build_ui(self):
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=16, pady=(16, 8))

        ctk.CTkLabel(header, text="Listener", font=ctk.CTkFont(size=24, weight="bold")).pack(
            side="left", padx=8
        )
        self.status_label = ctk.CTkLabel(header, text="Loading...", text_color="gray")
        self.status_label.pack(side="right", padx=8)

        settings = ctk.CTkFrame(self)
        settings.pack(fill="x", padx=16, pady=8)

        ctk.CTkLabel(settings, text="Source:").pack(side="left", padx=(8, 4))
        self.device_var = ctk.StringVar(value="")
        self.device_menu = ctk.CTkOptionMenu(settings, variable=self.device_var, width=300)
        self.device_menu.pack(side="left", padx=4)

        ctk.CTkButton(settings, text="Refresh", command=self._refresh_devices, width=80, fg_color="gray30").pack(
            side="left", padx=6
        )
        ctk.CTkButton(settings, text="Clear", command=self._clear_text, width=70, fg_color="gray30").pack(
            side="left", padx=2
        )

        self._refresh_devices()

        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=16, pady=8)

        ctk.CTkLabel(
            frame,
            text="Jo sunay — saaf aur clear likha jayega",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2ecc71",
        ).pack(anchor="w", padx=8, pady=(8, 4))

        self.text_box = ctk.CTkTextbox(frame, font=ctk.CTkFont(size=18))
        self.text_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        controls = ctk.CTkFrame(self)
        controls.pack(fill="x", padx=16, pady=(0, 16))

        self.listen_btn = ctk.CTkButton(
            controls,
            text="Start Listening",
            command=self._toggle_listening,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2ecc71",
            hover_color="#27ae60",
            state="disabled",
        )
        self.listen_btn.pack(side="left", fill="x", expand=True, padx=(8, 4), pady=8)

        ctk.CTkButton(
            controls, text="Copy", command=self._copy_text, height=50, width=90, fg_color="gray30"
        ).pack(side="right", padx=(4, 8), pady=8)

    def _refresh_devices(self):
        devices = list_audio_devices()
        labels = []
        self._device_map.clear()
        for dev in devices:
            labels.append(dev.label)
            self._device_map[dev.label] = dev.id
        if not labels:
            labels = ["Default Microphone"]
            self._device_map["Default Microphone"] = "mic:default"
        if self.device_menu:
            self.device_menu.configure(values=labels)
        if not self.device_var.get() or self.device_var.get() not in labels:
            mic = next((d.label for d in devices if d.kind == "mic"), labels[0])
            self.device_var.set(mic)

    def _init_backend(self):
        LiveListener.preload(
            on_ready=lambda: self.after(0, self._on_ready),
            on_error=lambda e: self.after(0, lambda: self._on_error(e)),
            on_status=lambda s: self.after(0, lambda: self._set_status(s)),
        )

    def _on_ready(self):
        self.listen_btn.configure(state="normal")
        self._set_status("Ready")

    def _show(self):
        parts = list(self._done_lines)
        if self._live:
            parts.append(self._live)
        text = "\n".join(parts)
        self.text_box.delete("1.0", "end")
        if text:
            self.text_box.insert("1.0", text)
        self.text_box.see("end")

    def _clear_text(self):
        self._done_lines.clear()
        self._live = ""
        self.text_box.delete("1.0", "end")

    def _toggle_listening(self):
        if self._listener and self._listener.is_listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self):
        device_id = self._device_map.get(self.device_var.get(), "mic:default")
        self._listener = LiveListener(
            on_partial=self._on_partial,
            on_final=self._on_final,
            on_error=self._on_error,
            device_id=device_id,
        )
        self._listener.start()
        self.listen_btn.configure(text="Stop", fg_color="#e74c3c", hover_color="#c0392b")
        self._set_status("Listening...", "#2ecc71")

    def _stop_listening(self):
        if self._live:
            self._done_lines.append(self._live)
            self._live = ""
            self._show()
        if self._listener:
            self._listener.stop()
            self._listener = None
        self.listen_btn.configure(text="Start Listening", fg_color="#2ecc71", hover_color="#27ae60")
        self._set_status("Stopped")

    def _on_partial(self, text: str):
        self.after(0, lambda: self._apply_partial(text))

    def _apply_partial(self, text: str):
        self._live = text
        self._show()

    def _on_final(self, text: str, elapsed: float = 0.0):
        self.after(0, lambda: self._apply_final(text))

    def _apply_final(self, text: str):
        self._live = ""
        if text:
            if self._done_lines and self._done_lines[-1].lower() == text.lower():
                self._done_lines[-1] = text
            else:
                self._done_lines.append(text)
        self._show()

    def _on_error(self, message: str):
        self.after(0, lambda: self._set_status(message, "#e74c3c"))

    def _copy_text(self):
        text = self.text_box.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)

    def _set_status(self, text: str, color: str = "gray"):
        self.status_label.configure(text=text, text_color=color)

    def on_closing(self):
        self._stop_listening()
        self.destroy()


def run_app():
    app = ListenerApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
