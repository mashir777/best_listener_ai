import threading
import time
from typing import Callable, Optional

from google import genai
from google.genai import types

from app.config import INTERVIEW_TYPES, get_gemini_key, get_gemini_model


class AIResponder:
    """Fast Gemini answers with streaming and pre-warmed client."""

    def __init__(
        self,
        on_answer: Callable[[str, float], None],
        on_chunk: Callable[[str], None],
        on_error: Callable[[str], None],
        interview_type: str = "general_hr",
        resume_context: str = "",
    ):
        self.on_answer = on_answer
        self.on_chunk = on_chunk
        self.on_error = on_error
        self.interview_type = interview_type
        self.resume_context = resume_context
        self._client: Optional[genai.Client] = None
        self._chat = None
        self._ready = threading.Event()
        self._gen_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def preload(self):
        threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self):
        try:
            self._get_client()
            self._ready.set()
        except Exception as e:
            self.on_error(f"AI warmup failed: {e}")

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=get_gemini_key())
        return self._client

    def _reset_chat(self):
        client = self._get_client()
        self._chat = client.chats.create(
            model=get_gemini_model(),
            config=types.GenerateContentConfig(
                system_instruction=self._build_system_prompt(),
                temperature=0.25,
                max_output_tokens=280,
            ),
        )

    def set_interview_type(self, interview_type: str):
        self.interview_type = interview_type
        self._chat = None

    def set_resume_context(self, context: str):
        self.resume_context = context
        self._chat = None

    def clear_history(self):
        self._chat = None

    def _build_system_prompt(self) -> str:
        base = INTERVIEW_TYPES.get(self.interview_type, INTERVIEW_TYPES["general_hr"])
        rules = (
            "You are the job candidate in a LIVE interview. "
            "Answer in first person (I, my, me). "
            "Give accurate, specific, professional answers — not generic fluff. "
            "Use 3-5 clear sentences. Be confident and natural. "
            "Never say you are AI. Never refuse — always give the best real interview answer."
        )
        if self.resume_context.strip():
            return f"{base}\n\nCandidate background (use this for accurate answers):\n{self.resume_context.strip()}\n\n{rules}"
        return f"{base}\n\n{rules}"

    def generate_async(self, question: str):
        threading.Thread(target=self._generate, args=(question,), daemon=True).start()

    def _generate(self, question: str):
        with self._gen_lock:
            for attempt in range(2):
                try:
                    if self._chat is None:
                        self._reset_chat()

                    t0 = time.perf_counter()
                    parts = []
                    first_token_at: Optional[float] = None

                    for chunk in self._chat.send_message_stream(question):
                        if chunk.text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter() - t0
                            parts.append(chunk.text)
                            self.on_chunk("".join(parts))

                    answer = "".join(parts).strip()
                    total = time.perf_counter() - t0
                    if answer:
                        self.on_answer(answer, first_token_at or total)
                    else:
                        self.on_error("Empty answer from AI.")
                    return
                except Exception as e:
                    err = str(e)
                    if "429" in err and attempt == 0:
                        time.sleep(2)
                        self._chat = None
                        continue
                    self.on_error(f"AI error: {e}")
                    return
