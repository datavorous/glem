import os
import queue
import random
import threading

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play


class TTSRunner:
    def __init__(
        self,
        voice_id: str | None = None,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "mp3_44100_128",
    ):
        load_dotenv()
        self.voice_id = voice_id or "JBFqnCBsd6RMkjVDRZzb"  # "JBFqnCBsd6RMkjVDRZzb"
        self.model_id = model_id
        self.output_format = output_format
        self._api_keys = self._load_api_keys()
        self._queue: "queue.Queue[tuple[str, threading.Event] | None]" = queue.Queue()
        self._thread = None
        self._client = None
        self._clients = {}
        self._ready = threading.Event()
        self._stop = threading.Event()

    def _load_api_keys(self):
        keys = []
        raw_list = os.getenv("ELEVENLABS_API_KEYS") or os.getenv("ELEVEN_LABS_API_KEYS")
        if raw_list:
            keys.extend(k.strip() for k in raw_list.split(",") if k.strip())
        single = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_LABS_API_KEY")
        if single:
            if "," in single:
                keys.extend(k.strip() for k in single.split(",") if k.strip())
            else:
                keys.append(single.strip())
        seen = set()
        deduped = []
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _get_client(self):
        if not self._api_keys:
            if not self._client:
                self._client = ElevenLabs()
            return self._client
        api_key = random.choice(self._api_keys)
        client = self._clients.get(api_key)
        if not client:
            client = ElevenLabs(api_key=api_key)
            self._clients[api_key] = client
        return client

    def _run(self):
        self._ready.set()
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            text, done = item
            try:
                client = self._get_client()
                audio = client.text_to_speech.convert(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    output_format=self.output_format,
                )
                play(audio)
            except Exception as exc:
                print(f"[TTS] Error: {exc}")
            finally:
                done.set()

    def start(self):
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def speak(self, text: str):
        if not text:
            return
        if not self._thread:
            self.start()
        done = threading.Event()
        self._queue.put((text, done))
        done.wait()

    def close(self):
        if not self._thread:
            return
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=2)
        self._thread = None
        self._client = None
