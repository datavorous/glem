from RealtimeSTT import AudioToTextRecorder
import queue
import time


class STTListener:
    def __init__(self):
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._recorder = None
        self._busy = False
        self._last_text = ""
        self._last_ts = 0.0

    def _on_text(self, text: str) -> None:
        if self._busy:
            return
        text = (text or "").strip()
        if text:
            now = time.time()
            if text == self._last_text and (now - self._last_ts) < 2.0:
                return
            self._last_text = text
            self._last_ts = now
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put(text)

    def __enter__(self):
        self._recorder = AudioToTextRecorder()
        self._recorder.__enter__()
        self._recorder.text(self._on_text)
        time.sleep(1.25)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._recorder:
            return self._recorder.__exit__(exc_type, exc, tb)
        return False

    def get_text(self) -> str:
        if self._recorder:
            self._recorder.text(self._on_text)
        text = self._queue.get()
        print(f"[STT] {text}")
        return text

    def set_busy(self, value: bool) -> None:
        self._busy = value


def handle_command_blocking(text: str) -> None:
    print(f"[HANDLER] {text}")
    time.sleep(2)


def main():
    q: "queue.Queue[str]" = queue.Queue()
    busy = False

    def on_text(text: str):
        nonlocal busy
        if busy:
            return
        text = (text or "").strip()
        if text:
            q.put(text)

    with AudioToTextRecorder() as recorder:
        recorder.text(on_text)
        while True:
            text = q.get()
            busy = True
            print(f"[YOU] {text}")
            handle_command_blocking(text)
            busy = False
            print("[READY]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
