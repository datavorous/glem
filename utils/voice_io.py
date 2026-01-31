from __future__ import annotations


class VoiceIO:
    def __init__(
        self,
        enabled: bool = False,
        rate: int | None = None,
        volume: float | None = None,
        mic_device_index: int | None = None,
        phrase_time_limit: int | None = None,
        debug: bool = False,
    ):
        self.enabled = enabled
        self.engine = None
        self.mic_device_index = mic_device_index
        self.phrase_time_limit = phrase_time_limit
        self.debug = debug
        if not enabled:
            return
        try:
            import pyttsx3
        except ImportError:
            self.enabled = False
            return
        self.engine = pyttsx3.init()
        if rate is not None:
            self.engine.setProperty("rate", rate)
        if volume is not None:
            self.engine.setProperty("volume", volume)

    def speak(self, text: str) -> None:
        if not self.enabled or not self.engine or not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self, prompt: str | None = None, fallback_to_text: bool = True) -> str:
        if prompt:
            print(prompt)
            self.speak(prompt)

        try:
            import speech_recognition as sr
        except ImportError:
            if self.debug:
                print("[voice] speech_recognition not installed; falling back to text input.")
            if fallback_to_text:
                return input("\nYou: ")
            return ""

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone(device_index=self.mic_device_index) as source:
                print("[+] Listening...")
                audio = recognizer.listen(source, phrase_time_limit=self.phrase_time_limit)
            text = recognizer.recognize_google(audio)
            print(f"\nYou: {text}")
            return text
        except Exception as exc:
            if self.debug:
                print(f"[voice] listen failed: {exc}")
            if fallback_to_text:
                return input("\nYou: ")
            return ""
