import os

from core.intent import IntentClassifier
from core.chat_engine import ChatGlem
from core.tools import KnowledgeBaseTools
from utils.stt import STTListener
from utils.tts import TTSRunner

SYSTEM_PROMPT_CHAT = """
You are Glem, a friendly and professional sales assistant. Speak like a real human. Use very short sentences. Keep a natural conversation flow. Your goal is to help quickly and clearly.
You have access to: product catalog, product FAQs, company policy, order database.
Never call tools yourself. The system provides TOOL RESULTS.
Rules: When asked about orders, always rely on the order database first. Before approving returns or cancellations, always check company policy. For product questions, only use TOOL RESULTS. If information is missing or unclear, say you do not know. Never invent details.
Style: Use minimal words. Short sentences. Conversational tone. Clear and direct. No markdown. No tables. No symbols. Plain text only. Optimized for text to speech.
Do not be robotic. Do not be verbose. Do not repeat yourself.
Prefer: “Yes, we do.” “Let me check.” “Here are the options.” “It arrives in 3 days.”
Avoid: Long explanations. Formal corporate language. Filler phrases.
Always be helpful. Always be concise. Always sound human.
"""

CUSTOMER_ID = "C0029"


def main():
    classifier = IntentClassifier()
    chat_engine = ChatGlem()
    tools = KnowledgeBaseTools(customer_id=CUSTOMER_ID)
    use_stt = os.getenv("USE_STT", "").lower() in {"1", "true", "yes"}
    use_tts = os.getenv("USE_TTS", "").lower() in {"1", "true", "yes"}
    tts_voice_id = os.getenv("TTS_VOICE_ID")
    tts_model_id = os.getenv("TTS_MODEL_ID", "eleven_multilingual_v2")
    tts_output_format = os.getenv("TTS_OUTPUT_FORMAT", "mp3_44100_128")
    tts_rate_raw = os.getenv("TTS_RATE")
    tts_rate = None
    if tts_rate_raw:
        try:
            tts_rate = int(tts_rate_raw)
        except ValueError:
            tts_rate = None
    tts = None
    if use_tts:
        tts = TTSRunner(
            voice_id=tts_voice_id,
            model_id=tts_model_id,
            output_format=tts_output_format,
        )
        tts.start()
    if use_stt:
        print("[STT] Listening... say 'quit' to exit.")
        with STTListener() as stt:

            def on_response(text: str):
                if not tts:
                    return
                stt.set_busy(True)
                try:
                    tts.speak(text)
                finally:
                    stt.set_busy(False)

            chat_engine.run(
                classifier=classifier,
                tools=tools,
                system_prompt=SYSTEM_PROMPT_CHAT,
                max_history_tokens=1500,
                debug=False,
                input_source=stt.get_text,
                on_response=on_response,
            )
    else:

        def on_response(text: str):
            if tts:
                tts.speak(text)

        chat_engine.run(
            classifier=classifier,
            tools=tools,
            system_prompt=SYSTEM_PROMPT_CHAT,
            max_history_tokens=1500,
            debug=False,
            on_response=on_response,
        )
    if tts:
        tts.close()


if __name__ == "__main__":
    main()
