import os

from core.intent import IntentClassifier
from core.chat_engine import ChatAlita
from core.tools import KnowledgeBaseTools

SYSTEM_PROMPT_CHAT = """
You are a professional, concise sales assistant named Alita.
You have access to a product catalog, the product faqs, company policy, and order database.
Do not call tools directly. The system will provide TOOL RESULTS when data is retrieved.
When the user asks about orders, always check the order database first.
Before confirming returns or cancellations, always check the company policy to ensure eligibility.
When the user asks about products, use the TOOL RESULTS provided to answer their questions accurately.
If the data is missing or doesn't match, admit you don't know; do not hallunicate products.
Keep answers concise, formal and helpful.
Use as less words as possible, while being clear and informative.
"""

CUSTOMER_ID = "C0029"
VOICE_ENABLED = os.getenv("ALITA_VOICE", "0") == "1"
VOICE_INPUT = os.getenv("ALITA_VOICE_INPUT", "0") == "1"
VOICE_DEBUG = os.getenv("ALITA_VOICE_DEBUG", "0") == "1"
VOICE_MIC_INDEX = os.getenv("ALITA_VOICE_MIC_INDEX")
VOICE_PHRASE_LIMIT = os.getenv("ALITA_VOICE_PHRASE_LIMIT")


def main():
    classifier = IntentClassifier()
    chat_engine = ChatAlita()
    tools = KnowledgeBaseTools(customer_id=CUSTOMER_ID)
    chat_engine.run(
        classifier=classifier,
        tools=tools,
        system_prompt=SYSTEM_PROMPT_CHAT,
        max_history_tokens=1500,
        debug=False,
        voice=VOICE_ENABLED,
        voice_input=VOICE_INPUT,
        voice_debug=VOICE_DEBUG,
        voice_mic_index=int(VOICE_MIC_INDEX) if VOICE_MIC_INDEX else None,
        voice_phrase_limit=int(VOICE_PHRASE_LIMIT) if VOICE_PHRASE_LIMIT else None,
    )


if __name__ == "__main__":
    main()
