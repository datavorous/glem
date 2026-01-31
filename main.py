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


def main():
    classifier = IntentClassifier()
    chat_engine = ChatAlita()
    tools = KnowledgeBaseTools(customer_id=CUSTOMER_ID)
    chat_engine.run(
        classifier=classifier,
        tools=tools,
        system_prompt=SYSTEM_PROMPT_CHAT,
        max_history_tokens=1500,
        debug=True,
    )


if __name__ == "__main__":
    main()
