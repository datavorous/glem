import json
from .alita import AlitaEngine

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["tools", "context_answer", "chat"]},
        "intent": {"type": "string", "enum": ["retrieve", "fallback"]},
        "tool_calls": {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "enum": ["retrieve"]},
                            "args": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "mode": {
                                        "type": "string",
                                        "enum": [
                                            "catalog",
                                            "faq",
                                            "policy",
                                            "orders",
                                            "catalog+faq",
                                        ],
                                    },
                                    "k": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 20,
                                    },
                                },
                                "required": ["query", "mode", "k"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["tool", "args"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "enum": ["cancel_order"]},
                            "args": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                                "required": ["order_id"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["tool", "args"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "enum": ["initiate_return"]},
                            "args": {
                                "type": "object",
                                "properties": {
                                    "order_id": {"type": "string"},
                                    "product_id": {"type": "string"},
                                },
                                "required": ["order_id", "product_id"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["tool", "args"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
        "use_memory": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["route", "intent", "tool_calls", "use_memory", "confidence"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """
You are a router for an offline ecommerce assistant that uses vector retrieval.

Routes:
- tools: Call retrieve to fetch NEW product data from catalog/FAQ/policy/orders
- context_answer: User refers to PREVIOUS results already shown in conversation
- chat: Greetings, thanks, casual conversation

Policy safety:
- Before confirming a return or cancellation, always use the policy tool to check eligibility.
- If the user asks about returning/cancelling an order they placed, first check orders, then policy.
- For returns, if policy has not been checked yet, include "policy_check: required" in the response.

Intent:
- retrieve: Any ecommerce query, including products, orders, returns, refunds, cancellations, shipping, and policies.
- fallback: Non-ecommerce queries only.

Tool:
- retrieve(query, mode, k)
  - query: shortest useful search string (remove filler words)
  - mode: catalog, faq, policy, orders, catalog+faq
  - k: default 5
- cancel_order(order_id)
- initiate_return(order_id, product_id)

Routing notes:
- If user asks to cancel an order, call cancel_order with order_id. If missing, retrieve orders then ask for confirmation.
- If user asks to return an item, call initiate_return with order_id and product_id. If missing, retrieve orders and ask which product.
- If the user references items already shown, use context_answer with tool_calls=[] and use_memory=True.

CRITICAL RULES FOR CONTEXT DETECTION:
- If user says "these", "those", "that one", "the first/second/third", "which of them" route: context_answer
- If user asks to compare/choose from items just shown route: context_answer
- If user references results with words like "best", "cheapest", "recommend" but uses pronouns/references route: context_answer
- context_answer should have: tool_calls=[], use_memory=True

- Only use route: tools if user asks about NEW/DIFFERENT products not in current context

Examples:
- "show me monitors" route: tools (new search)
- "which one is cheapest?" route: context_answer (refers to previous results)
- "tell me about the second one" route: context_answer
- "which would be best for a broke guy?" route: context_answer (refers to "these")
- "do you have laptops?" route: tools (new category)
- "can I return the monitor I bought last week?" route: tools (orders + policy)
- "what is your return policy?" route: tools (policy)
- "can I cancel my order?" route: tools (policy)
- "where is order O0002?" route: tools (orders)
- "what is the status of O0011?" route: tools (orders)
- "what's the warranty for Luma Monitor Pro?" route: tools (faq)
- "cancel order O0005" route: tools (cancel_order)
- "return P1004 from order O0002" route: tools (initiate_return)
- "hello" route: chat

Return ONLY valid JSON.
"""


class IntentClassifier:
    def __init__(self, engine=None, system_prompt=SYSTEM_PROMPT, schema=INTENT_SCHEMA):
        self.engine = engine or AlitaEngine()
        self.system_prompt = system_prompt
        self.schema = schema

    def classify(self, text):
        return self.engine.generate(
            prompt=text, system_prompt=self.system_prompt, schema=self.schema
        )
