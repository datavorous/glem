import re

from .alita import AlitaEngine


def build_sliding_window(history, use_memory: bool, max_tokens: int):
    if not history:
        return []

    system_msg = history[0:1]
    if not use_memory:
        latest_user = next(
            (msg for msg in reversed(history) if msg.get("role") == "user"), None
        )
        return system_msg + ([latest_user] if latest_user else [])

    window = list(history)
    total_tokens = sum(len(m.get("content", "")) // 4 for m in window)
    idx = 1
    while total_tokens > max_tokens and idx < len(window) - 1:
        total_tokens -= len(window[idx].get("content", "")) // 4
        idx += 1
    return system_msg + window[idx:]


def _normalize_tools(tools):
    if tools is None:
        return {}
    if isinstance(tools, dict):
        return tools
    registry = {}
    for name in dir(tools):
        if name.startswith("_"):
            continue
        value = getattr(tools, name)
        if callable(value):
            registry[name] = value
    return registry


def _execute_tool_calls(tool_calls, tools, default_query, debug=False, tools_obj=None):
    outputs = []
    for call in tool_calls:
        tool_name = call.get("tool")
        args = dict(call.get("args") or {})
        if tool_name == "retrieve":
            if "query" not in args:
                args["query"] = default_query
            mode = (args.get("mode") or "").lower()
            if mode == "orders":
                order_id = _extract_order_id(str(args.get("query", "")))
                customer_id = _extract_customer_id(str(args.get("query", "")))
                if not order_id and not customer_id:
                    default_customer = getattr(tools_obj, "customer_id", None)
                    if default_customer:
                        args["query"] = default_customer
            k_value = args.get("k", 5)
            try:
                k_value = int(k_value)
            except (TypeError, ValueError):
                k_value = 5
            if k_value < 1 or k_value > 20:
                k_value = 5
            args["k"] = k_value
        elif tool_name == "cancel_order":
            if "order_id" not in args:
                order_id = _extract_order_id(default_query)
                if order_id:
                    args["order_id"] = order_id
        elif tool_name == "initiate_return":
            if "order_id" not in args:
                order_id = _extract_order_id(default_query)
                if order_id:
                    args["order_id"] = order_id
            if "product_id" not in args:
                product_id = _extract_product_id(default_query)
                if product_id:
                    args["product_id"] = product_id

        tool_fn = tools.get(tool_name)
        if not tool_fn:
            outputs.append(f"[Missing tool '{tool_name}']")
            continue

        if debug:
            print("[LOG] tool_call =", {"tool": tool_name, "args": args})

        try:
            result = tool_fn(**args)
        except Exception as exc:
            outputs.append(f"[Tool '{tool_name}' error: {exc}]")
            continue

        outputs.append(
            f"[{tool_name} results]:\n{result}"
        )
    return "\n".join(outputs).strip()


def _needs_policy_check(text: str) -> bool:
    if not text:
        return False
    text = text.lower()
    keywords = (
        "return",
        "refund",
        "exchange",
        "cancel",
        "cancellation",
        "return policy",
    )
    return any(word in text for word in keywords)


def _needs_return_warning(text: str) -> bool:
    if not text:
        return False
    text = text.lower()
    keywords = ("return", "refund", "exchange")
    return any(word in text for word in keywords)


def _needs_order_lookup(text: str) -> bool:
    if not text:
        return False
    text = text.lower()
    if re.search(r"\bo\d{4}\b", text):
        return True
    if re.search(r"\bc\d{4}\b", text):
        return True
    keywords = (
        "order",
        "bought",
        "purchased",
        "shipment",
        "delivery",
        "where is my",
    )
    return any(word in text for word in keywords)


def _is_customer_id_question(text: str) -> bool:
    if not text:
        return False
    text = text.lower()
    return "customer id" in text or "customerid" in text


def _extract_order_id(text: str):
    if not text:
        return None
    match = re.search(r"\bo\d{4}\b", text.lower())
    if match:
        return match.group(0).upper()
    return None


def _extract_product_id(text: str):
    if not text:
        return None
    match = re.search(r"\bp\d{4}\b", text.lower())
    if match:
        return match.group(0).upper()
    return None


def _extract_customer_id(text: str):
    if not text:
        return None
    match = re.search(r"\bc\d{4}\b", text.lower())
    if match:
        return match.group(0).upper()
    return None


def _get_catalog_matches(tools, query: str, k: int = 5):
    if tools is None:
        return None
    if hasattr(tools, "catalog") and hasattr(tools.catalog, "search"):
        return tools.catalog.search(query=query, k=k)
    if hasattr(tools, "search"):
        return tools.search(query=query, k=k)
    return None


def _get_order_matches(tools, query: str, k: int = 5):
    if tools is None:
        return None
    if hasattr(tools, "orders") and hasattr(tools.orders, "search"):
        return tools.orders.search(query=query, k=k)
    return None


def _get_retrieve_query(tool_calls, modes):
    for call in tool_calls or []:
        if call.get("tool") != "retrieve":
            continue
        args = call.get("args") or {}
        mode = (args.get("mode") or "").lower()
        if mode in modes:
            return args.get("query")
    return None


def _resolve_generic_order_query(tools, query: str | None):
    if not query:
        return query
    normalized = query.strip().lower()
    if normalized in {"orders", "order", "my orders", "order history", "previous orders"}:
        return getattr(tools, "customer_id", None) or query
    if not _extract_order_id(query) and not _extract_customer_id(query):
        if "order" in normalized or "orders" in normalized:
            return getattr(tools, "customer_id", None) or query
    return query


def _suggest_categories(tools, limit: int = 4):
    if not tools or not hasattr(tools, "catalog"):
        return []
    products = getattr(tools.catalog, "products", [])
    seen = []
    for product in products:
        category = product.get("category")
        if category and category not in seen:
            seen.append(category)
        if len(seen) >= limit:
            break
    return seen


def _ensure_order_tool_call(plan, user_input):
    if not _needs_order_lookup(user_input):
        return plan

    tool_calls = list(plan.get("tool_calls", []))
    has_orders = any(
        call.get("tool") == "retrieve"
        and (call.get("args") or {}).get("mode") == "orders"
        for call in tool_calls
    )
    if has_orders:
        return plan

    customer_id = _extract_customer_id(user_input)
    query = customer_id or user_input
    tool_calls.append(
        {
            "tool": "retrieve",
            "args": {"query": query, "mode": "orders", "k": 5},
        }
    )

    updated = dict(plan)
    updated["route"] = "tools"
    updated["tool_calls"] = tool_calls
    return updated


def _ensure_policy_tool_call(plan, user_input):
    if not _needs_policy_check(user_input):
        return plan

    tool_calls = list(plan.get("tool_calls", []))
    has_policy = any(
        call.get("tool") == "retrieve"
        and (call.get("args") or {}).get("mode") == "policy"
        for call in tool_calls
    )
    if has_policy:
        return plan

    tool_calls.append(
        {
            "tool": "retrieve",
            "args": {"query": user_input, "mode": "policy", "k": 5},
        }
    )

    updated = dict(plan)
    updated["route"] = "tools"
    updated["tool_calls"] = tool_calls
    return updated


class ChatAlita(AlitaEngine):
    def chat(self, messages, model="openai/gpt-oss-20b", temperature=0.7):
        total_attempts = len(self.api_keys) * 2

        for attempt in range(total_attempts):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature
                )
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty response")

                return response.choices[0].message.content

            except Exception as e:
                if self._should_rotate_key(e):
                    print(f"Chat Error: {e}. Rotating key...")
                    self._rotate_key()
                else:
                    print(f"Chat Error: {e}.")

        return "I'm having trouble connecting to my brain right now. Please try again."

    def run(
        self,
        classifier,
        tools,
        system_prompt,
        max_history_tokens=1500,
        debug=True,
        model="openai/gpt-oss-20b",
    ):
        history = [{"role": "system", "content": system_prompt}]
        tool_registry = _normalize_tools(tools)

        if debug:
            print("[+] Debug logging: ON")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                break

            if _is_customer_id_question(user_input) and getattr(tools, "customer_id", None):
                response_text = f"Your customer ID is {tools.customer_id}."
                print(f"Alita: {response_text}")
                history.append({"role": "assistant", "content": response_text})
                continue

            plan = classifier.classify(user_input)
            if debug:
                print("\n[LOG] intent_plan =", plan)

            if "error" in plan:
                if debug:
                    print("[LOG] intent_error =", plan["error"])
                plan = {
                    "route": "chat",
                    "intent": "fallback",
                    "tool_calls": [],
                    "use_memory": False,
                    "confidence": 0.0,
                }

            plan = _ensure_order_tool_call(plan, user_input)
            plan = _ensure_policy_tool_call(plan, user_input)
            if plan.get("route") in {"tools", "context_answer"}:
                plan["intent"] = "retrieve"

            tool_output = ""
            if plan.get("route") == "tools" and plan.get("tool_calls"):
                print("[+] Searching...")
                tool_output = _execute_tool_calls(
                    tool_calls=plan.get("tool_calls", []),
                    tools=tool_registry,
                    default_query=user_input,
                    debug=debug,
                    tools_obj=tools,
                )
                if debug:
                    print("[LOG] tool_output =", tool_output.strip() or "<empty>")

                followup_prompt = (
                    "User:\n"
                    f"{user_input}\n\n"
                    "Tool results:\n"
                    f"{tool_output}\n\n"
                    "Decide if more tool calls are needed."
                )
                followup_plan = classifier.classify(followup_prompt)
                if debug:
                    print("[LOG] followup_plan =", followup_plan)
                if "error" not in followup_plan and followup_plan.get("tool_calls"):
                    print("[+] Searching more...")
                    followup_output = _execute_tool_calls(
                        tool_calls=followup_plan.get("tool_calls", []),
                        tools=tool_registry,
                        default_query=user_input,
                        debug=debug,
                        tools_obj=tools,
                    )
                    if followup_output:
                        tool_output = "\n".join(
                            part for part in [tool_output, followup_output] if part
                        )

            catalog_query = _get_retrieve_query(
                plan.get("tool_calls", []), {"catalog", "catalog+faq"}
            )
            order_query = _get_retrieve_query(plan.get("tool_calls", []), {"orders"})
            order_query = _resolve_generic_order_query(tools, order_query)
            catalog_matches = (
                _get_catalog_matches(tools, catalog_query, k=5) or []
                if catalog_query
                else []
            )
            order_matches = (
                _get_order_matches(tools, order_query, k=5) or []
                if order_query
                else []
            )
            order_id = _extract_order_id(user_input)

            if plan.get("route") == "tools" and catalog_query and catalog_matches:
                if len(catalog_matches) > 1:
                    names = []
                    for item in catalog_matches:
                        name = item.get("product_name", "Unknown")
                        if name not in names:
                            names.append(name)
                    if len(names) > 1:
                        user_text = user_input.lower()
                        if not any(name.lower() in user_text for name in names):
                            options = ", ".join(names[:2])
                            response_text = f"Do you mean {options}?"
                            print(f"Alita: {response_text}")
                            history.append({"role": "assistant", "content": response_text})
                            continue

            if plan.get("route") == "tools" and catalog_query and not catalog_matches:
                suggestions = _suggest_categories(tools)
                if suggestions:
                    response_text = (
                        "I couldn't find a matching product. Are you looking for "
                        + ", ".join(suggestions)
                        + "?"
                    )
                else:
                    response_text = (
                        "I couldn't find a matching product. Could you clarify the "
                        "product name or category?"
                    )
                print(f"Alita: {response_text}")
                history.append({"role": "assistant", "content": response_text})
                continue

            if plan.get("route") == "tools" and order_query and not order_matches:
                if order_id:
                    response_text = (
                        "I couldn't find that order ID. Please double-check the "
                        "digits or share your customer ID."
                    )
                else:
                    response_text = (
                        "I couldn't find an order. Could you share the order ID or "
                        "customer ID?"
                    )
                print(f"Alita: {response_text}")
                history.append({"role": "assistant", "content": response_text})
                continue

            history.append({"role": "user", "content": user_input})

            use_memory = bool(plan.get("use_memory", True))
            current_turn_messages = build_sliding_window(
                history=history,
                use_memory=use_memory,
                max_tokens=max_history_tokens,
            )
            if debug:
                print("[LOG] use_memory =", use_memory)
                print("[LOG] window_size =", len(current_turn_messages))

            if tool_output:
                current_turn_messages.append(
                    {
                        "role": "assistant",
                        "content": f"TOOL RESULTS:\n{tool_output}",
                    }
                )
                if debug:
                    print("[LOG] injected_tool_data = True")
            else:
                if debug:
                    print("[LOG] injected_tool_data = False")

            response_text = self.chat(
                messages=current_turn_messages,
                model=model,
            )

            if _needs_return_warning(user_input):
                non_returnable = next(
                    (
                        item
                        for item in catalog_matches
                        if item.get("return_eligible") is False
                    ),
                    None,
                )
                if non_returnable and "non-returnable" not in response_text.lower():
                    response_text = (
                        f"{response_text}\nNote: {non_returnable.get('product_name', 'This item')} "
                        "is marked non-returnable."
                    )

            print(f"Alita: {response_text}")
            history.append({"role": "assistant", "content": response_text})
