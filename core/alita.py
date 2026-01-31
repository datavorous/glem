import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
from groq import Groq


class AlitaEngine:
    def __init__(self, env_key_name: str = "API_KEYS"):
        load_dotenv()
        keys = os.getenv(env_key_name, "").split(",")
        self.api_keys = [k.strip() for k in keys if k.strip()]
        if not self.api_keys:
            raise ValueError(f"No API keys found in: {env_key_name}")
        self.current_key_index = 0

    def _get_client(self) -> Groq:
        return Groq(api_key=self.api_keys[self.current_key_index])

    def _rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

    def _should_rotate_key(self, error: Exception) -> bool:
        message = str(error).lower()
        keywords = (
            "rate limit",
            "too many requests",
            "quota",
            "429",
            "unauthorized",
            "authentication",
            "auth",
            "invalid api key",
            "api key",
            "permission",
            "401",
        )
        return any(word in message for word in keywords)

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.3,
        max_retries: int = 2,
        model: str = "openai/gpt-oss-20b",
        schema_name: str = "response_schema",
    ) -> Dict[str, Any]:
        total_attempts = len(self.api_keys) * max_retries

        if "type" not in schema:
            schema["type"] = "object"
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        last_error = None
        for attempt in range(total_attempts):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty response from API")
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                last_error = e
                print(
                    f"[Attempt {attempt + 1}/{total_attempts}] Error: {type(e).__name__}: {e}"
                )
                if self._should_rotate_key(e):
                    self._rotate_key()

        return {
            "error": f"Generation failed after {total_attempts} attempts. Last error: {last_error}"
        }
