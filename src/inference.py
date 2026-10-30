import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import requests
import os

from schema import ActorTurnOutput

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAdapter(ABC):
    """Base interface for all model inference endpoints."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_turn(self, prompt_payload: List[Dict[str, str]]) -> Optional[ActorTurnOutput]:
        """Sends the context to the model and returns a validated Pydantic object."""
        pass
        
    @abstractmethod
    def generate_diary(self, prompt_payload: List[Dict[str, str]]) -> Optional[str]:
        """Generates a plain-text, tweet-length diary entry."""
        pass


class OllamaAdapter(LLMAdapter):
    """
    Inference adapter for local Ollama instances.
    Expects Ollama to be running on the default localhost:11434.
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        think_mode: Optional[str] = None, # <-- ADD THIS LINE
    ):
        super().__init__(model_name)
        self.base_url = base_url
        # Automatically default gpt-oss models to think="low" per Oct 2026 study
        if think_mode is None and "gpt-oss" in model_name.lower():
            self.think_mode = "low"
        else:
            self.think_mode = think_mode

    def _inject_json_instructions(self, prompt_payload: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Appends strict JSON schema instructions to the system prompt."""
        # Deep copy to avoid mutating the original prompt list
        payload_copy = [dict(msg) for msg in prompt_payload]
        
        schema_instruction = (
            "\n\nCRITICAL INSTRUCTION: You MUST respond in perfectly valid JSON. "
            "Your output must exactly match this JSON schema:\n"
            "{\n"
            '  "internal_thought": "Your internal monologue",\n'
            '  "action_type": "dialogue" OR "action",\n'
            '  "public_content": "What you say or do",\n'
            '  "target_actors": ["List", "of", "names", "or", "all"]\n'
            "}"
        )
        
        for msg in payload_copy:
            if msg["role"] == "system":
                msg["content"] += schema_instruction
                break
                
        return payload_copy

    def generate_turn(self, prompt_payload: List[Dict[str, str]]) -> Optional[ActorTurnOutput]:
        """Requests a JSON-formatted action from the Ollama model."""
        
        structured_payload = self._inject_json_instructions(prompt_payload)
        
        data = {
            "model": self.model_name,
            "messages": structured_payload,
            "format": "json", # Ollama's native flag to force JSON output
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 300 # Capping token output to prevent rambling
            }
        }

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=data, timeout=120)
            response.raise_for_status()
            
            raw_content = response.json()["message"]["content"]
            
            # 1. Parse string to dict
            parsed_json = json.loads(raw_content)
            
            # 2. Validate dict through Pydantic
            validated_output = ActorTurnOutput(**parsed_json)
            return validated_output
            
        except json.JSONDecodeError as e:
            logger.error(f"[{self.model_name}] Failed to parse JSON: {e}\nRaw Output: {raw_content}")
            return None
        except Exception as e:
            logger.error(f"[{self.model_name}] Inference request failed: {e}")
            return None

    def generate_diary(self, prompt_payload: List[Dict[str, str]]) -> Optional[str]:
        """Requests a creative, plain-text diary entry."""
        
        data = {
            "model": self.model_name,
            "messages": prompt_payload,
            "stream": False,
            "options": {
                "temperature": 0.9, # Higher temperature for creative variance
                "num_predict": 100  # Strict cap for tweet-length outputs
            }
        }

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=data, timeout=120)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"[{self.model_name}] Diary generation failed: {e}")
            return None

    class AnthropicAdapter(LLMAdapter):
        """Frontier judge adapter for Claude Sonnet via the Anthropic Messages API."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = "https://api.anthropic.com/v1/messages"

    def _call_raw(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> str:
        if not self.api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY environment variable.")

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat_messages = [m for m in messages if m["role"] != "system"]

        if json_schema:
            system_parts.append(
                f"Respond strictly with valid JSON matching this JSON Schema:\n{json.dumps(json_schema)}"
            )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": "\n\n".join(system_parts),
            "messages": chat_messages,
        }
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


class GeminiAdapter(LLMAdapter):
    """Frontier judge adapter for Gemini Pro via the Google Generative Language REST API."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def _call_raw(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> str:
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable.")

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        if json_schema:
            system_parts.append(
                f"Respond strictly with valid JSON matching this JSON Schema:\n{json.dumps(json_schema)}"
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": "\n\n".join(system_parts)}]},
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]