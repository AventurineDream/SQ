import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar
import requests
from pydantic import BaseModel, ValidationError

from schema import (
    ActorTurnOutput,
    DiaryGenerationOutput,
    FailureCode,
    ModelCallDiagnostic,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.MULTILINE)

def sanitize_model_text(raw_text: str) -> str:
    """Strips <think> reasoning traces and markdown code fences before parsing."""
    if not raw_text:
        return ""
    cleaned = THINK_TAG_RE.sub("", raw_text)
    cleaned = CODE_FENCE_RE.sub("", cleaned)
    return cleaned.strip()

class LLMAdapter(ABC):
    """Base interface for all model inference endpoints."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_structured(
        self,
        prompt_payload: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: int = 350,
        max_retries: int = 3,
        validator_fn: Optional[Callable[[T], Tuple[bool, Optional[str], FailureCode]]] = None,
        diagnostic_callback: Optional[Callable[[ModelCallDiagnostic], None]] = None,
        call_meta: Optional[Dict] = None,
    ) -> Optional[T]:
        """Generic self-healing structured output generator with retry-with-feedback."""
        messages = [dict(m) for m in prompt_payload]
        meta = call_meta or {}

        for attempt in range(1, max_retries + 1):
            start_time = time.monotonic()
            raw_output = ""
            try:
                raw_output = self._call_raw(
                    messages=messages,
                    json_schema=response_model.model_json_schema(),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = int((time.monotonic() - start_time) * 1000)
                cleaned = sanitize_model_text(raw_output)
                parsed_dict = json.loads(cleaned)
                validated_obj = response_model(**parsed_dict)

                if validator_fn is not None:
                    is_valid, rejection_reason, fail_code = validator_fn(validated_obj)
                    if not is_valid:
                        self._record_diag(
                            diagnostic_callback, meta, attempt, False,
                            fail_code, rejection_reason, latency_ms
                        )
                        logger.warning(
                            f"[{self.model_name}] Attempt {attempt}/{max_retries} rejected by boundary gate: {rejection_reason}"
                        )
                        messages.append({"role": "assistant", "content": cleaned})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"REJECTED: {rejection_reason}\n"
                                "Rewrite your response to fix this violation and output valid JSON."
                            ),
                        })
                        continue

                self._record_diag(
                    diagnostic_callback, meta, attempt, True,
                    FailureCode.NONE, None, latency_ms
                )
                return validated_obj

            except (json.JSONDecodeError, ValidationError) as err:
                latency_ms = int((time.monotonic() - start_time) * 1000)
                err_msg = str(err)
                self._record_diag(
                    diagnostic_callback, meta, attempt, False,
                    FailureCode.FORMAT_VIOLATION, err_msg, latency_ms
                )
                logger.warning(f"[{self.model_name}] Attempt {attempt}/{max_retries} schema error: {err_msg}")
                messages.append({"role": "assistant", "content": raw_output or "{}"})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous output failed JSON schema validation:\n{err_msg}\n"
                        "Please return ONLY valid JSON matching the required schema."
                    ),
                })
            except Exception as err:
                latency_ms = int((time.monotonic() - start_time) * 1000)
                self._record_diag(
                    diagnostic_callback, meta, attempt, False,
                    FailureCode.FORMAT_VIOLATION, str(err), latency_ms
                )
                logger.error(f"[{self.model_name}] Transport error on attempt {attempt}: {err}")

        return None

    def _record_diag(
        self,
        callback: Optional[Callable[[ModelCallDiagnostic], None]],
        meta: Dict,
        attempt: int,
        accepted: bool,
        failure_code: FailureCode,
        error_detail: Optional[str],
        latency_ms: int,
    ) -> None:
        if callback is None:
            return
        diag = ModelCallDiagnostic(
            season_id=meta.get("season_id", "unknown"),
            day=meta.get("day", 1),
            slot=meta.get("slot", 1),
            turn_number=meta.get("turn_number", 0),
            actor_id=meta.get("actor_id", self.model_name),
            model_name=self.model_name,
            call_type=meta.get("call_type", "generic"),
            attempt=attempt,
            accepted=accepted,
            failure_code=failure_code,
            error_detail=error_detail,
            latency_ms=latency_ms,
        )
        callback(diag)

    @abstractmethod
    def _call_raw(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 350,
    ) -> str:
        raise NotImplementedError

    def generate_turn(self, prompt_payload: List[Dict[str, str]]) -> Optional[ActorTurnOutput]:
        return self.generate_structured(prompt_payload, ActorTurnOutput, temperature=0.7, max_tokens=300)

    def generate_diary(self, prompt_payload: List[Dict[str, str]]) -> Optional[str]:
        res = self.generate_structured(prompt_payload, DiaryGenerationOutput, temperature=0.9, max_tokens=180)
        return res.content if res else None


class OllamaAdapter(LLMAdapter):
    """Inference adapter for local Ollama instances with native JSON schema enforcement."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://10.0.0.103:11434",
        think_mode: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.base_url = base_url
        if think_mode is None and "gpt-oss" in model_name.lower():
            self.think_mode = "low"
        else:
            self.think_mode = think_mode

    def _call_raw(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 350,
    ) -> str:
        data = {
            "model": self.model_name,
            "messages": messages,
            "format": json_schema if json_schema is not None else "json",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.think_mode is not None:
            data["think"] = self.think_mode

        response = requests.post(f"{self.base_url}/api/chat", json=data, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]