import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import requests

from src.ai.config import AIConfig


class LLMProviderError(Exception):
    """Raised when an LLM provider encounters an error."""
    pass


class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        """Generate response from messages."""
        pass


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for OpenAI and compatible REST APIs (OpenRouter, Groq, Ollama, etc.)."""

    def __init__(self, config: AIConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key or ""
        self.model = config.model
        self.timeout = config.timeout_seconds

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        if not self.config.is_configured:
            raise LLMProviderError(
                "AI Provider is not configured. Please supply an API key in environment variables, "
                "Streamlit secrets, or the sidebar configuration panel."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.strip()}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": max(0.0, min(temperature, 1.0)),
            "max_tokens": min(max_tokens, self.config.max_output_tokens),
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if response.status_code == 401:
                raise LLMProviderError("Authentication failed: Invalid API key provided.")
            if response.status_code == 429:
                raise LLMProviderError("Rate limit exceeded or insufficient quota on API provider.")
            if response.status_code != 200:
                raise LLMProviderError(
                    f"Provider returned error status {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMProviderError("No response choices returned by provider.")
            content = choices[0].get("message", {}).get("content", "")
            return content.strip()
        except requests.Timeout:
            raise LLMProviderError(
                f"Provider request timed out after {self.timeout} seconds. Please try again later."
            )
        except requests.RequestException as exc:
            raise LLMProviderError(f"Network error connecting to AI provider: {exc}")
        except json.JSONDecodeError:
            raise LLMProviderError("Failed to parse response JSON from AI provider.")


class MockProvider(BaseLLMProvider):
    """Deterministic mock provider for unit testing, offline demonstrations, and verification."""

    def __init__(self, mock_response: str = "This is a verified analytical response grounded in IPL data."):
        self.mock_response = mock_response
        self.call_history: List[List[Dict[str, str]]] = []

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        self.call_history.append(messages)
        # Check if the prompt has specific keywords to simulate grounded responses
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "").lower()
                break

        if "csk" in user_message or "chennai" in user_message:
            return (
                "Based on the verified match records, Chennai Super Kings have consistently maintained "
                "one of the highest win percentages in IPL history, particularly strong at home venues."
            )
        if "prediction" in user_message or "chase" in user_message:
            return (
                "The calibrated live-chase logistic regression model evaluates match state using balls remaining, "
                "wickets in hand, and required run rate. Required run rate and wickets remaining have the largest impact."
            )
        return self.mock_response


def get_llm_provider(config: AIConfig) -> BaseLLMProvider:
    """Factory to create provider instance based on config."""
    if config.provider == "mock":
        return MockProvider()
    return OpenAICompatibleProvider(config)
