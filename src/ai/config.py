import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AIConfig:
    provider: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 30
    max_output_tokens: int = 800
    enable_analytics_queries: bool = True

    @property
    def is_configured(self) -> bool:
        """Returns True if an API key is available or if using mock provider."""
        if self.provider == "mock":
            return True
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    @property
    def masked_key(self) -> str:
        """Safe representation of the API key for diagnostics, never exposing secrets."""
        if not self.api_key:
            return "Not configured"
        clean = self.api_key.strip()
        if len(clean) <= 8:
            return "********"
        return f"{clean[:3]}...{clean[-4:]}"


def load_ai_config(session_api_key: Optional[str] = None) -> AIConfig:
    """Load configuration from environment, Streamlit secrets, or session state.
    
    Session API key takes precedence if provided by user in the UI.
    """
    api_key = session_api_key or os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    provider = os.getenv("AI_PROVIDER", "openai").lower()
    model = os.getenv("AI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("AI_BASE_URL", "https://api.openai.com/v1")

    # Check Streamlit secrets safely if available
    try:
        import streamlit as st
        if not api_key and hasattr(st, "secrets") and "AI_API_KEY" in st.secrets:
            api_key = str(st.secrets["AI_API_KEY"])
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets and not api_key:
            api_key = str(st.secrets["OPENAI_API_KEY"])
    except Exception:
        pass

    try:
        timeout = int(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError:
        timeout = 30

    try:
        max_tokens = int(os.getenv("AI_MAX_OUTPUT_TOKENS", "800"))
    except ValueError:
        max_tokens = 800

    return AIConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout,
        max_output_tokens=max_tokens,
    )
