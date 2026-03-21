import json
import os
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from google.oauth2 import service_account as google_service_account

from .base_client import BaseLLMClient
from .validators import validate_model


def _normalize_content(response):
    """Normalize Gemini 3 list content to string."""
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    return response


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output."""

    def invoke(self, input, config=None, **kwargs):
        return _normalize_content(super().invoke(input, config, **kwargs))


class NormalizedChatVertexAI(ChatVertexAI):
    """ChatVertexAI with normalized content output."""

    def invoke(self, input, config=None, **kwargs):
        return _normalize_content(super().invoke(input, config, **kwargs))


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def _use_vertex_ai(self) -> bool:
        """Check if Vertex AI should be used."""
        if self.kwargs.get("vertexai"):
            return True
        if os.environ.get("GOOGLE_VERTEX_SA_JSON"):
            return True
        return False

    def _resolve_secret(self, secret_id: str) -> str | None:
        """Resolve a secret via secret-resolver subprocess."""
        import subprocess
        try:
            request = json.dumps({"protocolVersion": 1, "ids": [secret_id]})
            result = subprocess.run(
                ["sudo", "-n", "-u", "credproxy", "/usr/bin/node",
                 "/opt/openclaw-security/secret-resolver.mjs"],
                input=request, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                payload = json.loads(result.stdout)
                return payload.get("values", {}).get(secret_id)
        except Exception:
            pass
        return None

    def _get_vertex_credentials(self):
        """Build Vertex AI credentials from service account JSON."""
        sa_json = (
            self.kwargs.get("google_service_account_json")
            or os.environ.get("GOOGLE_VERTEX_SA_JSON")
        )
        # Try resolving from secret-resolver if we have a secret ID but no SA JSON
        if not sa_json:
            secret_id = self.kwargs.get("google_service_account_secret_id")
            if secret_id:
                sa_json = self._resolve_secret(secret_id)
                if sa_json:
                    os.environ["GOOGLE_VERTEX_SA_JSON"] = sa_json

        if not sa_json:
            return None
        sa_info = json.loads(sa_json) if isinstance(sa_json, str) else sa_json
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        return google_service_account.Credentials.from_service_account_info(
            sa_info, scopes=scopes
        )

    def _apply_thinking(self, llm_kwargs: dict) -> None:
        """Apply thinking_level config to llm_kwargs in place."""
        thinking_level = self.kwargs.get("thinking_level")
        if not thinking_level:
            return
        model_lower = self.model.lower()
        if "gemini-3" in model_lower:
            if "pro" in model_lower and thinking_level == "minimal":
                thinking_level = "low"
            llm_kwargs["thinking_level"] = thinking_level
        else:
            llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0

    def get_llm(self) -> Any:
        """Return configured Google LLM instance (Vertex AI or AI Studio)."""
        llm_kwargs = {
            "model": self.model,
            "timeout": 120,
            "max_retries": 1,
        }

        for key in ("timeout", "max_retries", "google_api_key", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        self._apply_thinking(llm_kwargs)

        if self._use_vertex_ai():
            # Remove google_api_key — not used with Vertex AI
            llm_kwargs.pop("google_api_key", None)

            credentials = self._get_vertex_credentials()
            if credentials:
                llm_kwargs["credentials"] = credentials

            project = self.kwargs.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = self.kwargs.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            if project:
                llm_kwargs["project"] = project
            if location:
                llm_kwargs["location"] = location

            return NormalizedChatVertexAI(**llm_kwargs)

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
