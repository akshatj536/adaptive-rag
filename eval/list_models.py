#!/usr/bin/env python3
"""List the chat models each API key can actually reach.

Never trust docs or a model name from memory for this: Groq retired the whole
Llama family and Gemini retired 2.5 mid-project, and both were only caught by
asking the live API.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import get_config  # noqa: E402  (loads .env)

import httpx  # noqa: E402

get_config()

# provider -> (env var candidates, models endpoint)
OPENAI_COMPAT = {
    "groq":       (["GROQ_API_KEY"],                          "https://api.groq.com/openai/v1/models"),
    "cerebras":   (["CEREBRAS_API_KEY"],                      "https://api.cerebras.ai/v1/models"),
    "mistral":    (["MISTRAL_API_KEY"],                       "https://api.mistral.ai/v1/models"),
    "nvidia_nim": (["NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"],  "https://integrate.api.nvidia.com/v1/models"),
}

# Substrings that mark a model as not a text chat model.
NON_CHAT = ("whisper", "embed", "rerank", "tts", "guard", "moderation", "ocr",
            "vision", "image", "video", "audio", "riva", "speech", "orpheus",
            "diffusion", "flux", "sana", "clip", "nemoretriever", "parakeet")


def key_for(names: list[str]) -> str | None:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return None


def is_chat(name: str) -> bool:
    return not any(s in name.lower() for s in NON_CHAT)


def list_openai_compat(provider: str) -> list[str] | str:
    envs, url = OPENAI_COMPAT[provider]
    key = key_for(envs)
    if not key:
        return f"no key ({' or '.join(envs)})"
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {type(exc).__name__}: {str(exc)[:90]}"


def list_cohere() -> list[str] | str:
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        return "no key (COHERE_API_KEY)"
    try:
        r = httpx.get("https://api.cohere.com/v1/models",
                      headers={"Authorization": f"Bearer {key}"},
                      params={"page_size": 100}, timeout=30)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", [])
                      if "chat" in (m.get("endpoints") or []))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {type(exc).__name__}: {str(exc)[:90]}"


def list_gemini() -> list[str] | str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return "no key (GOOGLE_API_KEY)"
    try:
        from google import genai
        c = genai.Client(api_key=key)
        out = []
        for m in c.models.list():
            acts = getattr(m, "supported_actions", None) or []
            if "generateContent" in acts or not acts:
                out.append(m.name.replace("models/", ""))
        return sorted(out)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {type(exc).__name__}: {str(exc)[:90]}"


def main() -> int:
    results = {p: list_openai_compat(p) for p in OPENAI_COMPAT}
    results["cohere"] = list_cohere()
    results["gemini"] = list_gemini()

    for provider, models in results.items():
        if isinstance(models, str):
            print(f"\n=== {provider}: {models}")
            continue
        chat = [m for m in models if is_chat(m)]
        print(f"\n=== {provider}: {len(models)} models, {len(chat)} chat-capable")
        for m in chat:
            print(f"     {m}")

    (ROOT / "eval" / "available_models.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
