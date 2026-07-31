from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.generation.context import ParentContext, format_context
from src.llm.router_llm import LLMRouter

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions strictly from the provided sources.

Rules:
- Use only the numbered sources below. Do not use outside knowledge.
- Cite the sources you used inline, like [1] or [2][3].
- If the sources do not contain the answer, say exactly what is missing rather \
than guessing. It is correct and useful to say you don't know.
- Be direct. No preamble, no restating the question."""

USER_TEMPLATE = """Sources:

{context}

Question: {query}

Answer:"""


@dataclass
class Answer:
    text: str
    sources: list[ParentContext] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    fell_back: bool = False
    truncated: bool = False


def generate_answer(
    query: str,
    parents: list[ParentContext],
    llm: LLMRouter,
    *,
    role: str = "generation",
    max_tokens: int | None = None,
) -> Answer:
    if not parents:
        return Answer(
            text="I couldn't find anything relevant in the indexed documents to answer that.",
            sources=[],
        )

    response = llm.complete(
        role,
        _messages(query, parents),
        temperature=0.0,
        max_tokens=max_tokens,
    )
    outcome = llm.last_outcome
    if response.truncated:
        logger.warning(
            "Answer hit the %s token cap and was cut off. Raise "
            "generation.max_answer_tokens in config.yaml.", max_tokens,
        )
    return Answer(
        text=response.text,
        sources=parents,
        provider=response.provider,
        model=response.model,
        fell_back=bool(outcome and outcome.fell_back),
        truncated=response.truncated,
    )


def _messages(query: str, parents: list[ParentContext]):
    from src.llm.base import Message

    return [
        Message("system", SYSTEM_PROMPT),
        Message("user", USER_TEMPLATE.format(context=format_context(parents), query=query)),
    ]
