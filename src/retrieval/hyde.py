"""HyDE - Hypothetical Document Embeddings.

A question and its answer often look quite different in embedding space: the
question is short and interrogative, the answer is declarative and shares
vocabulary with the source passage. HyDE has the LLM draft a plausible answer
and searches with *that* instead, so we compare passage-to-passage.

The draft is allowed to be factually wrong - it is never shown to the user and
never enters the final context. It only has to look like the right kind of text.
"""

from __future__ import annotations

import logging

from src.config import HydeConfig
from src.llm.base import LLMError, Message
from src.llm.router_llm import LLMRouter
from src.retrieval.base import PRE_RETRIEVAL, RetrievalComponent, RetrievalContext

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write a short, plausible passage that would answer the \
user's question, as if excerpted from a reference document.

Rules:
- Write 2-4 declarative sentences. No preamble, no hedging, no bullet points.
- Do not say you are unsure or that you lack information. Invent specifics \
freely; this text is used only as a search probe and is never shown to anyone.
- Match the register of an encyclopedia or technical manual."""


class HydeComponent(RetrievalComponent):
    """Replaces ctx.effective_query with a hypothetical answer passage.

    Costs one LLM call per query - the only rate-limited step in retrieval,
    which is why it is off by default.
    """

    name = "hyde"
    stage = PRE_RETRIEVAL

    def __init__(self, llm: LLMRouter, cfg: HydeConfig) -> None:
        self._llm = llm
        self._cfg = cfg

    def run(self, ctx: RetrievalContext) -> RetrievalContext:
        try:
            response = self._llm.complete(
                self._cfg.role,
                [Message("system", SYSTEM_PROMPT), Message("user", ctx.query)],
                temperature=0.0,
                max_tokens=self._cfg.max_tokens,
            )
        except LLMError as exc:
            # Retrieval must survive a rate-limited or failed provider; falling
            # back to the raw query is strictly better than returning nothing.
            logger.warning("hyde: generation failed (%s); using the raw query", exc)
            ctx.log(f"hyde: failed ({type(exc).__name__}), fell back to raw query")
            return ctx

        hypothetical = response.text.strip()
        if not hypothetical:
            ctx.log("hyde: empty draft, using raw query")
            return ctx

        # Keep the original question in the probe: the draft may drift off topic,
        # and the question's own keywords are still signal.
        ctx.effective_query = f"{ctx.query}\n\n{hypothetical}"
        ctx.log(f"hyde: drafted {len(hypothetical)} chars via {response.model}")
        return ctx
