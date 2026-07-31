"""Dispatch a query to a path.

Paths live in a registry, so adding one (slice 4's `agentic`, a future `graph`)
means registering a pipeline and listing it in `router.paths` - the router
itself never changes.
"""

from __future__ import annotations

import logging

from src.config import Config
from src.pipelines.base import Pipeline, ProgressEvent, RagResult
from src.router.classifier import QueryClassifier, RouteDecision

logger = logging.getLogger(__name__)


class Router:
    def __init__(
        self,
        config: Config,
        pipelines: dict[str, Pipeline],
        classifier: QueryClassifier | None = None,
    ) -> None:
        self._config = config
        self._pipelines = pipelines
        self._classifier = classifier

    @property
    def available_paths(self) -> list[str]:
        return sorted(self._pipelines)

    def decide(self, query: str) -> RouteDecision:
        cfg = self._config.router
        if not cfg.enabled or self._classifier is None:
            # No classifier call at all - the router being off must cost nothing.
            return RouteDecision(
                path=cfg.default_path, complexity="", reason="router disabled", classified=False
            )

        decision = self._classifier.classify(query)

        # A path can be chosen before it exists (agentic lands in slice 4).
        # Degrade to the default rather than failing, and record that we did.
        if decision.path not in self._pipelines:
            logger.warning(
                "Path %r is not implemented yet; running %r instead",
                decision.path, cfg.default_path,
            )
            decision = RouteDecision(
                path=cfg.default_path,
                requested_path=decision.path,
                complexity=decision.complexity,
                reason=decision.reason,
                classified=decision.classified,
            )
        return decision

    def stream(self, query: str):
        """Classify, announce the path, then stream the chosen pipeline.

        The route event is emitted before any pipeline work begins, so a UI can
        show which path was picked while the answer is still being produced.
        """
        yield ProgressEvent("status", message="Classifying query complexity…")
        decision = self.decide(query)
        yield ProgressEvent("route", route=decision)

        pipeline = self._pipelines.get(decision.path)
        if pipeline is None:
            raise KeyError(
                f"No pipeline registered for path {decision.path!r}. "
                f"Available: {', '.join(self.available_paths) or '<none>'}"
            )

        for event in pipeline.stream(query):
            if event.kind == "result" and event.result is not None:
                self._attach(event.result, decision)
            yield event

    def run(self, query: str) -> RagResult:
        result = None
        for event in self.stream(query):
            if event.kind == "result":
                result = event.result
        if result is None:
            raise RuntimeError("router produced no result")
        return result

    @staticmethod
    def _attach(result: RagResult, decision: RouteDecision) -> None:
        result.route = decision
        if decision.complexity:
            result.trace.insert(
                0, f"router: {decision.complexity} -> {decision.path}"
                   + (f" (wanted {decision.requested_path})" if decision.degraded else "")
            )
