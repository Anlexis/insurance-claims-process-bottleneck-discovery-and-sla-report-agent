"""AgentCore Platform v1.0"""

# INS-C2-020 — RerankFilterNode
# Inner domain node 3: rerank the candidates and keep only the ones relevant
# enough to ground an answer.
#
# The rerank is the retrieval score itself (a stable sort); a deployment that
# wires a cross-encoder replaces `_rank` at the same seam. The relevance floor
# is the declared score_threshold: it decides whether the agent answers from
# evidence or abstains, which is why an invalid declared value must fall back
# to the documented default rather than propagate — a non-finite threshold
# compares False against every score and would silently drop every passage.
#
# Declared settings consumed (state["runtime_settings"]):
#   score_threshold — relevance floor, 0.0-1.0
#
# Inner node — ANONYMOUS trust (the boundary is enforced at the outer slot).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.service import finite_in_range

logger = logging.getLogger(__name__)

_DEFAULT_SCORE_THRESHOLD = 0.3
_SCORE_THRESHOLD_BOUNDS = (0.0, 1.0)

# Cap on the grounding set handed to the answer step. Not caller-settable and
# not declared: it bounds how much evidence one answer may cite, which is a
# property of the deliverable rather than a tuning knob.
_MAX_GROUNDING_DOCUMENTS = 5


class RerankFilterNode(FunctionNode):
    """Rerank the retrieved documents and apply the relevance floor.

    Input state keys:
        retrieved_documents: str  — JSON list from RetrieveNode
        runtime_settings:    str  — JSON declared settings

    Output state keys (partial dict):
        reranked_documents: str  — JSON list, score-sorted and filtered
        reranked_count:     int
        status:             str
        error_log:          list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A missing key means the retrieval step never ran — fail fast. An
        # empty list is a legitimate "nothing matched" and passes through.
        if state.get("retrieved_documents") is None:
            emit_trace_event(
                "rerank_filter_failed",
                {"reason": "missing_retrieved_documents"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RerankFilterNode: retrieved_documents is absent from state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("RerankFilterNode: retrieved_documents is absent from state"),
            }

        documents = from_json(state.get("retrieved_documents"), [])
        if not isinstance(documents, list):
            documents = []

        settings: dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        declared = finite_in_range(settings.get("score_threshold"), *_SCORE_THRESHOLD_BOUNDS)
        threshold = declared if declared is not None else _DEFAULT_SCORE_THRESHOLD

        ranked = sorted(documents, key=self._sort_key, reverse=True)
        kept = [doc for doc in ranked if self._sort_key(doc) >= threshold][:_MAX_GROUNDING_DOCUMENTS]

        logger.info(
            "RerankFilterNode: in=%d threshold=%.3f kept=%d",
            len(documents),
            threshold,
            len(kept),
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "input_count": len(documents),
                "score_threshold": threshold,
                "kept_count": len(kept),
            },
            state,
        )

        return {
            "reranked_documents": to_json(kept),
            "reranked_count": len(kept),
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _sort_key(doc: Any) -> float:
        """Relevance score of a candidate; 0.0 for anything unparseable.

        A candidate whose score is missing or non-numeric sorts last and falls
        below any positive floor, so a malformed record can never displace a
        real one.
        """
        if not isinstance(doc, dict):
            return 0.0
        score = finite_in_range(doc.get("score"), 0.0, 1.0)
        return score if score is not None else 0.0
