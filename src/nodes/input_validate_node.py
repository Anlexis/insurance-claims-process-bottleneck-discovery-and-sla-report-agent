"""AgentCore Platform v1.0"""

# INS-C2-020 — InputValidateNode
# Inner domain node 1: the retrieval workflow's own input contract.
#
# Distinct from PreProcessNode, which owns the agent's external trust boundary
# and the request envelope. This node owns the retrieval contract: a non-empty
# question and a set of retrieval facets drawn from the published vocabulary.
# It re-checks both because the inner graph is a composable unit — a caller
# that embeds DomainWorkflowGraph directly gets the same guarantees as a caller
# that goes through the backbone.
#
# Inner node — ANONYMOUS trust: the trust boundary is enforced once, at the
# outer pre_process slot, and the inner nodes must accept the context that
# passes through the subgraph boundary.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import json
import logging
from typing import Any, ClassVar, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import screen_structure, validate_filters

logger = logging.getLogger(__name__)

# Bound on the question the retrieval step will act on. The outer boundary caps
# the request envelope; this caps what reaches the scorer, and it refuses
# rather than truncates — a truncated question is a different question.
_MAX_QUERY_CHARS = 4000


# Reason codes whose message names the screen that fired rather than a field the caller
# can correct. These refusals publish nothing; every other reason publishes its reason.
_SILENT_REASONS = frozenset({"instruction_override"})


class InputValidateNode(FunctionNode):
    """Validate the retrieval request for INS-C2-020.

    Input state keys:
        validated_input: str  — normalised JSON {"query", "filters"} from
                                PreProcessNode; falls back to user_input when
                                the inner graph is driven directly

    Output state keys (partial dict):
        retrieval_query: str  — JSON {"query", "filters"}
        status:          str
        error_log:       list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")

        if not isinstance(raw, str) or not raw.strip():
            return self._reject("empty_input", "no question supplied", state)

        parsed = self._parse(raw)
        if parsed is None:
            return self._reject("malformed_payload", "request is not a supported shape", state)

        query, raw_filters = parsed
        query = query.strip()

        if not query:
            return self._reject("empty_query", "question is missing or empty", state)

        if len(query) > _MAX_QUERY_CHARS:
            return self._reject(
                "query_too_long",
                f"question exceeds the {_MAX_QUERY_CHARS}-character limit",
                state,
            )

        finding = screen_structure({"query": query, "filters": raw_filters})
        if finding:
            return self._reject(
                "instruction_override",
                "request rejected: disallowed instruction content",
                state,
                detail=finding,
            )

        filters, filter_error = validate_filters(raw_filters)
        if filter_error is not None:
            return self._reject("invalid_filters", filter_error, state)

        logger.info(
            "InputValidateNode: query_chars=%d filters=%d",
            len(query),
            len(filters),
        )
        emit_trace_event(
            "input_validate_complete",
            {"query_chars": len(query), "filter_keys": sorted(filters)},
            state,
        )

        return {
            "retrieval_query": to_json({"query": query, "filters": filters}),
            "status": AgentStatus.SUCCESS.value,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: str) -> Optional[tuple[str, Any]]:
        """Split the request into (question, raw filters); None when unusable."""
        text = raw.strip()
        if not text.startswith("{"):
            return text, None
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        query = payload.get("query", "")
        if not isinstance(query, str):
            return None
        return query, payload.get("filters")

    @staticmethod
    def _reject(reason: str, message: str, state: AgentState, detail: Any = None) -> dict[str, Any]:
        """Fail closed with an audit event that names the field, never the value."""
        event: dict[str, Any] = {"reason": reason}
        if detail is not None:
            event["detail"] = detail
        emit_trace_event("input_validate_failed", event, state)
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"InputValidateNode: {message}"],
            # A SCREENED refusal stays silent. Its message names the marker that caught the
            # payload, and handing that back lets an attacker probe the screen one try at a
            # time. A VALIDATION refusal says what to fix -- without it the caller cannot tell
            # a rejected request from a hung one.
            **(
                {}
                if reason in _SILENT_REASONS
                else {"formatted_output": "Request could not be completed. " + f"InputValidateNode: {message}"}
            ),
        }
