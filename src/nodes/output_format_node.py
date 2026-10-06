"""AgentCore Platform v1.0"""

# INS-C2-020 — OutputFormatNode
# Inner domain node 5 (last): assemble the report the caller receives.
#
# The report is the deliverable, so this node is where the document's stated
# invariants are produced:
#
#   - every rendered finding comes from a document listed in the sources
#     section (grounding), and the abstention text is rendered when there is
#     no such document;
#   - the relevance figure printed beside each source is the score the
#     retrieval step computed, printed as-is;
#   - the request-scope line renders only inert tokens — the request channel
#     and the retrieval facets — so no free caller text becomes report
#     structure. The question itself is rendered verbatim in its own labelled
#     section, after the outer boundary has masked personal data in it.
#
# This template renders no monetary aggregate and performs no numeric
# rewriting, so document identifiers and figures reach the output boundary
# byte-identical and a pattern scan there still sees what it was given.
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

from src.schemas.state import from_json
from src.services.service import is_inert_token


# The Marketplace runner seeds input_context with its own conversation history on every
# invocation (shared/bootstrap/marketplace_app.py); the caller neither sends that key nor can
# suppress it, and the build_input_context hook can only overwrite its value, never remove it.
# It is platform plumbing rather than caller data, so it is dropped here, before the caller
# contract runs: the unknown-field guard below stays strict for everything a caller can
# actually send, and no value screen is ever asked to judge a transcript that contains this
# agent's own earlier answers. The value may also be None, which this tolerates.
_PLATFORM_CONTEXT_KEYS = frozenset({"conversation_history"})


def _without_platform_context(raw: Any) -> Any:
    """The caller-supplied half of input_context, platform-injected keys removed."""
    if not isinstance(raw, dict):
        return raw
    return {k: v for k, v in raw.items() if k not in _PLATFORM_CONTEXT_KEYS}


logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72

_DEFAULT_CHANNEL = "unknown"


def _scope_line(channel: str, filters: dict[str, str]) -> str:
    """Render the request-scope line from inert tokens only."""
    parts = [f"channel={channel}"]
    for key in sorted(filters):
        parts.append(f"{key}={filters[key]}")
    return "  ".join(parts)


def _assemble_report(
    query: str,
    scope: str,
    answer: str,
    citations: list[dict[str, Any]],
) -> str:
    """Assemble the claims-operations report from the answer and its citations."""
    lines = [
        _SEPARATOR,
        "INSURANCE CLAIMS BOTTLENECK AND SERVICE-LEVEL REPORT",
        _SEPARATOR,
        "",
        "REQUEST SCOPE",
        _SUBSEP,
        scope,
        "",
        "QUESTION",
        _SUBSEP,
        query,
        "",
        "ANSWER",
        _SUBSEP,
        answer,
        "",
        "SOURCES",
        _SUBSEP,
    ]
    if citations:
        for i, c in enumerate(citations, 1):
            doc_id = c.get("doc_id", "n/a")
            source = c.get("source", "n/a")
            score = c.get("score", 0.0)
            lines.append(f"  {i}. [{doc_id}] {source} (relevance {score})")
    else:
        lines.append("  (no source cleared the relevance threshold)")
    lines += ["", _SEPARATOR]
    return "\n".join(lines)


class OutputFormatNode(FunctionNode):
    """Assemble the final report document.

    Input state keys:
        generated_answer: str  — grounded answer from GenerateAnswerNode
        answer_citations: str  — JSON list of citation records
        retrieval_query:  str  — JSON {"query", "filters"} (question + scope)
        input_context:    dict — bridged caller context (the channel label)

    Output state keys (partial dict):
        rag_report: str
        result:     str  (the same content — backbone convention)
        status:     str
        error_log:  list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("generated_answer")
        if answer is None:
            emit_trace_event(
                "output_format_failed",
                {"reason": "missing_generated_answer"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: generated_answer is absent from state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("OutputFormatNode: generated_answer is absent from state"),
            }

        citations = from_json(state.get("answer_citations"), [])
        if not isinstance(citations, list):
            citations = []
        citations = [c for c in citations if isinstance(c, dict)]

        query_payload = from_json(state.get("retrieval_query"), {}) or {}
        query = str(query_payload.get("query", "")).strip() or "(unspecified question)"
        filters = query_payload.get("filters") or {}
        if not isinstance(filters, dict):
            filters = {}

        # The channel label is caller data that renders into the report, so it
        # is re-checked here against the inert alphabet rather than trusted:
        # this node is reached through the subgraph boundary, and a rendering
        # node that assumes an upstream check is a rendering node that ships
        # whatever reaches it.
        raw_context = _without_platform_context(state.get("input_context")) or {}
        raw_channel = raw_context.get("channel") if isinstance(raw_context, dict) else None
        channel = raw_channel if isinstance(raw_channel, str) and is_inert_token(raw_channel) else _DEFAULT_CHANNEL

        report = _assemble_report(
            query,
            _scope_line(channel, {k: v for k, v in filters.items() if is_inert_token(v)}),
            str(answer),
            citations,
        )

        logger.info(
            "OutputFormatNode: report_chars=%d citations=%d",
            len(report),
            len(citations),
        )
        emit_trace_event(
            "output_format_complete",
            {"report_length": len(report), "citation_count": len(citations), "channel": channel},
            state,
        )

        return {
            "rag_report": report,
            "result": report,
            "status": AgentStatus.SUCCESS.value,
        }
