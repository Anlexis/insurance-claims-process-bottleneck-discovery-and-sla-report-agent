"""AgentCore Platform v1.0"""

# State contract for INS-C2-020.
#
# The state is a flat TypedDict extending the framework's AgentState — never a
# Pydantic model. Checkpoints are serialised with msgpack, which cannot round-
# trip arbitrary Python objects, so a model instance in state corrupts silently
# rather than failing loudly.
#
# Every dict- or list-valued field is stored as a JSON string
# (Optional[str]) and passed through the to_json() / from_json() helpers below
# at every producer and every consumer, so there is one serialisation contract
# end to end. Never type such a field as a bare dict or list.
#
# Never place credentials, secrets or personal data in state: it is written to
# the checkpoint store.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for state storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string held in state; return *default* on any failure."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat state for INS-C2-020.

    Shared fields (user_input, input_context, status, session_id, node_history,
    error_log, formatted_output, hitl_*) are inherited from AgentState.
    formatted_output is deliberately NOT re-declared here: re-declaring an
    inherited field with a narrower type is a state-contract defect.
    """

    # ------------------------------------------------------------------
    # Outer layer — written by PreProcessNode (pre_process backbone slot)
    # ------------------------------------------------------------------

    # Normalised JSON query payload {"query", "filters"}.
    # Produced by PreProcessNode; consumed by the inner InputValidateNode.
    validated_input: NotRequired[Optional[str]]

    # JSON-serialised request metadata. Shape: {"source": str, "channel": str}.
    enriched_context: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer — the retrieval workflow (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # JSON-serialised, already-validated runtime settings from
    # config/config.yaml. Seeded by DomainWorkflowGraph._extra_initial_state();
    # it is the only route by which a declared value reaches a domain node,
    # because node constructors take no arguments and execute() takes no config
    # argument.
    runtime_settings: NotRequired[Optional[str]]

    # JSON-serialised normalised query payload.
    # Shape: {"query": str, "filters": {"process_step"?: str, "sla_metric"?: str}}
    retrieval_query: NotRequired[Optional[str]]

    # JSON-serialised list of retrieved candidate documents.
    # Each entry: {"doc_id", "source", "title", "text", "score"}
    retrieved_documents: NotRequired[Optional[str]]

    # Number of candidate documents returned by RetrieveNode.
    retrieved_count: NotRequired[Optional[int]]

    # JSON-serialised list of reranked + threshold-filtered documents.
    # Same entry shape as retrieved_documents; sorted by score, descending.
    reranked_documents: NotRequired[Optional[str]]

    # Number of documents surviving the rerank/filter step.
    reranked_count: NotRequired[Optional[int]]

    # Grounded answer synthesised from reranked_documents by GenerateAnswerNode.
    generated_answer: NotRequired[Optional[str]]

    # JSON-serialised list of citation records: {"doc_id", "source", "score"}.
    answer_citations: NotRequired[Optional[str]]

    # The assembled report (question + grounded answer + sources).
    # Produced by the inner OutputFormatNode; screened by PostProcessNode.
    rag_report: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Outer layer — written by PostProcessNode (post_process backbone slot)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller. Set to the same content as
    # rag_report once the output boundary has cleared it; formatted_output
    # (inherited) is set alongside it.
    result: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing — framework-managed; never written by node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    # node_history is inherited from AgentState
