"""AgentCore Platform v1.0"""

# INS-C2-020 — RetrieveNode
# Inner domain node 2: retrieve candidate knowledge for the validated question.
#
# Retrieval runs over an embedded, curated claims-operations corpus and is
# deterministic: a lexical token-overlap scorer ranks the documents, and the
# caller's retrieval facets restrict the candidate set to the documents that
# actually cover the requested process step and service-level metric. A
# deployment that wires a managed document index replaces `_CLAIMS_KB` and
# `_score`; the facet contract and the settings read here are unchanged.
#
# Declared settings consumed (state["runtime_settings"], seeded by the inner
# graph from config/config.yaml):
#   top_k          — candidate cap
#   hybrid_search  — when true, an exact match on a document identifier or
#                    source name also retrieves the document, so a question
#                    that names a reference retrieves it even with no lexical
#                    overlap in the prose
#
# Inner node — ANONYMOUS trust (the boundary is enforced at the outer slot).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.service import finite_in_range

logger = logging.getLogger(__name__)

# Fallbacks used when the corresponding key is absent from runtime_settings —
# either not declared, or declared with a value that failed validation.
_DEFAULT_TOP_K = 8
_DEFAULT_HYBRID_SEARCH = False

_TOP_K_BOUNDS = (1.0, 50.0)

# Embedded claims-operations reference corpus. Each entry is a self-contained,
# citable knowledge snippet carrying the retrieval facets it covers; the facet
# values are exactly the published vocabulary in src/services/service.py, which
# a test asserts.
_CLAIMS_KB: list[dict[str, Any]] = [
    {
        "doc_id": "KB-FNOL-01",
        "source": "claims_process_playbook.md#fnol",
        "title": "First notice of loss: intake and triage",
        "text": (
            "First notice of loss is the intake step where a claim is registered. "
            "Bottlenecks here are usually caused by incomplete first-contact data and manual "
            "channel routing, which delay assignment to an adjuster."
        ),
        "process_step": ["fnol"],
        "sla_metric": ["acknowledgement_time"],
    },
    {
        "doc_id": "KB-ADJ-02",
        "source": "claims_process_playbook.md#adjuster-assignment",
        "title": "Adjuster assignment and workload balancing",
        "text": (
            "Adjuster assignment breaches occur when queue depth exceeds team capacity. "
            "The most common bottleneck is uneven workload balancing across adjusters, which "
            "inflates cycle time between first notice of loss and the first adjuster touch."
        ),
        "process_step": ["adjuster_assignment"],
        "sla_metric": ["assignment_time", "cycle_time"],
    },
    {
        "doc_id": "KB-SLA-03",
        "source": "service_level_reference.md#thresholds",
        "title": "Service-level thresholds for claims settlement",
        "text": (
            "Standard targets: first-notice acknowledgement within 24 hours, adjuster assignment "
            "within 48 hours, and a settlement decision within 15 business days. A breach is any "
            "step whose elapsed time exceeds its target threshold."
        ),
        "process_step": ["fnol", "adjuster_assignment", "settlement"],
        "sla_metric": ["acknowledgement_time", "assignment_time", "settlement_time"],
    },
    {
        "doc_id": "KB-BTL-04",
        "source": "process_analysis_notes.md#bottleneck-discovery",
        "title": "Bottleneck discovery from the event log",
        "text": (
            "Process bottlenecks are discovered by ranking process steps by mean waiting time "
            "and by the proportion of cases that stall at each step. The step with the highest "
            "waiting-time contribution to end-to-end cycle time is the primary bottleneck."
        ),
        "process_step": ["analysis"],
        "sla_metric": ["cycle_time"],
    },
    {
        "doc_id": "KB-DEV-05",
        "source": "process_analysis_notes.md#deviation-patterns",
        "title": "Deviation and rework patterns",
        "text": (
            "Deviation patterns such as rework loops (repeated document requests) and back-and-forth "
            "between adjuster and customer are strong drivers of settlement delay. Frequent rework "
            "at the documentation step signals a data-quality problem at intake."
        ),
        "process_step": ["documentation"],
        "sla_metric": ["settlement_time", "cycle_time"],
    },
    {
        "doc_id": "KB-SET-06",
        "source": "claims_process_playbook.md#settlement",
        "title": "Settlement and payment step",
        "text": (
            "Settlement delay is often concentrated in approval hand-offs when a claim exceeds an "
            "authority limit and must be escalated. Escalation queues without an elapsed-time timer "
            "are a recurring source of silent breaches of the settlement threshold."
        ),
        "process_step": ["settlement"],
        "sla_metric": ["settlement_time", "breach_count"],
    },
    {
        "doc_id": "KB-REG-07",
        "source": "regulatory_reference.md#fair-claims-handling",
        "title": "Fair-claims handling obligations",
        "text": (
            "Insurers are required to demonstrate fair and efficient claims processing with "
            "documented service-level compliance. Bottleneck evidence and breach tracking are used "
            "as audit artefacts by supervisors reviewing claims conduct."
        ),
        "process_step": ["compliance"],
        "sla_metric": ["sla_adherence"],
    },
    {
        "doc_id": "KB-SUP-08",
        "source": "regulatory_reference.md#operational-resilience",
        "title": "Operational-resilience metrics supervisors expect",
        "text": (
            "Supervisory guidance emphasises operational-resilience metrics. Average claim "
            "settlement time and step-level adherence are the operational indicators supervisors "
            "expect insurers to monitor and report on."
        ),
        "process_step": ["compliance"],
        "sla_metric": ["settlement_time", "sla_adherence"],
    },
    {
        "doc_id": "KB-KPI-09",
        "source": "service_level_reference.md#kpi-definitions",
        "title": "Cycle-time and adherence indicators",
        "text": (
            "Key operational indicators: end-to-end cycle time from first notice of loss to "
            "settlement, per-step adherence rate, and breach count by step. Improvement "
            "recommendations target the step with the worst adherence rate and the largest "
            "cycle-time share."
        ),
        "process_step": ["analysis"],
        "sla_metric": ["cycle_time", "sla_adherence", "breach_count"],
    },
    {
        "doc_id": "KB-REC-10",
        "source": "process_analysis_notes.md#recommendations",
        "title": "Improvement recommendations",
        "text": (
            "Typical recommendations: automate intake data capture to cut rework, add elapsed-time "
            "timers to escalation and approval queues, and rebalance adjuster workload to reduce "
            "assignment waiting time. Each recommendation maps to a discovered bottleneck."
        ),
        "process_step": ["analysis", "fnol", "adjuster_assignment", "settlement"],
        "sla_metric": ["cycle_time", "assignment_time"],
    },
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase word/number tokens for lexical overlap scoring."""
    return set(_TOKEN_RE.findall(text.lower()))


def _score(query_tokens: set[str], doc: dict[str, Any]) -> float:
    """Lexical overlap score in [0, 1] = |query & doc| / |query|.

    Document tokens combine the title and the body so a strong title match
    still scores. Deterministic and dependency-free.
    """
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(str(doc.get("title", "")) + " " + str(doc.get("text", "")))
    overlap = len(query_tokens & doc_tokens)
    return round(overlap / len(query_tokens), 4)


def _matches_facets(doc: dict[str, Any], filters: dict[str, str]) -> bool:
    """True when the document carries every facet value the caller asked for."""
    for facet, wanted in filters.items():
        if wanted not in doc.get(facet, []):
            return False
    return True


def _keyword_hit(query: str, doc: dict[str, Any]) -> bool:
    """True when the question names the document's identifier or its source.

    This is the keyword half of hybrid retrieval: claims references are codes,
    and a code shares no prose tokens with the passage that defines it, so
    lexical overlap alone scores it zero.
    """
    lowered = query.lower()
    return str(doc["doc_id"]).lower() in lowered or str(doc["source"]).lower() in lowered


class RetrieveNode(FunctionNode):
    """Retrieve candidate knowledge documents for the validated question.

    Input state keys:
        retrieval_query:  str  — JSON {"query", "filters"} from InputValidateNode
        runtime_settings: str  — JSON declared settings; the route by which
                                 top_k / hybrid_search reach this node

    Output state keys (partial dict):
        retrieved_documents: str  — JSON list of {doc_id, source, title, text, score}
        retrieved_count:     int
        status:              str
        error_log:           list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        payload = from_json(state.get("retrieval_query"), {}) or {}
        query = str(payload.get("query", "")).strip()
        filters: dict[str, str] = payload.get("filters") or {}

        if not query:
            emit_trace_event("retrieve_failed", {"reason": "missing_query"}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: retrieval_query is missing or carries an empty question"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("RetrieveNode: retrieval_query is missing or carries an empty question"),
            }

        settings: dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}

        declared_top_k = finite_in_range(settings.get("top_k"), *_TOP_K_BOUNDS)
        top_k = int(declared_top_k) if declared_top_k is not None else _DEFAULT_TOP_K

        declared_hybrid = settings.get("hybrid_search")
        hybrid = declared_hybrid if isinstance(declared_hybrid, bool) else _DEFAULT_HYBRID_SEARCH

        query_tokens = _tokenize(query)
        scored: list[dict[str, Any]] = []
        for doc in _CLAIMS_KB:
            if not _matches_facets(doc, filters):
                continue
            score = _score(query_tokens, doc)
            if hybrid and _keyword_hit(query, doc):
                # An exact reference to the document is a definitive match, so
                # it scores 1.0. Anything less would be filtered out by the
                # relevance floor downstream and the keyword half of hybrid
                # retrieval would never reach an answer.
                score = 1.0
            if score <= 0.0:
                continue
            scored.append(
                {
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": score,
                }
            )

        scored.sort(key=lambda d: float(d["score"]), reverse=True)
        retrieved = scored[:top_k]

        logger.info(
            "RetrieveNode: query_tokens=%d candidates=%d top_k=%d hybrid=%s returned=%d",
            len(query_tokens),
            len(scored),
            top_k,
            hybrid,
            len(retrieved),
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "candidate_count": len(scored),
                "returned_count": len(retrieved),
                "top_k": top_k,
                "hybrid_search": hybrid,
                "filter_keys": sorted(filters),
            },
            state,
        )

        return {
            "retrieved_documents": to_json(retrieved),
            "retrieved_count": len(retrieved),
            "status": AgentStatus.SUCCESS.value,
        }
