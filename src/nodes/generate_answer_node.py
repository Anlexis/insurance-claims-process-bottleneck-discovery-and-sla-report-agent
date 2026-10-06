"""AgentCore Platform v1.0"""

# INS-C2-020 — GenerateAnswerNode
# Inner domain node 4: assemble the grounded answer.
#
# The grounding contract is the point of this node: every sentence of the
# answer comes from a document that survived the relevance floor, and when
# nothing survived the node says so explicitly instead of producing an
# ungrounded answer. The manifest declares generation_mode: deterministic, so
# no model is invoked; the declared prompt template names the grounding
# contract a model deployment would be given, and whether it resolved is
# recorded in the audit event so the two deployments are distinguishable.
#
# Declared settings consumed (state["runtime_settings"]):
#   system_prompt_template — repository-relative path to the grounding prompt
#
# Inner node — ANONYMOUS trust (the boundary is enforced at the outer slot).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from pathlib import Path
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# src/nodes/generate_answer_node.py -> parents[2] is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_INSUFFICIENT_GROUNDING = (
    "No claims-operations knowledge cleared the relevance threshold for this question. "
    "The agent has no grounded evidence to answer from and will not speculate. "
    "Refine the question (name a specific process step or service-level metric), widen the "
    "retrieval facets, or extend the knowledge base."
)


def _grounded_answer(query: str, docs: list[dict[str, Any]]) -> str:
    """Assemble the answer from the grounding set.

    The leading document frames the answer and the rest are folded in as
    supporting findings. Every line is drawn from a document that is cited in
    the report's sources section, which is what makes the answer auditable.
    """
    lead = docs[0]
    lines = [
        f"Question: {query}",
        "",
        "Grounded answer (drawn strictly from the retrieved claims-operations knowledge):",
        "",
        f"- {lead.get('title', 'Primary finding')}: {str(lead.get('text', '')).strip()}",
    ]
    for doc in docs[1:]:
        title = doc.get("title", "Supporting finding")
        text = str(doc.get("text", "")).strip()
        lines.append(f"- {title}: {text}")
    lines += [
        "",
        "This answer is limited to the cited sources; see the sources section for provenance.",
    ]
    return "\n".join(lines)


class GenerateAnswerNode(FunctionNode):
    """Produce a grounded answer from the reranked documents.

    Input state keys:
        reranked_documents: str  — JSON list from RerankFilterNode
        retrieval_query:    str  — JSON {"query", "filters"} (for the question text)
        runtime_settings:   str  — JSON declared settings

    Output state keys (partial dict):
        generated_answer: str
        answer_citations: str  — JSON list of {doc_id, source, score}
        status:           str
        error_log:        list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        if state.get("reranked_documents") is None:
            emit_trace_event(
                "generate_answer_failed",
                {"reason": "missing_reranked_documents"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["GenerateAnswerNode: reranked_documents is absent from state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("GenerateAnswerNode: reranked_documents is absent from state"),
            }

        documents = from_json(state.get("reranked_documents"), [])
        if not isinstance(documents, list):
            documents = []
        documents = [d for d in documents if isinstance(d, dict)]

        query_payload = from_json(state.get("retrieval_query"), {}) or {}
        query = str(query_payload.get("query", "")).strip() or "(unspecified question)"

        settings: dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        template_resolved = self._template_resolves(settings.get("system_prompt_template"))

        if documents:
            answer = _grounded_answer(query, documents)
            citations = [
                {
                    "doc_id": d.get("doc_id"),
                    "source": d.get("source"),
                    "score": d.get("score", 0.0),
                }
                for d in documents
            ]
            grounded = True
        else:
            answer = _INSUFFICIENT_GROUNDING
            citations = []
            grounded = False

        logger.info(
            "GenerateAnswerNode: grounded=%s citations=%d answer_chars=%d",
            grounded,
            len(citations),
            len(answer),
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "grounded": grounded,
                "citation_count": len(citations),
                "prompt_template_resolved": template_resolved,
            },
            state,
        )

        return {
            "generated_answer": answer,
            "answer_citations": to_json(citations),
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _template_resolves(declared: Any) -> bool:
        """True when the declared grounding-prompt path names a file that exists.

        A declared path is repository-relative and must stay inside the
        repository: a declared value is not caller data, but resolving an
        arbitrary path out of a configuration file is not a capability this
        node needs.
        """
        if not isinstance(declared, str) or not declared:
            return False
        candidate = (_REPO_ROOT / declared).resolve()
        if not candidate.is_relative_to(_REPO_ROOT):
            return False
        return candidate.is_file()
