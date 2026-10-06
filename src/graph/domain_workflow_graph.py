"""AgentCore Platform v1.0"""

# INS-C2-020 — DomainWorkflowGraph (inner BaseGraph)
#
# The inner graph of the two-layer nested composition. It encapsulates the
# claims-knowledge retrieval pipeline:
#
#   START
#     -> input_validate   (InputValidateNode)
#     -> retrieve         (RetrieveNode)
#     -> rerank_filter    (RerankFilterNode)
#     -> generate_answer  (GenerateAnswerNode)   <- grounded answer step
#     -> output_format    (OutputFormatNode)
#     -> END
#
# Called by ClaimsRAGGraphNode.get_subgraph() (src/graph/graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Composition rules this graph holds to:
#   - inherits BaseGraph (fully custom topology — no fixed backbone)
#   - register_nodes() does NOT call super() (it is abstract on BaseGraph)
#   - initialize / finalize are outer-backbone concerns and are not registered
#   - every inner node declares required_trust_level = TrustLevel.ANONYMOUS,
#     because the trust boundary is enforced once, at the outer pre_process slot
#   - node constructors take no arguments; runtime settings arrive through state

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for INS-C2-020.

    Pipeline (linear):
        START -> input_validate -> retrieve -> rerank_filter -> generate_answer
              -> output_format -> END
    """

    # -- Identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ins_c2_020_claims_bottleneck_retrieval_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # -- Config validation ----------------------------------------------------

    def _validate_config(self) -> None:
        """No key is mandatory: every declared setting was already validated.

        ClaimsRAGGraphNode._parent_config() parses each declared value for
        type, finiteness and range and forwards only the ones that pass, so an
        absent key here means "not declared, or declared invalid" and the
        consuming node keeps its own documented default.
        """
        return None

    # -- Initial state --------------------------------------------------------

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the inner graph's initial state.

        Two things cross the outer -> inner boundary here, and neither can
        travel any other way:

        - ``input_context`` — the framework's GraphNode.execute() invokes this
          subgraph without forwarding it, so it is stashed by the outer node's
          extract_input() and picked up here (src/graph/context_bridge.py).
        - ``runtime_settings`` — the declared, already-validated retrieval
          settings. Node constructors take no arguments and ``execute(state)``
          takes no config argument, so state is the route by which a value in
          config/config.yaml reaches a domain node.
        """
        return {
            "input_context": get_caller_input_context(),
            "runtime_settings": to_json(dict(self.config)),
        }

    # -- Node registration ----------------------------------------------------

    def register_nodes(self) -> None:
        """Register the 5 domain nodes. Every key registered is used in add_edges()."""
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring ----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear retrieval topology.

        There is no conditional branching: a query either produces a grounded
        answer or an explicit abstention, and both are produced by the same
        path. route() satisfies the abstract contract but is never reached.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing --------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Required by the abstract contract; not wired to a conditional edge.

        Returns END on an error status so an unexpected invocation cannot
        re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ---------------------------------------------------------

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the dict returned to the outer graph as sub_result.

        Received by ClaimsRAGGraphNode.merge_output() in src/graph/graph.py.
        Both methods are written together so the field names cannot drift:

            emitted here -> "rag_report", "generated_answer",
                            "answer_citations", "reranked_count", "status"
            read there   -> sub_result.get(...) for each key above
        """
        return {
            "rag_report": state.get("rag_report"),
            "generated_answer": state.get("generated_answer"),
            "answer_citations": state.get("answer_citations"),
            "reranked_count": state.get("reranked_count", 0),
            "status": state.get("status"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
