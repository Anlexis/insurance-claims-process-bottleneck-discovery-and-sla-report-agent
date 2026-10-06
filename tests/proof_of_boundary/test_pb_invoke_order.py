# PB-6: Invoke Execution Order Verification
#
# Verifies that BaseNode.__call__() enforces, for every concrete node under
# src/nodes/:
#   trust gate -> node_start -> _security_gate_input() -> execute()
#   -> _security_gate_output() -> node_complete
#
# and that a full backbone invoke of InsuranceClaimsBottleneckDiscoveryAgent
# runs its five slots in order:
#   InitializeNode -> PreProcessNode (pre_process) -> ClaimsRAGGraphNode (main)
#   -> PostProcessNode (post_process) -> FinalizeNode
#
# The backbone invoke uses VERIFIED_EXTERNAL caller trust — the real external
# path. It clears the outer PreProcessNode trust gate (which requires
# VERIFIED_EXTERNAL) and passes through the inner ANONYMOUS domain nodes.
# for_internal() would not represent a real external caller and is deliberately
# not used.

import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest

# -- Template-specific constants ----------------------------------------------

# Class name of the node in the `main` backbone slot.
_MAIN_SLOT_NODE = "ClaimsRAGGraphNode"

# A question that yields a grounded answer. It has strong lexical overlap with
# the embedded knowledge base, so at least one document clears the relevance
# floor and the answer step returns cited evidence rather than the abstention
# text. The retrieval facets are drawn from the published vocabulary.
#
# CONTRACT: deploy/invoke_payload.json["input"] MUST equal this exact string —
# the deployment evidence invoke and this test must exercise the identical
# payload. test_invoke_payload_matches_pb6 below asserts that equality so the
# two can never drift.
_VALID_PAYLOAD = json.dumps(
    {
        "query": (
            "What is the primary bottleneck in the insurance claims settlement "
            "process, and which service-level thresholds are being breached?"
        ),
        "filters": {
            "process_step": "settlement",
            "sla_metric": "settlement_time",
        },
    }
)

# -----------------------------------------------------------------------------


def _discover_node_classes() -> list[type]:
    """Import every module under src/nodes/ and collect concrete node classes."""
    from framework.nodes.base_node import BaseNode

    try:
        pkg = importlib.import_module("src.nodes")
    except ImportError:
        return []

    discovered = []
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="src.nodes."):
        module = importlib.import_module(modname)
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseNode)
                and attr is not BaseNode
                and attr.__module__ == modname
                and not inspect.isabstract(attr)
            ):
                discovered.append(attr)
    return discovered


def _patch_domain_emit(monkeypatch):
    """Patch emit_trace_event in every domain node module (avoids audit-backend calls)."""
    for mod_suffix in (
        "pre_process_node",
        "input_validate_node",
        "retrieve_node",
        "rerank_filter_node",
        "generate_answer_node",
        "output_format_node",
        "post_process_node",
        "main_node",
    ):
        try:
            monkeypatch.setattr(f"src.nodes.{mod_suffix}.emit_trace_event", lambda *a, **k: None)
        except AttributeError:
            pass  # module not yet imported / no emit symbol; fine


class TestInvokeOrder:
    """PB-6: __call__ runs the gate pipeline around execute(), in order."""

    def test_call_order_for_every_node(self, monkeypatch):
        node_classes = _discover_node_classes()
        if not node_classes:
            pytest.skip("no concrete node classes found under src/nodes/")

        import framework.nodes.base_node as base_node_module

        failures: list[str] = []
        for node_cls in node_classes:
            order: list[str] = []
            monkeypatch.setattr(
                base_node_module,
                "emit_trace_event",
                lambda event_type, _payload, _state, _o=order: _o.append(f"event:{event_type}"),
            )

            for method_name, label in (
                ("_security_gate_input", "security_gate_input"),
                ("execute", "execute"),
                ("_security_gate_output", "security_gate_output"),
            ):
                original = getattr(node_cls, method_name)

                def spy(self, arg, _o=order, _label=label, _orig=original):
                    _o.append(_label)
                    return _orig(self, arg)

                monkeypatch.setattr(node_cls, method_name, spy)

            instance = node_cls()
            # Caller trust == the node's required level so the gate always passes
            # here; the denial branch is asserted separately below.
            state = {
                "caller_trust_level": node_cls.required_trust_level.value,
                "correlation_id": "pb6-invoke-order-test",
            }
            instance(state)

            expected = [
                "event:node_start",
                "security_gate_input",
                "execute",
                "security_gate_output",
                "event:node_complete",
            ]
            if order != expected:
                failures.append(
                    f"{node_cls.__name__}: invoke order violation.\nexpected: {expected}\nactual:   {order}"
                )

        assert not failures, "\n\n".join(failures)


class TestTrustGate:
    """PB-6: the trust gate runs BEFORE execute() and denies an under-trusted caller."""

    def test_pre_process_denies_anonymous_caller(self, monkeypatch):
        """PreProcessNode (requires VERIFIED_EXTERNAL) must refuse an ANONYMOUS caller."""
        _patch_domain_emit(monkeypatch)
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.trust_level import TrustLevel

        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
                "user_input": _VALID_PAYLOAD,
                "correlation_id": "pb6-trust-denial",
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert any(
            "trust gate" in e.lower() for e in result.get("error_log", [])
        ), f"expected a trust-gate denial, got error_log={result.get('error_log')}"
        # The denial must not carry any part of the request forward.
        assert "validated_input" not in result

    def test_pre_process_admits_verified_external_caller(self, monkeypatch):
        """The same node admits a VERIFIED_EXTERNAL caller and runs execute() to success."""
        _patch_domain_emit(monkeypatch)
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.trust_level import TrustLevel

        from src.nodes.pre_process_node import PreProcessNode

        node = PreProcessNode()
        result = node(
            {
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
                "user_input": _VALID_PAYLOAD,
                "input_context": {},
                "correlation_id": "pb6-trust-admit",
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] is not None


class TestBackboneInvokeOrder:
    """PB-6 backbone: a full Graph().invoke() runs the five slots in order."""

    def _invoke(self, monkeypatch):
        _patch_domain_emit(monkeypatch)
        from framework.schemas.invocation_context import InvocationContext, TrustLevel

        from src.graph.graph import Graph, runtime_config

        agent = Graph(config=runtime_config())
        agent.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        return agent.invoke(_VALID_PAYLOAD, ctx=ctx)

    def test_backbone_invoke_succeeds_and_returns_output(self, monkeypatch):
        from framework.schemas.agent_status import AgentStatus

        result = self._invoke(monkeypatch)
        assert result.get("status") == AgentStatus.SUCCESS.value, (
            f"Expected status={AgentStatus.SUCCESS.value!r}, got: {result.get('status')!r}\n"
            f"error_log: {result.get('error_log')}"
        )
        assert result.get("output"), "output must be set after a successful invoke"
        # The assembled report must be present in the surfaced output ...
        assert "INSURANCE CLAIMS BOTTLENECK" in result["output"]
        # ... and it must be grounded: at least one knowledge-base citation
        # surfaced in the sources section, which proves the retrieve -> rerank
        # -> answer path produced evidence rather than the abstention text.
        assert "KB-" in result["output"]

    def test_backbone_node_history_matches_expected_order(self, monkeypatch):
        result = self._invoke(monkeypatch)
        history = result.get("node_history", [])
        assert history == [
            "InitializeNode",
            "PreProcessNode",
            "ClaimsRAGGraphNode",
            "PostProcessNode",
            "FinalizeNode",
        ], f"unexpected backbone node_history: {history}"

    def test_main_slot_is_claims_rag_graph_node(self):
        """The `main` backbone slot must be ClaimsRAGGraphNode (a GraphNode)."""
        from framework.nodes.graph_node import GraphNode

        from src.graph.graph import (
            ClaimsRAGGraphNode,
            InsuranceClaimsBottleneckDiscoveryAgent,
        )

        agent = InsuranceClaimsBottleneckDiscoveryAgent()
        agent.compile()
        main_node = agent._nodes.get("main")
        assert main_node is not None, "main slot must be registered"
        assert isinstance(
            main_node, ClaimsRAGGraphNode
        ), f"main slot must be ClaimsRAGGraphNode, got {type(main_node).__name__}"
        assert isinstance(main_node, GraphNode), "the main slot node must subclass GraphNode"
        assert main_node.__class__.__name__ == _MAIN_SLOT_NODE

    def test_invoke_payload_matches_pb6(self):
        """deploy/invoke_payload.json["input"] MUST equal _VALID_PAYLOAD.

        The deployment evidence invoke posts invoke_payload.json as the request
        body, so it must exercise the same payload this module asserts yields a
        grounded answer.
        """
        repo_root = Path(__file__).resolve().parents[2]
        payload_file = repo_root / "deploy" / "invoke_payload.json"
        assert payload_file.exists(), "deploy/invoke_payload.json is required for the deployment invoke"
        body = json.loads(payload_file.read_text())
        assert (
            body.get("input") == _VALID_PAYLOAD
        ), "deploy/invoke_payload.json['input'] must equal the PB-6 _VALID_PAYLOAD"
        # And the payload the deployed server forwards to agent.invoke() must
        # itself be a request shape PreProcessNode accepts.
        query_payload = json.loads(body["input"])
        assert "query" in query_payload, "invoke_payload input is missing the required field: query"
        assert query_payload["query"].strip(), "invoke_payload query must be non-empty"
