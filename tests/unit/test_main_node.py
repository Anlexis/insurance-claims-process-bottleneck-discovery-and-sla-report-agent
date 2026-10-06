# MainNode is the reference main-slot node used by the flat composition sample
# in src/examples/. It is not part of the claims pipeline — this agent's main
# slot is ClaimsRAGGraphNode — so what is asserted here is the node contract
# itself: the declared trust level, the execute(state) -> partial dict shape,
# and the audit event every boundary node owes.

import pytest
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.main_node import MainNode


@pytest.fixture(autouse=True)
def _quiet_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.main_node.emit_trace_event", lambda *a, **k: None)


class TestNodeContract:
    def test_subclasses_the_framework_node_base(self):
        assert issubclass(MainNode, FunctionNode)

    def test_declares_its_trust_level_explicitly(self):
        assert "required_trust_level" in MainNode.__dict__
        assert MainNode.required_trust_level is TrustLevel.ANONYMOUS

    def test_execute_takes_state_and_returns_a_partial_dict(self):
        result = MainNode().execute({"correlation_id": "unit"})
        assert isinstance(result, dict)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["result"]
        # Partial-dict contract: only the keys this node writes.
        assert set(result) == {"status", "result"}

    def test_emits_a_domain_audit_event(self, monkeypatch):
        events: list[str] = []
        monkeypatch.setattr(
            "src.nodes.main_node.emit_trace_event",
            lambda event_type, *a, **k: events.append(event_type),
        )
        MainNode().execute({"correlation_id": "unit"})
        assert events == ["main_node_reference_invoked"]


class TestTrustGateAroundExecute:
    """Driven through __call__, which is where the framework's trust gate runs."""

    def test_anonymous_caller_is_admitted(self):
        result = MainNode()({"caller_trust_level": TrustLevel.ANONYMOUS.value, "correlation_id": "unit"})
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_the_gate_runs_before_execute(self, monkeypatch):
        """A node that required more trust than the caller holds must not run
        execute() at all — the denial is not a post-hoc filter."""

        class _Privileged(MainNode):
            required_trust_level = TrustLevel.INTERNAL

        called: list[bool] = []
        monkeypatch.setattr(_Privileged, "execute", lambda self, state: called.append(True) or {})
        result = _Privileged()({"caller_trust_level": TrustLevel.ANONYMOUS.value, "correlation_id": "unit"})
        assert called == []
        assert result["status"] == AgentStatus.ERROR.value
        assert any("trust gate" in entry.lower() for entry in result["error_log"])
