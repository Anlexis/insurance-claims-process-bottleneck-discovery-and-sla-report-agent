# Test Specification — INS-C2-020

## 1. Test strategy

- **Agent:** INS-C2-020 — grounded answers about claims-process bottlenecks and
  service-level compliance. Category 2, retrieval pattern, two-layer nested
  graph (outer `AgentBaseGraph` backbone + inner `DomainWorkflowGraph`).
- **Pipeline:** validate → retrieve → rerank/filter → grounded answer (with
  abstention) → report assembly → output boundary. All evidence comes from the
  embedded claims-operations knowledge base (`RetrieveNode._CLAIMS_KB`); the
  scorer is deterministic and is the seam a managed document index replaces.
- **Test types:** unit (each node and each caller-contract helper, driven
  directly), proof-of-boundary (framework contracts, AST scans, invoke order),
  and end-to-end through the real HTTP entry point.
- **What is proved where.** Unit tests call `execute()` — and the service
  helpers — with no framework wrapper in front, so each node owns its own
  guarantee rather than inheriting it from whatever runs ahead of it. The
  end-to-end module drives the real ASGI app, which is the only place the
  questions in §4 can be answered at all: a declared configuration value
  reaching the pipeline, the caller's context crossing the subgraph boundary,
  and the output boundary containing a violation are all properties of the
  assembled system.
- **Framework provisioning:** `framework` (agenticstar-agentcore) is installed
  from the platform's package registry by the pipeline. Tests import the real
  modules; there are no stub nodes.
- **Audit events:** `emit_trace_event` is patched at node module level in unit
  tests to avoid contacting an audit backend — never through a `sys.modules`
  stub, which would break the real `shared` package the framework imports.

### Test file map

| File | Scope |
|------|-------|
| `tests/unit/test_caller_contract.py` | The caller contract helpers: inert alphabet, finite bounded parser, instruction-override screen, structural screen, filter vocabulary, personal-data mask |
| `tests/unit/test_nodes.py` | Every node's `execute()` plus the knowledge-base/published-contract consistency checks |
| `tests/unit/test_main_node.py` | The reference main-slot node used by the flat composition sample |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | TC-06/TC-07: the framework security gates cannot be replaced |
| `tests/proof_of_boundary/test_invoke_e2e.py` | PB-8: end-to-end behaviour through the real HTTP entry point |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6: per-node invoke order, trust gate, backbone order, payload alignment |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4: no platform-internal imports (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2/PB-5: state is checkpoint-safe and carries no credential-named field (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7: skip stub — no cross-boundary human-in-the-loop propagation |

### Canonical request payload

The request used by the backbone invoke test and by
`deploy/invoke_payload.json` — the two must stay identical, asserted by
`test_invoke_payload_matches_pb6`:

```json
{
  "query": "What is the primary bottleneck in the insurance claims settlement process, and which service-level thresholds are being breached?",
  "filters": {
    "process_step": "settlement",
    "sla_metric": "settlement_time"
  }
}
```

It retrieves documents that clear the relevance floor, so the answer step
returns cited evidence rather than the abstention text and the assembled report
surfaces `KB-` sources.

## 2. Framework compliance

| TC-ID | Test | Expected result | Where |
|-------|------|-----------------|-------|
| TC-01 | State is a flat `TypedDict` with `NotRequired` domain fields; no model classes | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | An empty or unusable question is refused at both the outer and the inner boundary | `status=error`, error_log populated, no partial output carried forward | `TestPreProcessNode`, `TestInputValidateNode` |
| TC-03 | No credential-named field in state | AST scan + credential gate: 0 violations | `test_state_safety.py` |
| TC-04 | Node contract is `execute(self, state) -> dict`, returning only written keys | signature and partial-dict shape asserted | `test_main_node.py`, `TestInvokeOrder` |
| TC-05 | Every boundary node emits at least one domain audit event | ≥1 domain event inside each `execute()` | node modules + the audit-trace gate |
| TC-06 | The default input gate cannot be replaced | `TypeError` at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-07 | The default output gate cannot be replaced | `TypeError` at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | The trust gate runs before `execute()` | ANONYMOUS caller refused, VERIFIED_EXTERNAL admitted, `execute()` never entered on refusal | `TestTrustGate`, `TestTrustGateAroundExecute` |
| TC-09 | Outer `pre_process` requires VERIFIED_EXTERNAL; every other node is ANONYMOUS | trust level asserted per node | `TestPreProcessNode`, `TestInputValidateNode` |

## 3. Proof-of-boundary

| PB-ID | Boundary | Expected result | Where |
|-------|----------|-----------------|-------|
| PB-2 | State serialization | primitives and JSON strings only | `test_state_safety.py` |
| PB-4 | Import isolation | 0 platform-internal imports under `src/` | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named field, no prohibited annotation | `test_state_safety.py` |
| PB-6 | Per-node invoke order | `node_start → input gate → execute() → output gate → node_complete` for every node under `src/nodes/` | `TestInvokeOrder` |
| PB-6b | Backbone order | `[InitializeNode, PreProcessNode, ClaimsRAGGraphNode, PostProcessNode, FinalizeNode]` | `TestBackboneInvokeOrder` |
| PB-6c | Real external caller | `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` — never `for_internal()` | `TestBackboneInvokeOrder` |
| PB-6d | Payload alignment | `deploy/invoke_payload.json["input"]` equals the canonical payload | `test_invoke_payload_matches_pb6` |
| PB-7 | Human-in-the-loop propagation | skipped: `ClaimsRAGGraphNode.propagate_hitl` is false and there is no interrupt checkpoint | `test_pb7_hitl_interrupt_propagation.py` |
| PB-8 | The deployed HTTP path | see §4 | `test_invoke_e2e.py` |

## 4. End-to-end behaviour (PB-8, real ASGI `/invoke`)

| E2E-ID | Question the test answers | Expected result |
|--------|---------------------------|-----------------|
| E2E-01 | Can the agent serve a request at the trust level its manifest declares? | Bearer caller gets `status=success` and a report citing `KB-SLA-03` |
| E2E-02 | What happens with no caller credential configured? | trust gate denies; `status=error` and an empty output — never an ungated report |
| E2E-03 | Does the report depend on the question? | two questions produce different reports, each citing what its own question retrieved |
| E2E-04 | Does the agent abstain rather than speculate? | an off-domain question returns the abstention text and no citation |
| E2E-05 | Do the caller's retrieval facets narrow retrieval? | a faceted request cites strictly fewer sources and renders the facet in the request-scope line |
| E2E-06 | Is an unsupported facet refused, not ignored? | unknown key and out-of-vocabulary value both fail closed |
| E2E-07 | Does a value declared in `config/config.yaml` change what the caller receives? | `score_threshold`, `top_k` and `hybrid_search` each visibly change the report |
| E2E-08 | Does an invalid declared value fall back safely? | an out-of-range `score_threshold` yields the shipped behaviour, not a disabled filter |
| E2E-09 | Does the caller's context cross the subgraph boundary? | `channel=ops_portal` appears in the rendered report; absent context renders `channel=unknown` |
| E2E-10 | Is the structured channel closed and inert? | unknown keys dropped at the adapter; free text refused inside the graph |
| E2E-11 | Is a credential-shaped context value refused readably? | HTTP 400 naming `input_context.channel`, never echoing the value |
| E2E-12 | Does the refusal set match the framework's block set? | parametrized property: refused exactly when `detect_credentials_in_value` fires |
| E2E-13 | Are instruction-override payloads refused end to end? | control tokens, directive phrases and a hostile field name all fail closed and are never echoed |
| E2E-14 | Does the screen leave real work alone? | ordinary claims-operations sentences containing the same verbs still succeed |
| E2E-15 | Is personal data kept out of the report? | values written directly against Kanji are masked; business identifiers are untouched |
| E2E-16 | Does the output boundary contain a violation? | withheld notice is truthy; secret, passage, document, traceback and source paths all absent |
| E2E-17 | Clean-path control | the same request without the injected fault still answers, so the containment assertions cannot pass vacuously |

## 5. Negative and boundary cases

| Case | Node | Expected |
|------|------|----------|
| empty / non-string question | PreProcessNode | `status=error`; no `validated_input` |
| question over 4000 characters | PreProcessNode | `status=error` (refused, never truncated) |
| JSON body that is not a request shape | PreProcessNode, InputValidateNode | `status=error` |
| unsupported filter key or value | PreProcessNode, InputValidateNode | `status=error`, field named, value never echoed |
| control token in a filter KEY | PreProcessNode, InputValidateNode | `status=error` |
| missing `retrieval_query` | RetrieveNode | `status=error` |
| non-finite or out-of-range `top_k` / `score_threshold` | RetrieveNode, RerankFilterNode | falls back to the documented default |
| candidate with a non-finite score | RerankFilterNode | sorts last and is dropped |
| empty candidate list | RerankFilterNode | `status=success`, `reranked_count=0` |
| missing `reranked_documents` | GenerateAnswerNode | `status=error` |
| declared prompt path outside the repository | GenerateAnswerNode | reported unresolved; nothing is read |
| missing `generated_answer` | OutputFormatNode | `status=error` |
| non-inert channel reaching the renderer | OutputFormatNode | rendered as `unknown`, caller text never printed |
| empty report | PostProcessNode | truthy fallback notice, `status=success` |
| credential material in any output-bearing field | PostProcessNode | withheld notice, `status=error`, every output-bearing field cleared |

## 6. Output invariant

This template renders no monetary aggregate, so there is no rounding grid to
enforce and no numeric rewriting anywhere in the pipeline. The invariants the
report does carry, and which the tests enforce, are:

1. every rendered finding comes from a document listed in the sources section,
   and the abstention text is rendered when there is no such document;
2. the relevance figure printed beside each source is the score retrieval
   computed, printed unchanged;
3. only inert tokens are rendered into the request-scope line;
4. document identifiers and figures reach the output boundary byte-identical —
   nothing rewrites them, so a pattern scan at the boundary still sees what the
   pipeline produced.

## 7. Execution

- `pytest tests/` against the framework wheel the pipeline installs.
- Repository gates: dependency pinning, stub check, category consistency,
  manifest schema, credential scan, trust level, forbidden strings, audit
  trace, licence, and structural integrity.
- PB-7 ships as a skip stub by design; every other test executes.
