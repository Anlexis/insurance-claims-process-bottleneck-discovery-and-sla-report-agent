# Design Specification — INS-C2-020

A domain agent that answers claims-operations questions — where the process
bottleneck is, which service-level thresholds are breached, what to do about it
— grounded strictly in a curated claims-operations knowledge base. It answers
from cited evidence or it says it cannot; it never speculates.

## Position in the framework

| Aspect | Value |
|---|---|
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Agent class | `InsuranceClaimsBottleneckDiscoveryAgent` |
| Category | 2 — a domain pipeline built from a general retrieval pattern |
| Composition | two-layer nested: a `GraphNode` in the `main` slot delegating to an inner `BaseGraph` |
| Entry point | `src.graph.graph.InsuranceClaimsBottleneckDiscoveryAgent` (declared in `config/agent.yaml`) |
| Trust level required of a caller | `VERIFIED_EXTERNAL` |
| Generation mode | deterministic — no model is invoked |

Three concerns are kept apart: the state is a flat `TypedDict` of primitives and
JSON strings, each node is a `FunctionNode` overriding only `execute()`, and the
graphs compose nodes without containing domain logic themselves.

## Architecture

The fixed five-slot backbone is never modified. The `main` slot is a
`GraphNode` (`ClaimsRAGGraphNode`) that delegates the whole domain workflow to
an inner `BaseGraph` (`DomainWorkflowGraph`), so domain complexity is
encapsulated one layer down.

```
START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
                                       |        \ (retry, budget from config/config.yaml)
                                       |         -> pre_process
                                       v
                       ClaimsRAGGraphNode (main slot)
                                       v
        DomainWorkflowGraph (inner graph):
          input_validate -> retrieve -> rerank_filter -> generate_answer -> output_format
```

### Outer backbone

| Slot | Responsibility | Reads | Writes | Class |
|------|----------------|-------|--------|-------|
| initialize | session, correlation, trust level | user_input | session_id, trust level | framework default |
| pre_process | trust boundary + caller-data contract | user_input, input_context | validated_input, enriched_context | `PreProcessNode` |
| main | delegate to the inner workflow | validated_input, input_context | rag_report, generated_answer, answer_citations, reranked_count | `ClaimsRAGGraphNode` |
| post_process | output boundary; release or withhold | rag_report and the other output-bearing fields | formatted_output, result | `PostProcessNode` |
| finalize | response metadata, timings | result | response_metadata | framework default |

`pre_process` requires `VERIFIED_EXTERNAL`; every other node is `ANONYMOUS`,
because the trust boundary is enforced once and the inner nodes must accept the
context that passes through the subgraph boundary.

### Inner workflow

| Node | Responsibility | Reads | Writes |
|------|----------------|-------|--------|
| input_validate | the retrieval contract: question + published facets | validated_input | retrieval_query |
| retrieve | score the knowledge base; apply facets and the candidate cap | retrieval_query, runtime_settings | retrieved_documents, retrieved_count |
| rerank_filter | sort by relevance; apply the relevance floor | retrieved_documents, runtime_settings | reranked_documents, reranked_count |
| generate_answer | assemble a grounded answer, or abstain | reranked_documents, retrieval_query, runtime_settings | generated_answer, answer_citations |
| output_format | render the report | generated_answer, answer_citations, retrieval_query, input_context | rag_report, result |

### Two things cross the outer/inner boundary explicitly

The framework invokes a subgraph as `subgraph.invoke(user_input, session_id=,
ctx=)`. Neither the caller's `input_context` nor the graph's configuration is
forwarded, and a node takes no per-invocation configuration argument. Both
therefore travel through the inner graph's initial state:

- **`input_context`** is stashed by `ClaimsRAGGraphNode.extract_input()` — the
  last template hook that sees the outer state — and picked up by
  `DomainWorkflowGraph._extra_initial_state()`. The hand-off is a `ContextVar`
  (`src/graph/context_bridge.py`), so concurrent invocations in one process
  cannot see each other's context.
- **`runtime_settings`** is the already-validated subset of
  `config/config.yaml`, produced by `ClaimsRAGGraphNode._parent_config()`.

## Configuration

`config/agent.yaml` is the flat registration manifest: identity, entry point,
trust level, and the compile-time `requires` block. It carries no runtime
values — a runtime value placed there is read by nothing.

`config/config.yaml` holds every runtime parameter, and is the file the platform
registry loads and passes as `Graph(config=...)`. The standalone HTTP entry
point reads the same file, so a directly deployed agent and a registry-loaded
agent behave identically.

| Key | Consumer | Bounds | Behaviour when invalid |
|---|---|---|---|
| `max_retry` | `AgentBaseGraph.route()` | non-negative integer below the framework ceiling | the framework raises at compile time |
| `timeout_s` | framework configuration contract | — | no consumer in this template |
| `llm.system_prompt_template` | `GenerateAnswerNode` | a path inside the repository | reported unresolved in the audit event |
| `rag.top_k` | `RetrieveNode` | 1–50 | falls back to 8 |
| `rag.score_threshold` | `RerankFilterNode` | 0.0–1.0 | falls back to 0.3 |
| `rag.hybrid_search` | `RetrieveNode` | boolean | falls back to off |
| `security.s3_gate_enabled` | posture record | — | the output boundary runs unconditionally |

Every declared number is parsed by the same finite-and-bounded parser the caller
inputs use. A non-finite value is the reason: `NaN` parses through `float()` and
compares `False` against every bound, so an unchecked `score_threshold` would
silently drop every retrieved passage — or keep every one — with nothing
surfacing.

## The caller contract

`POST /invoke` takes three fields.

- **`input`** — the question, either as plain text or as a JSON object
  `{"query": ..., "filters": {...}}`. Capped at 4000 characters and refused
  rather than truncated: silently answering a different question than the one
  asked is worse than saying no.
- **`filters`** — the retrieval facets. Both the key set and each key's value
  set are closed and published (`process_step`, `sla_metric`); an unsupported
  key or value is REFUSED, not dropped. A silently ignored filter is the worst
  outcome, because the caller reads a narrowed answer as if it had been
  narrowed. Every value the contract offers is carried by at least one document
  — a test asserts it, so the contract cannot drift from the knowledge base.
- **`input_context`** — structured invocation parameters. The only supported key
  is `channel`, an inert lowercase token that renders in the report's
  request-scope line. Unknown keys are dropped at the adapter, not merely
  ignored downstream: the framework's first node copies `input_context`
  verbatim into its own result and the output gate scans every value of that
  result, so an undeclared key carrying credential-shaped text would fail the
  whole request at node one with an error the caller cannot act on. The adapter
  screens the surviving fields with the framework's own credential detector and
  refuses with HTTP 400 naming the field, never the value.

Caller text is screened for instruction-override content in the node that owns
the contract, not only by whatever gate happens to run in front of it. The
screen covers chat-template control tokens as a class (`<|...|>`, `[INST]`,
`<<SYS>>`) as well as directive phrases, and it runs over the parsed structure
including mapping keys, so an escaped payload cannot slip through. It is probed
in both directions: attack forms refused, and ordinary claims-operations
sentences containing the same verbs ("can an adjuster override the SLA
threshold…") accepted.

Personal data is masked in the question before it is normalised, because the
question is rendered verbatim into the report. The mask uses explicit character
guards rather than word boundaries: a word boundary is computed over word
characters, which include Kanji and Kana, and Japanese is written without
spaces — so a personal number written the way a claims handler writes it sits
directly against a Kanji and a boundary-anchored pattern never matches it.
Business identifiers (document ids, claim references) and figures are
deliberately outside the mask: the report exists to carry them.

## The output boundary

`PostProcessNode` screens the assembled report and every other output-bearing
field, then either releases the report or withholds it.

Two properties it exists to hold:

1. **The screen is a superset of the framework's own detector.** The framework
   re-scans every value of every node result and raises on a match; a raise is
   caught by the node wrapper, which returns a bare error partial and discards
   whatever this node decided. A pattern the framework knows and this node does
   not is therefore not a smaller net — it is a bypass of the containment
   below. The framework's detector is called directly so the two sets cannot
   drift.
2. **On a violation it clears every output-bearing field.** The framework
   resolves the caller's output as `formatted_output or result`, with no status
   check, so leaving the ungated report in state would ship it inside the error
   envelope. The withheld notice is deliberately non-empty: an empty
   `formatted_output` is falsy and re-opens that same fallback.

The screen walks nested structures depth-first, because the report's material
also rides in the structured citation list and a scan of top-level strings only
would report zero findings on exactly the payload shape that carries one.

**Rounding grid: not applicable.** This template renders no monetary aggregate
and performs no numeric rewriting anywhere. That is also why layer ordering is
not a hazard here: identifiers and figures reach the output boundary
byte-identical, so a pattern scan at the boundary still sees what the pipeline
produced.

## Retrieval

The knowledge base is embedded, curated, and tagged with the published facets.
Scoring is lexical token overlap between the question and each document's title
and body — deterministic and dependency-free. With `hybrid_search` on, a
question that names a document's identifier or source retrieves it outright: a
reference code shares no prose tokens with the passage that defines it, so
lexical overlap alone scores it zero.

A deployment that wires a managed document index replaces `_CLAIMS_KB` and
`_score`; the facet contract, the settings, and everything downstream are
unchanged.

## Design decisions

| Decision | Alternative | Chosen | Rationale |
|---|---|---|---|
| Base class | AutonomousBaseGraph | AgentBaseGraph | a fixed retrieval workflow, not an autonomous loop |
| Composition | flat single layer | nested two layers | the domain workflow is five steps; the backbone stays untouched |
| Unsupported filter | drop it | refuse it | a dropped filter makes the caller read an un-narrowed answer as narrowed |
| Unusable question | truncate | refuse | answering a different question than the one asked is worse than refusing |
| Violating output | redact | withhold | a redaction that misses one form ships it; withholding fails closed |
| Personal data | refuse the request | mask and answer | the question is still answerable; the value simply must not be rendered |
