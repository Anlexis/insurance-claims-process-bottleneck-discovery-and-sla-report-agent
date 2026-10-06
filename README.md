# Claims Process Bottleneck & SLA Report Agent

AI agent for discovering claims process bottlenecks and reporting SLA performance, built with Agentic Star.

> **Category**: Cat 2 (a domain pipeline — a fixed sequence of steps that answers one kind of question)
> **Industry**: Insurance
> **Template ID**: INS-C2-020

## Overview

Answers claims-operations questions — where the process bottleneck is, which service-level
threshold is breaking, what to change first — from a curated knowledge base of process
playbooks, service-level references, process-analysis notes and regulatory guidance, and says
so plainly when the knowledge base does not cover the question.

Given a question such as *"what is the primary bottleneck in claims settlement, and which
service-level thresholds are being breached?"*, the agent retrieves the relevant passages, keeps
only the ones that clear a relevance floor, and returns a plain-text report that cites every
passage it used.

The question can be narrowed to a process step or a service-level metric. Both filter
vocabularies are closed and published, and an unsupported value is refused rather than quietly
dropped — a silently ignored filter would let you read an un-narrowed answer as a narrowed one.

It is deliberately a **retrieval-grounded answering** agent, not a measurement engine: it does
not compute metrics from an event log and does not issue a binding operational finding. When no
passage clears the floor it abstains, returning an explicit "no grounded evidence, refine the
question" answer rather than an unsupported opinion. The report passes an output boundary that
withholds it entirely if credential-shaped material is found in it, and personal data in the
question is masked before the question is rendered into the report.

The bundled knowledge base is a small worked example (intake and triage, adjuster assignment,
service-level thresholds, bottleneck discovery, deviation and rework, settlement, regulatory
obligations, operational indicators, recommendations). Replace it with your own corpus and
retriever — the pipeline and its contracts do not change.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from the platform's package registry. |
| Python | >= 3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode: if the platform is unreachable or the installed framework version does not match, start-up
fails rather than bringing up a partially working agent. This is intentional — a half-running
agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, graphs, services, schemas)
tests/        unit and boundary tests
config/       registration manifest and runtime parameters
prompts/      the grounding contract for the answer step
docs/         design and operational documentation
```

See `docs/02_design.md` for the architecture and `docs/03_test_spec.md` for what the test suite
covers.

## Customising

1. `config/agent.yaml` is the registration manifest (identity, entry point, required caller
   trust); `config/config.yaml` holds the runtime parameters — the retry budget and the
   retrieval tuning (`rag.top_k`, `rag.score_threshold`, `rag.hybrid_search`). Every declared
   value is validated for type, finiteness and range before it is used; an invalid entry is
   dropped and the node falls back to its documented default.
2. Replace the knowledge base in `src/nodes/retrieve_node.py` with your own corpus and
   retriever. The node contract (a question in, scored passages out) does not change. Each
   document carries the retrieval facets it covers, and a test asserts that the published filter
   vocabulary and the documents' facets agree — change the two together.
3. `prompts/claims_rag.j2` states the grounding contract for the answer step.
4. Review the caller-data contract in `src/nodes/pre_process_node.py` and
   `src/services/service.py`, and the output boundary in `src/nodes/post_process_node.py`,
   before you widen either.
5. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
