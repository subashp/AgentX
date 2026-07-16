# AgentX

AgentX is a provider-neutral command-line gateway for agentic coding. It routes development tasks to the best available coding agent or private model while enforcing local policy for data access, tool use, and cost.

The project is designed for developers who use multiple coding assistants and want one consistent CLI for planning, implementation, review, and automation.

The core is intended to be platform-neutral. Platform-specific paths, process launching, credential storage, and filesystem behavior should stay behind implementation abstractions so the project can be ported by developers to the environments they need.

## Goals

- Provide a single CLI for interactive and scripted agentic coding workflows.
- Route tasks across providers such as Codex, Claude Code, Kiro CLI, and OpenAI-compatible private model endpoints.
- Select both the provider and the model tier based on task complexity, required capability, latency, and cost.
- Keep provider selection policy-driven instead of hard-coded to one vendor.
- Enforce file-level data handling rules before any provider sees repository context.
- Support local or privately hosted models for sensitive code and confidential workloads.
- Preserve auditable local session context, transcripts, manifests, generated diffs, and user-controlled memories.
- Keep one reusable core that can be driven from command-line automation, an interactive CLI, or a future UI.

## Example UX

```text
agentx
agentx "fix the failing tests"
agentx -p "summarize this module"
agentx plan "refactor the scheduler"
agentx run "implement auth refresh" --mode execute
agentx run "review this PR" --provider codex
agentx providers list
agentx policy explain src/core/planner.py
agentx context save --name auth-refactor
agentx context use <session-id>
agentx memory list
agentx memory edit <memory-id>
agentx memory delete <memory-id>
```

## Architecture

```text
Command Line / Interactive CLI / Future UI
    |
    v
Command Parser
    |
    v
Agent Run Envelope
    |
    v
Policy + Context Compiler
    |
    +--> File Classification
    +--> Context Slicing
    +--> Session + Memory Selection
    +--> MCP Tool Policy
    +--> Cost / Subscription Constraints
    +--> Task Complexity Estimate
    |
    v
Provider + Model Router
    |
    +--> Codex CLI Adapter
    +--> Claude Code Adapter
    +--> Kiro CLI Adapter
    +--> OpenAI-Compatible Private Model Adapter
    +--> Private Cloud Container Adapter
    |
    v
Scoped Workspace / Sandbox / MCP Proxy
    |
    v
Session Context + Memory Store + Run Artifacts
```

## Privacy Model

AgentX treats external coding assistants as execution backends, not as the authority for privacy. The gateway decides what each backend can see.

Repository context is organized into tiers:

```text
Tier 0: task prompt and user instructions
Tier 1: public repository summary
Tier 2: non-sensitive relevant files
Tier 3: confidential or proprietary files
Tier 4: secrets, credentials, customer data
```

External providers should only receive context permitted by policy. Private hosted models can be configured for higher-classification context when the user wants local or private-cloud processing.

The policy and context compiler also controls memory exposure. Memories can be selected, summarized, excluded, or redacted before a public provider such as OpenAI, Anthropic, or another external service receives context. Editing or deleting saved memories is a local metadata operation and does not require a model call.

## Provider Types

AgentX is intended to support two provider categories:

- **Agent providers:** coding assistants or model endpoints that perform planning, editing, review, or explanation.
- **Compute providers:** local or cloud infrastructure capable of running a private model container on demand.

Example agent providers:

- Codex CLI
- Claude Code
- Kiro CLI
- Local OpenAI-compatible model server
- Private cloud-hosted OpenAI-compatible model server

Users may default to one or more public providers, or run without any self-hosted model. When no private model is configured, policy should prevent confidential workloads from being routed rather than silently relaxing privacy constraints.

Providers can expose multiple models with different cost, speed, context, and capability profiles. AgentX should maintain a model catalog and route at the model level, not only the provider level. For example, planning a risky architecture change may require a high-capability model, while test generation, documentation updates, summarization, or bounded execution can use a lower-cost model when policy and quality requirements allow it.

## Local State

AgentX keeps local state outside the project by default, with user-configurable overrides:

```text
<AGENTX_HOME>/settings.json       # or settings.yaml
<AGENTX_HOME>/sessions/           # per-session saved context
<AGENTX_HOME>/memories/           # user-editable memory files
<AGENTX_HOME>/auth/               # service-scoped authentication material
```

`AGENTX_HOME` resolves to a user-local application data directory by default. The exact platform path is an implementation detail, and users can override it globally or per run. The settings file records local paths, provider defaults, session-store location, memory-store location, policy preferences, and auth-store location.

MCP services may require authentication. AgentX should store MCP and provider authentication material under service-specific entries in the configured auth directory, using secure local storage where available.

## Policy

Project policy should be explicit and versionable. A future `.agentx/policy.toml` may define classification and routing rules:

```toml
[defaults]
external_max_classification = "internal"
private_provider = "private-local"
public_providers = ["codex", "claude"]

[model_tiers]
planning = "high"
execution = "standard"
tests = "economy"
docs = "economy"
review = "standard"

[classification]
"docs/public/**" = "public"
"tests/**" = "internal"
"src/sensitive/**" = "confidential"

[routing]
confidential = ["private-local", "private-cloud"]
internal = ["codex", "claude", "kiro", "private-local"]
public = ["codex", "claude", "kiro", "private-local"]
```

The router should first filter by privacy and availability, then choose the lowest-cost model that satisfies the task's required capability, context size, tool needs, and confidence threshold.

## Run Artifacts

Each run should produce local artifacts for auditing and reproducibility. By default, run context is saved under the configured session directory:

```text
<AGENTX_HOME>/sessions/<session-id>/
  manifest.json
  prompt.md
  context-map.json
  memory-map.json
  redactions.json
  provider.json
  transcript.jsonl
  patch.diff
  cost.json
  outcome.json
```

Run artifacts are local by default and should not be committed unless intentionally exported.

## Status

This repository is in the planning stage. The initial implementation target is a minimal CLI that can:

- Detect available providers.
- Load provider model catalogs and explain model-tier choices.
- Explain routing decisions.
- Compile a scoped context manifest.
- Run a plan-only task through one provider.
- Save local run artifacts.
