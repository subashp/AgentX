# Reusable Agent Memory Module Plan

Status: draft for review
Created: 2026-08-02
Branch: `feature/reusable-memory-plan`
Canonical implementation repo: `git@github.com:subashp/AgentMemory.git`
Prototype integration consumer: AgentX
Next integration target: `demo/` physical AI / machine-intelligence stack
Later candidate consumers: Nemmadi, other agentic runtimes

## Purpose

Build a reusable, local-first memory module for agentic systems. The module
should help agents remember recent conversations, durable user and project
preferences, persona, domain-specific long-term facts, and machine or workflow
history while preserving privacy, correction, deletion, and provider-routing
boundaries.

The module should start as its own repository from the beginning:
`git@github.com:subashp/AgentMemory.git`. AgentX is the first integration client,
not the long-term owner of the memory implementation. This keeps the API clean
enough for the `demo/` machine-intelligence repo, which is C++-heavy and
currently lacks a durable memory layer. Nemmadi already has its own
healthcare-oriented memory system, so it should be treated as a later candidate
and reference point rather than a near-term migration target.

## Product Goals

- Keep memory local by default.
- Make cloud backup or sync optional, with Supabase as the first likely backend.
- Support OpenAI, Anthropic, local vLLM, and other model providers through a
  provider-neutral prompt assembly boundary.
- Keep one shared memory system across providers. Provider policy decides what
  memory is visible to each model call, but the durable memory store is not tied
  to one provider or to localhost.
- Keep memory autonomous enough to improve after every interaction, but auditable
  and correctable by the user.
- Support short-term, long-term, persona, preference, and machine/domain memory
  without hard-coding one product domain.
- Let users inspect, correct, disable, export, import, and delete memory.
- Let users explicitly classify memories as `generic`, `team`, or `private`,
  including through natural-language tool requests such as "remember this into
  my private memory".
- Avoid storing secrets, credentials, raw private artifacts, model weights,
  datasets, or local-only paths in reusable memory records.
- Keep physical AI safety boundaries explicit: advisory memories may influence
  planning, but must not directly widen actuation authority.
- Support C++ consumers through stable schemas and a narrow C-compatible or
  process/API boundary after the Python prototype stabilizes.

## Research Baseline

The design should borrow from these patterns:

- Generative Agents: append a memory stream, retrieve relevant memories, and
  synthesize higher-level reflections back into memory.
  Source: https://arxiv.org/abs/2304.03442
- MemGPT / Letta: use tiered context management and explicit memory blocks for
  stateful agents that learn over time.
  Source: https://arxiv.org/abs/2310.08560 and https://github.com/letta-ai/letta
- MemoryBank: support continuous updates, selective forgetting/reinforcement,
  and user-personality adaptation.
  Source: https://arxiv.org/abs/2305.10250
- LangGraph: separate thread-scoped short-term memory from cross-thread
  long-term memory, and distinguish semantic, episodic, and procedural memory.
  Source: https://langchain-5e9cc07a.mintlify.app/oss/python/concepts/memory
- Mem0: treat memory as a reusable agent layer with user/session/agent levels,
  extraction, retrieval, and multi-signal ranking.
  Source: https://github.com/mem0ai/mem0
- Graphiti / Zep: represent changing facts with temporal validity,
  provenance, entity links, and hybrid retrieval.
  Source: https://github.com/getzep/graphiti
- Supabase pgvector: optional remote storage for structured records and
  embeddings when cloud sync is enabled.
  Source: https://supabase.com/docs/guides/ai/vector-columns

## Current AgentX Starting Point

AgentX already has useful pieces:

- `AgentXPaths.memories` resolves a portable memory state location.
- `MemoryRecord(id, classification, content, summary)` exists for
  provider-policy filtering.
- `MemoryStore` provides JSON CRUD under the AgentX memory directory.
- `ContextManifest` can include, summarize, redact, or exclude memories before
  provider routing.
- Run artifacts already include `memory-map.json` for audit.
- The Halo Web UI independently stores named chats, rolling session summaries,
  and one durable user-memory text blob in SQLite.

Current gaps:

- CLI does not routinely load or update stored memory.
- Halo Web UI memory is not shared with the AgentX core library.
- Persona is not a first-class model.
- Timeline is split across run transcripts, Halo messages, and sub-agent
  summaries.
- Autonomous memory updates exist only in the Halo gateway and are blob-based.
- There is no first-class correction, memory proposal, item-level provenance, or
  full deletion workflow.

## Cross-Repo Priority And Lessons

AgentX is the prototype host and first integration. The `demo/` repo is the
next target because it represents the real machine-intelligence use case and
needs memory. Nemmadi is useful research input, but not a near-term migration
priority.

Nemmadi has practical application memory:

- short-term and long-term records
- local SQLite plus Supabase sync
- token-budget-aware prompt inclusion
- delete/reset flows

Do not over-index on Nemmadi or copy its email-keyed schema directly. A reusable
library should use opaque subject IDs and optional identity metadata. Healthcare
memory policy remains useful as an example of domain-specific constraints, but
it should not drive the first implementation.

Demo has the physical AI safety shape and should be the second integration:

- memory classes
- provenance
- retention and training-use policy
- advisory versus authoritative state separation
- supersede, invalidate, retire, and review concepts
- three-level decision routing: `local/rule-based`, `local/model`, and
  `remote/model`
- remote model profiles such as `Qwen/Qwen3-14B` and `Qwen/Qwen3-4B`
- C++ runtime boundaries that need schemas and stable ABI/API contracts, not
  Python-only objects

For physical AI, this distinction is mandatory. Memory cannot become a back door
around safety, replay, or operator approval.

## Memory Privacy Classes

Memory privacy must be simple enough for users to understand and strict enough
for provider routing. The initial user-facing classes are:

| Class | Meaning | Default provider exposure |
| --- | --- | --- |
| `generic` | Safe general preferences, public project conventions, or non-sensitive usage patterns. | May be sent to public, team-approved, or local/private models within prompt budget. |
| `team` | Workgroup, company, repo, or project-specific memory that may be shared with approved team services. | May be sent to local/private models. May be sent to public or SaaS models only when team policy and user config allow it. |
| `private` | Personal, sensitive, proprietary, confidential, or machine-local memory. | Strictly local/private models only. Never sent to OpenAI, Anthropic, Kiro, or other external providers by default. |

These classes are memory visibility classes, not model providers. The durable
memory store can contain all three classes, but the prompt assembler must filter
them per provider and task.

Default exposure policy:

```text
external/public provider: generic only
external/team-approved provider: generic + team when configured
local/private provider: generic + team + private, subject to task and prompt budget
```

AgentX can keep its existing file classification policy separately. The memory
module should expose a smaller, user-facing privacy taxonomy and let AgentX map
between the two when needed.

## Memory Tools

The agent must be able to act on explicit user memory commands through tools.
Natural-language requests should resolve to tool calls rather than hidden prompt
side effects.

Examples:

- "remember this in generic memory"
- "remember this for the team"
- "remember this into my private memory"
- "that memory is wrong; correct it to ..."
- "forget what you remembered about ..."
- "show me what you remember about this project"
- "delete all my memory"

Initial tool surface:

| Tool | Purpose |
| --- | --- |
| `memory_remember` | Store an explicit memory with `privacy_class`, `scope`, `kind`, and source event refs. |
| `memory_propose` | Create an autonomous memory proposal that awaits approval or policy-based auto-apply. |
| `memory_search` | Retrieve inspectable memories by query, kind, scope, or privacy class. |
| `memory_show` | Show one memory with provenance and visibility policy. |
| `memory_correct` | Supersede or invalidate an incorrect memory and create a corrected replacement. |
| `memory_forget` | Delete one memory, a filtered set, or all memory according to deletion policy. |
| `memory_policy_explain` | Explain whether a memory can be exposed to a selected provider. |

Tool calls must be audited. The model may propose memory writes, but the
library owns validation, classification, policy enforcement, and deletion.

## Core Memory Layers

### 1. Short-Term Memory

Purpose: keep recent working context.

Default behavior:

- Keep up to `last_n_conversations`, default `10`.
- Also support per-session limits such as `last_n_turns` and character/token
  caps.
- Use direct recent messages first, then compact summaries when the context
  budget is tight.

Short-term memory is thread/session scoped. It should not be treated as durable
truth unless promoted into long-term memory.

### 2. Timeline Memory

Purpose: preserve append-only provenance.

Examples:

- user prompt
- assistant response summary
- tool calls and results
- sub-agent tasks and returned summaries
- explicit user corrections
- user preference statements
- workspace/project events
- machine observations
- mission checkpoints
- safety decisions

Timeline records are the source of truth for later derived memories.

### 3. Long-Term Memory

Purpose: preserve stable, reusable facts and summaries across sessions.

Examples:

- user prefers concise engineering updates
- user often wants incremental commits
- project uses provider-neutral routing
- physical AI should remember machine baselines, operator preferences,
  recurring failure modes, inspection findings, and validated mission outcomes
- healthcare apps may remember clinically relevant user-stated context, but that
  is a later domain profile rather than the primary design driver

Long-term memory should be derived from timeline events with source references.
Updates should use `supersedes`, `invalidates`, or `corrects` rather than silent
overwrite.

### 4. Persona And Preference Memory

Purpose: provide stable, compact guidance that should usually be present in the
model prompt.

Examples:

- user communication preferences
- user likes/dislikes
- agent response style
- project-specific workflow preferences
- physical AI operator-assist behavior
- healthcare support tone as a later domain profile

Persona should be represented as structured profile blocks, not arbitrary chat
history. Blocks can be user-editable and can have different mutability rules:

- user profile
- agent persona
- project profile
- organization/team profile
- machine/operator profile

## Prompt Assembly Model

The user's intuition is correct: memory becomes part of the model context for
each call. The memory system must not assume a local hosted model will always be
used. AgentX routing may send generic or low-sensitivity work to OpenAI,
Anthropic, Codex, Claude, or Kiro for speed, while confidential or proprietary
work may be routed to the local/private Qwen endpoint. The durable memory store
is shared across these service providers; provider policy controls which memory
items are visible to each call.

The part to control carefully is how much memory gets included and in what
order. KV/prefix cache can speed repeated stable prefixes, especially on local
vLLM, but it is not durable memory and must not be relied on for correctness,
correction, deletion, or provider portability.

The prompt assembler should treat the model context as a budgeted packet:

```text
provider/system instructions
agent runtime policy
privacy and tool policy
persona/preferences
selected long-term memories
selected short-term summary
recent messages
workspace or machine environment context
tool results
actual user or machine prompt
reserved output budget
```

Memory must not be a single unbounded blob. It should be ranked and rendered
within explicit budgets.

Suggested default priority:

1. Safety, privacy, and tool policy.
2. Current user or machine prompt.
3. Explicitly selected workspace or machine context.
4. Persona and stable preferences.
5. Highly relevant long-term memories.
6. Short-term current conversation history.
7. Older summaries.
8. Low-confidence or weakly relevant memories.

Suggested default budgets:

```json
{
  "prompt_budget": {
    "max_context_tokens": 32000,
    "reserve_output_tokens": 8192,
    "system_policy_tokens": 2500,
    "persona_tokens": 1200,
    "long_term_tokens": 4000,
    "short_term_tokens": 8000,
    "timeline_summary_tokens": 3000,
    "workspace_or_environment_tokens": 8000,
    "tool_result_tokens": 5000
  }
}
```

Provider renderers can convert the same assembled memory packet into:

- OpenAI-style messages
- Anthropic-style system plus messages
- local vLLM/OpenAI-compatible chat messages
- CLI-provider prompt text

The same retrieval result can therefore be rendered differently for each
provider while preserving one local memory source of truth.

## Configuration Model

The memory library should be controlled by a portable config file. AgentX should
have defaults, but users and downstream apps should be able to override them.

Candidate file names:

- `memory.json`
- `agent-memory.json`
- app-owned wrapper config such as `agentx-memory.json`

Example:

```json
{
  "schema_version": 1,
  "enabled": true,
  "local_first": true,
  "subject": {
    "subject_id": "default-user",
    "workspace_id": "default-workspace"
  },
  "short_term": {
    "last_n_conversations": 10,
    "last_n_turns_per_conversation": 20,
    "max_tokens": 8000,
    "summarize_when_tokens_exceed": 6000
  },
  "long_term": {
    "enabled": true,
    "max_tokens": 4000,
    "auto_update": "propose",
    "default_privacy_class": "private",
    "remember_kinds": [
      "stable_user_preference",
      "project_workflow_preference",
      "recurring_task_pattern",
      "explicit_follow_up",
      "validated_correction"
    ]
  },
  "persona": {
    "enabled": true,
    "max_tokens": 1200,
    "auto_update": "propose",
    "default_privacy_class": "private",
    "blocks": [
      "user_profile",
      "agent_persona",
      "project_profile"
    ]
  },
  "timeline": {
    "enabled": true,
    "append_only": true,
    "retention_days": 365,
    "max_event_content_chars": 12000
  },
  "domains": {
    "agentic_coding": {
      "remember_kinds": [
        "coding_style",
        "test_command",
        "development_workflow",
        "provider_preference",
        "repo_convention"
      ]
    },
    "healthcare": {
      "remember_kinds": [
        "user_stated_preference",
        "care_goal",
        "coping_strategy",
        "communication_preference"
      ],
      "requires_user_visibility": true,
      "avoid_kinds": [
        "unsupported_diagnosis",
        "sensitive_inference_without_user_statement"
      ],
      "priority": "later_reference_profile"
    },
    "physical_ai": {
      "remember_kinds": [
        "operator_preference",
        "machine_baseline",
        "inspection_finding",
        "validated_failure_mode",
        "skill_outcome",
        "mission_summary"
      ],
      "authority_boundary": "advisory_only",
      "decision_tiers": [
        "local/rule-based",
        "local/model",
        "remote/model"
      ],
      "remote_model_profiles": [
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-4B"
      ],
      "integration_target": "demo_cpp_runtime"
    }
  },
  "prompt_budget": {
    "max_context_tokens": 32000,
    "reserve_output_tokens": 8192,
    "persona_tokens": 1200,
    "long_term_tokens": 4000,
    "short_term_tokens": 8000,
    "timeline_summary_tokens": 3000
  },
  "sync": {
    "backend": "none",
    "supabase": {
      "enabled": false,
      "url_env": "MEMORY_SUPABASE_URL",
      "key_env": "MEMORY_SUPABASE_KEY",
      "use_pgvector": true
    }
  },
  "privacy": {
    "classes": [
      "generic",
      "team",
      "private"
    ],
    "default_classification": "private",
    "external_provider_allowed_classes": [
      "generic"
    ],
    "team_provider_allowed_classes": [
      "generic",
      "team"
    ],
    "local_provider_allowed_classes": [
      "generic",
      "team",
      "private"
    ],
    "require_explicit_user_class_for_direct_remember": true,
    "allow_external_provider_memory": true,
    "store_raw_prompts": true,
    "store_tool_results": "summaries_only",
    "training_use": "prohibited"
  },
  "deletion": {
    "hard_delete": true,
    "delete_embeddings": true,
    "delete_remote_copies": true,
    "keep_redacted_audit_tombstone": false
  }
}
```

Applications should be able to provide additional domain configs without
changing the core library.

## Data Model Draft

### MemoryEvent

Append-only timeline event.

Fields:

- `event_id`
- `subject_id`
- `workspace_id`
- `agent_id`
- `session_id`
- `event_type`
- `content`
- `summary`
- `source`
- `created_at`
- `actor`
- `privacy_class`: `generic`, `team`, or `private`
- `retention`
- `training_use`
- `metadata`

### MemoryRecord

Derived long-term fact or summary.

Fields:

- `memory_id`
- `subject_id`
- `workspace_id`
- `memory_kind`
- `memory_class`
- `content`
- `summary`
- `confidence`
- `valid_from`
- `valid_until`
- `source_event_ids`
- `supersedes`
- `invalidates`
- `created_at`
- `updated_at`
- `privacy_class`: `generic`, `team`, or `private`
- `retention`
- `training_use`
- `metadata`

### PersonaBlock

Structured profile or behavior block.

Fields:

- `block_id`
- `subject_id`
- `workspace_id`
- `block_type`
- `title`
- `content`
- `schema`
- `read_only`
- `auto_update`
- `source_event_ids`
- `updated_at`
- `privacy_class`: `generic`, `team`, or `private`

### MemoryProposal

Autonomous update candidate.

Fields:

- `proposal_id`
- `operation`: `create`, `update`, `supersede`, `invalidate`, `delete`
- `target_id`
- `candidate`
- `reason`
- `source_event_ids`
- `risk_level`
- `auto_apply_eligible`
- `status`: `pending`, `accepted`, `rejected`, `applied`

## Autonomous Update Policy

After every interaction:

1. Append timeline events for user input, assistant output, tool calls, sub-agent
   results, and explicit corrections.
2. Run a memory distiller in the background.
3. Generate memory proposals.
4. Auto-apply only low-risk proposals allowed by config.
5. Require user approval for sensitive persona, privacy, policy, healthcare, or
   physical-AI authority-adjacent changes.
6. Persist source references so every memory can explain why it exists.

Initial AgentX default should be conservative:

```json
{
  "long_term": { "auto_update": "propose" },
  "persona": { "auto_update": "propose" }
}
```

Once tests and UX are solid, low-risk coding preferences can move to
`auto_update: "apply_low_risk"`.

## Correction And Deletion

Correction is not just editing text. It is a lifecycle event.

Required operations:

- `memory list`
- `memory show <id>`
- `memory search <query>`
- `memory remember --class generic <text>`
- `memory remember --class team <text>`
- `memory remember --class private <text>`
- `memory correct <id> <replacement>`
- `memory forget <id>`
- `memory forget --scope <subject|workspace|agent|all>`
- `memory export`
- `memory import`
- `memory sync status`

Correction behavior:

- append a `user_correction` timeline event
- mark old memory as superseded or invalidated
- create a replacement memory with source reference to the correction
- prevent invalidated memory from retrieval unless historical lookup is requested

Deletion behavior:

- delete raw memory content
- delete summaries
- delete embeddings
- delete derived graph/entity edges
- delete synced remote copies when configured
- optionally keep a redacted tombstone only if audit policy requires it

Worst-case user control:

```text
agentx memory forget --all --hard
```

The API should expose the same capability for Web UI and downstream apps.

## Storage Backends

### Phase 1: Local SQLite

Use SQLite as the first durable backend:

- portable across Windows, macOS, Linux, and mobile-adjacent environments
- works offline
- supports transactions
- can use FTS5 for lexical search
- simple to test deterministically

Tables:

- `memory_subjects`
- `memory_sessions`
- `memory_events`
- `memory_records`
- `persona_blocks`
- `memory_proposals`
- `memory_sync_state`

### Phase 2: Optional Embeddings

Add an optional embedding provider interface:

- no runtime dependency by default
- local embedding model support later
- remote embedding model support only when configured
- embedding records must be deleted when the source memory is deleted

### Phase 3: Supabase Sync

Supabase should be an optional backup/sync backend:

- Postgres tables mirror the local schema
- pgvector stores embeddings when enabled
- RLS restricts rows by user/workspace/subject
- tombstones support deletion propagation
- sync conflict policy starts as last-writer-wins only for low-risk records
- corrections and invalidations should be append/supersede, not destructive
  overwrite

## Development Workflow

Use the same agentic development workflow as the AgentX work:

- The main agent owns architecture, sequencing, review, integration, and final
  acceptance.
- Worker agents own bounded implementation slices with clear file ownership.
- Test/documentation agents own focused fixtures, regression tests, docs, and
  validation commands.
- Worker agents should run their own focused tests and create green,
  self-contained commits in their working branch or fork when the boundary is
  clear.
- The main agent reviews worker diffs, validates the claimed tests, resolves
  integration conflicts, and merges or rebases the accepted commits.
- Each phase should end with an execution-log entry covering files changed,
  tests run, failures, fixes, and residual risk.
- Do not accept an agent summary as completion evidence without inspecting the
  files, diffs, and test output.

For the standalone `AgentMemory` repo, this implies small phase branches and
independent commits before AgentX integration begins.

## Standalone Repository Plan

### AM-001: Repository Foundation

Goal: initialize `git@github.com:subashp/AgentMemory.git` as the canonical
implementation repository.

Gate:

- package scaffold exists
- README states local-first and provider-neutral goals
- MIT license or chosen license is present
- deterministic test runner works
- no AgentX-specific imports

### AM-002: Architecture, Config, And Schemas

Goal: add this plan, config schema, JSON schemas, and ADRs for reusable memory
boundaries.

Gate:

- plan reviewed
- no runtime behavior changed
- config validates `generic`, `team`, and `private` memory classes

### AM-003: Library Skeleton

Goal: add core models and interfaces without depending on AgentX.

Gate:

- deterministic model validation tests
- no provider dependency
- tool request models cover explicit remember/correct/forget flows

### AM-004: SQLite Backend

Goal: local persistent timeline, records, persona blocks, and proposals.

Gate:

- CRUD tests
- transaction tests
- hard-delete tests
- Windows/macOS/Linux path-safe tests

### AM-005: Prompt Budget Assembler

Goal: build provider-neutral memory packets within config budgets.

Gate:

- tests for priority ordering
- tests for max token/char budget
- tests for provider-specific rendering
- tests that `private` memory is omitted from external-provider prompt packets

### AM-006: Memory Distiller

Goal: generate memory proposals from completed interactions.

Gate:

- deterministic fake model fixture
- no autonomous write without policy approval path
- tests for domain-specific remember and avoid rules, including physical-AI
  authority boundaries

### AM-007: Memory Tool Runtime

Goal: expose remember/search/show/correct/forget/export/import operations as
library APIs and optional CLI commands.

Gate:

- CLI tests
- tool-call tests for "remember this into my private memory"
- deletion non-recall test
- correction supersession test

### AM-008: Supabase Backend

Goal: optional backup/sync backend.

Gate:

- config-driven only
- no Supabase dependency in base install
- local-only mode remains default
- sync delete propagation tested with fake backend

### AM-009: Demo C++ Integration Boundary

Goal: define and prove the integration path for the `demo/` C++ machine
intelligence stack.

Gate:

- stable JSON schemas for events, records, persona blocks, proposals, and prompt
  memory packets
- C-compatible or process/API boundary documented
- decision-tier metadata supports `local/rule-based`, `local/model`, and
  `remote/model`
- physical-AI advisory-only memory policy is enforced in fixtures
- Qwen 14B/4B remote-model profiles are represented as model/provider metadata,
  not hard-coded into memory logic

### AM-010: Split-Out Readiness

Goal: confirm the package can be consumed by AgentX and later by `demo/`.

Gate:

- package has no AgentX-specific imports
- public docs reviewed
- API versioning policy exists
- release artifacts can be installed locally

## AgentX Integration Plan

### AX-MEM-001: Add AgentMemory Dependency

Goal: add the standalone AgentMemory package as a dependency, editable checkout,
or submodule once AM-001 through AM-007 are green.

Gate:

- AgentX imports only public AgentMemory APIs
- AgentX tests can run without network or Supabase

### AX-MEM-002: AgentX CLI Integration

Goal: append interaction events and retrieve budgeted memory for provider calls.

Gate:

- private provider sees relevant memory
- external provider redaction still holds
- `private` memory reaches only local/private providers
- `generic` memory may reach external providers when policy permits
- `team` memory exposure follows user/team config
- no memory sent when config disables it

### AX-MEM-003: AgentX Memory Tools

Goal: expose AgentMemory operations through AgentX's model/tool loop.

Gate:

- model can call `memory_remember` for explicit user requests
- model can call `memory_correct` and `memory_forget` through approved flows
- direct user commands remain available without a model call

### AX-MEM-004: Halo Web UI Integration

Goal: replace gateway-local blob memory with shared memory service.

Gate:

- existing sessions migrate
- Web UI can show/edit/delete memory
- machine-chat uses same prompt assembler

## Open Decisions

- Package name: `agent-memory`, `agentx-memory`, `memcore`, or another name.
- Initial package import name inside the standalone repo.
- Whether AgentX consumes AgentMemory as an editable dependency, git submodule,
  or package release during early development.
- Whether memory extraction should use the active user-selected model or a
  configured cheaper distiller model.
- Whether memory extraction should run locally by default even when the main
  routed provider is external.
- Default config file name and search path.
- Whether cloud sync should use Supabase directly or a generic sync backend API
  with Supabase as one implementation.
- Whether external providers may receive any persona/preference memory by
  default. Conservative default should be `generic` only. `team` memory requires
  explicit team/provider policy. `private` memory remains local/private only.

## Risks

- Memory can personalize incorrectly if inferred preferences are treated as
  facts. Mitigation: require source references, confidence, and correction UX.
- Memory can leak sensitive data to external providers. Mitigation: reuse
  AgentX classification and memory exposure policy.
- Memory can bloat prompts. Mitigation: strict budgets, priority ordering, and
  provider-specific rendering.
- Autonomous updates can amplify stale or wrong assumptions. Mitigation:
  proposal-first workflow and invalidation semantics.
- Physical AI memory can become unsafe if treated as authoritative. Mitigation:
  memory classes and advisory-only defaults.
- C++ integration can be blocked if the first API is too Python-specific.
  Mitigation: define schemas first and keep the runtime API serializable across
  process or C-compatible boundaries.
- Cloud sync can complicate deletion. Mitigation: tombstones, sync-state audit,
  and tests that deletion removes remote content and embeddings.

## Initial Acceptance Criteria

The first implementation is acceptable when:

- a user can run AgentX across several sessions and have it remember explicit
  stable preferences;
- the user can inspect what was remembered;
- the user can correct a wrong memory and the old version is not retrieved as
  current truth;
- the user can delete one memory or all memory;
- prompt assembly respects configured budgets;
- external providers receive only policy-allowed memory;
- local-only mode works without network, Supabase, or embedding dependencies;
- the core memory package can be imported without AgentX CLI or Halo gateway
  dependencies.
