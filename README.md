# AgentX

AgentX is a provider-neutral, privacy-first command-line gateway for agentic
coding workflows. It routes coding tasks through local policy boundaries before
any provider receives context, and it records deterministic local artifacts for
auditing.

The current implementation is a standard-library Python package for Python 3.11
and newer. It has no runtime package dependencies today.

AgentX is built so provider choice stays policy-driven rather than hard-coded to
one commercial backend. Public CLIs, private model endpoints, and future compute
backends are modeled as adapters behind the same routing and artifact contracts.

## Halo local-model deployment

The repository includes a reproducible AMD Halo/ROCm vLLM deployment under
[`deploy/halo`](deploy/halo/). It launches Qwen3-14B locally, provides a
persistent browser chat UI, proxies the OpenAI-compatible API for external
clients, and reserves a shared machine conversation endpoint for future AgentX
CLI integration. See [the Halo deployment guide](deploy/halo/README.md) for
hardware prerequisites, startup, client endpoints, and security constraints.

## Quickstart

From a source checkout, make `src` importable first. You can do that with an
editable install, or by setting `PYTHONPATH` to `src` in your shell. The examples
below assume `src` is already importable.

```sh
python -m agentx init
python -m agentx providers list
python -m agentx route "summarize the routing module"
python -m agentx plan --context README.md "plan a documentation cleanup"
python -m agentx init --profile codex --force
python -m agentx plan --provider codex --context README.md "plan a documentation cleanup"
python -m agentx execute --fake --allowed-patch README.md "try an offline execute run"
python -m agentx config path
python -m agentx config show
```

Use `--json` before the command for machine-readable output:

```sh
python -m agentx --json route "summarize the routing module"
python -m agentx --json config path
```

The installed console script exposes the same commands as `agentx` when the
package is installed in an environment:

```sh
agentx providers list
agentx plan --fake "plan with the deterministic local adapter"
```

Run `agentx` without a subcommand to enter the provider-aware interactive CLI:

```sh
python -m agentx
```

AgentX lists configured providers and lets you choose `auto`, `codex`, Claude,
or another available provider before accepting coding tasks. You can also fix
the provider for the session:

```sh
python -m agentx interactive --provider codex
python -m agentx --provider claude
```

Inside the session, enter a task at the `agentx[provider]>` prompt. Use
`/provider auto`, `/providers`, `/help`, or `/quit` to control the session.
Codex and the deterministic `fake-local` provider run their currently exposed
plan workflows. Other providers currently return routing explanations until
their live adapters are exposed through the public CLI.

## Provider Usage

AgentX can inspect provider availability, explain a route, and run the live
provider workflows exposed by the current CLI. Provider selection is policy-
filtered before a provider receives context.

### Codex

Initialize a Codex profile, then run a read-only plan against a scoped
workspace:

```sh
python -m agentx init --profile codex --codex-command codex --force
python -m agentx providers list
python -m agentx plan --provider codex --context README.md \
  "review the README and propose documentation improvements"
```

The Codex command must be installed and authenticated separately. AgentX stores
the plan transcript and policy artifacts under the configured local state root;
it does not apply source changes in plan mode.

### Claude

Claude Code can be discovered and included in route explanations when its CLI
is installed and enabled in settings:

```sh
python -m agentx providers list
python -m agentx route --provider claude --mode review --explain \
  "review the authentication module for design risks"
```

The current public CLI does not yet execute live Claude plans. In this phase,
the Claude adapter is represented by provider discovery and routing contracts;
`plan --provider claude` will report that live Claude execution is not yet
supported.

### Self-hosted models, such as Qwen3-Coder

Self-hosted models can be represented through an OpenAI-compatible `/v1` chat
endpoint. Configure the endpoint in an AgentX settings document, keeping the
file and any authentication material outside the source checkout:

```json
{
  "public_providers": ["codex", "claude"],
  "private_provider": "private-openai-compatible",
  "external_max_classification": "internal",
  "providers": {
    "private-openai-compatible": {
      "endpoint": "http://127.0.0.1:8000/v1",
      "enabled": true
    }
  }
}
```

Point AgentX at that settings file and inspect the endpoint and policy route:

```sh
export AGENTX_SETTINGS=/path/to/agentx-settings.json
python -m agentx providers list
python -m agentx route --provider private-openai-compatible --explain \
  "summarize the private planner implementation"
```

The endpoint must already be running; AgentX does not provision model weights,
start a Qwen3-Coder server, or manage cloud compute in this release. The
OpenAI-compatible adapter exists as a provider contract, but live private
endpoint execution is not yet exposed through the public CLI.

### Automatic routing across providers

Use `auto` to inspect the providers that pass availability and privacy policy
filters:

```sh
python -m agentx route --provider auto --mode review --explain \
  "review the current change and identify the least expensive suitable route"
```

The routing library supports model profiles with economy, standard, and high
tiers and can select the lowest-cost eligible model when a `ModelCatalog` is
provided. The current CLI does not yet load a persistent cost catalog, so its
automatic route command is a policy and availability dry run rather than a
complete cost-optimized provider execution workflow. Cost-aware live routing
will become the default once model catalog configuration and the remaining
provider adapters are exposed through the CLI.

## Current CLI

The public CLI currently supports:

- `init`: write first-run settings under the resolved AgentX state root.
- `interactive` or `shell`: enter a provider-aware interactive task session.
- `providers list`: inspect configured provider availability.
- `route`: explain provider and model-tier routing without running a provider.
- `plan --fake` or `plan --provider fake-local`: run the deterministic offline
  plan workflow and write local artifacts.
- `plan --provider codex`: run a live Codex CLI plan against a scoped read-only
  workspace built from policy-visible context.
- `execute --fake`: run the deterministic offline execute workflow, validate any
  adapter patch output, and write local artifacts without applying source
  mutations.
- `run --fake`: run the lower-level deterministic fake adapter path.
- `config path`: show the resolved AgentX state paths.
- `config show`: show resolved settings.

Live execution is exposed only for Codex plan mode in this phase. Execute/apply
mode remains fake-only and never applies source mutations.

## Privacy Model

AgentX treats providers as execution backends, not as the authority for privacy.
AgentX decides what a provider can see before a request crosses the adapter
boundary.

Implemented privacy controls include:

- Policy classification for repository paths using `public`, `internal`,
  `confidential`, `proprietary`, and `secret` levels.
- Provider eligibility filtering based on classification, public-provider
  defaults, private-provider settings, and explicit routing rules.
- Context manifest compilation for external and private provider classes.
- Path redaction and summary placeholders for withheld context.
- Memory exposure decisions that can include, summarize, redact, or exclude
  memories before provider visibility.
- A private OpenAI-compatible adapter that uses Python standard-library HTTP
  primitives and rejects base URLs containing credentials.
- MCP per-run config generation with service visibility, tool allowlists and
  denylists, auth path references, sanitized endpoints, and argument/result
  redaction primitives.
- Scoped workspace path normalization that rejects absolute paths, traversal,
  duplicate aliases, and case-ambiguous aliases.
- Execute-mode patch validation for allowed paths, denied paths, invalid patch
  paths, and configured secret markers.
- A fake offline flow for routine validation without live model calls, provider
  subscriptions, cloud compute, or network access.

## Local State

AgentX keeps local state under an AgentX state root. The exact default location
is platform-specific and intentionally treated as an implementation detail. Use
`agentx config path` or `python -m agentx config path` to inspect the resolved
paths for the current environment.

The state root is organized generically as:

```text
<AgentX state root>/
  settings.json
  sessions/
  memories/
  auth/
```

Environment overrides can redirect the state root or individual state areas:

- `AGENTX_HOME`: AgentX state root.
- `AGENTX_SETTINGS`: settings document path.
- `AGENTX_SESSIONS`: run/session artifact directory.
- `AGENTX_MEMORIES`: local memory record directory.
- `AGENTX_AUTH`: service-scoped authentication material directory.

Run artifacts are local by default and should not be committed unless
intentionally exported. Plan and execute runs write artifacts such as
`manifest.json`, `prompt.md`, `context-map.json`, `memory-map.json`,
`redactions.json`, `provider.json`, `transcript.jsonl`, `patch.diff`,
`cost.json`, and `outcome.json` under a session directory.

For private demos or local experiments, keep AgentX state outside the source
checkout. For example, set `AGENTX_HOME` to a private state directory and store
provider defaults, sessions, memories, and auth material there. Run
`agentx init` for an AgentX-only fake-local profile, or
`agentx init --profile codex --codex-command <command>` for a Codex plan profile.
Do not create or commit a repository-local `.agentx` directory for this setup.

## Provider Model

AgentX currently models these provider categories:

- CLI adapters for coding assistants exposed as local commands.
- Local or private-cloud OpenAI-compatible model endpoints.
- Deterministic fake local adapters for offline tests and examples.

The default provider registry can report availability for Codex CLI, Claude
Code, Kiro CLI, and a private OpenAI-compatible endpoint. Provider status is
based on configured settings, command discovery, endpoint configuration, and
optional auth or subscription checks.

Cloud compute providers and live private model lifecycle management are future
or optional wiring. Public documentation should describe them as adapter
contracts until user-facing commands exist.

## Configuration

Settings may be JSON, or simple YAML for supported scalar and list fields. The
resolved settings include:

```json
{
  "public_providers": ["codex", "claude"],
  "private_provider": "private-openai-compatible",
  "external_max_classification": "internal",
  "providers": {
    "private-openai-compatible": {
      "endpoint": "https://example.invalid",
      "enabled": true
    }
  }
}
```

Provider IDs are configuration data. Do not assume one public provider is the
only viable execution path.

The CLI can write common profiles:

```sh
python -m agentx init
python -m agentx init --profile codex --codex-command codex --force
```

`init` refuses to overwrite an existing settings file unless `--force` is
passed.

## Development

Run tests from a checkout with `src` importable:

```sh
python -m unittest discover -s tests
```

Routine tests use deterministic fixtures and fakes. They should not require live
provider credentials, subscriptions, cloud infrastructure, or network access.
