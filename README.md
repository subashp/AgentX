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
clients, and provides a shared machine conversation endpoint separate from
AgentX's stateless coding-run integration. See [the Halo deployment guide](deploy/halo/README.md) for
hardware prerequisites, startup, client endpoints, and security constraints.

The Halo-local path does not require ngrok: the Web UI and AgentX can both use
the loopback gateway on the Halo host. Halo supplies the ROCm, Python, and
vLLM prerequisites; the checked-in scripts pull the vLLM image and download
Qwen model artifacts on first launch. Codex, Claude Code, and Kiro CLI are
optional local integrations. When present, they remain selectable alongside
the private Qwen provider; when absent, Qwen can be used by itself.

The shortest Halo-local path is:

```sh
./deploy/halo/setup.sh
./deploy/halo/start.sh
agentx
```

This keeps the Web UI and AgentX on the Halo host. See the deployment guide
for optional remote access through a manually installed ngrok tunnel.

## AgentX quickstart

For a general AgentX checkout, install the `agentx` console command once:

```sh
python -m pip install --editable .
```

Then inspect the available providers and enter the interactive workflow:

```sh
agentx providers list
agentx
```

Use subcommands when you want a non-interactive workflow:

```sh
agentx providers list
agentx route "summarize the routing module"
agentx plan --context README.md "plan a documentation cleanup"
agentx config path
```

Use `--json` before a subcommand for machine-readable output. The module form
`python -m agentx` remains a fallback when the `agentx` console script is not on
your `PATH`; the documented interface is the `agentx` command.

Run `agentx` without a subcommand to enter the provider-aware interactive CLI:

```sh
agentx
```

AgentX checks Codex, Claude Code, Kiro CLI, and the configured private model at
startup. It then lets you choose a default provider for the session. That
provider remains selected for subsequent tasks until you use `/provider` to
change it:

```sh
agentx interactive --provider codex
agentx --provider claude
```

Inside the session, enter a task at the `agentx[provider]>` prompt. Use
`/provider auto`, `/providers`, `/context`, `/help`, or `/quit` to control the
session. `/context` sets the relative files or directories that the next task
may use; `/context clear` removes that selection:

```text
agentx[private-openai-compatible]> /context README.md src/agentx
Context paths: README.md, src/agentx
agentx[private-openai-compatible]> analyze the selected code
```

The private OpenAI-compatible adapter exposes bounded, read-only workspace
tools for tree listing, file reads, literal search, Git status, and scoped Git
diffs. When the model requests one, AgentX executes it locally and sends the
bounded result back through the provider's tool-call loop. Selected `/context`
paths become the tool scope; sensitive files, credentials, repository state,
and traversal paths remain blocked.

Private-provider runs can also create and inspect child agents through
`subagent_create`, `subagent_list`, and `subagent_get`. AgentX allows at most
ten children per parent. Each child gets its own session/artifact directory,
provider interaction, and explicitly selected context, then returns a summary
to the parent. Children run at depth one and do not receive subagent tools, so
they cannot create grandchildren.
Codex, Claude Code, Kiro CLI, configured OpenAI-compatible endpoints, and the
deterministic `fake-local` provider use the same read-only plan boundary and
local AgentX audit artifacts.

For the private OpenAI-compatible provider, AgentX streams the assistant's
response as it arrives. If the endpoint returns a separate reasoning/thinking
field, it is shown in dim grey terminal text before the response. The complete
response and reasoning are still saved in the run artifacts. `/quit` is the
interactive exit command.

Approval-gated patch and shell tool implementations are available as an
explicit adapter boundary for controlled integrations. They are not enabled
by default: patches require allowed paths and approval, while shell commands
require approval and are executed as an argv list without a shell.

If the private model endpoint is missing, AgentX prints a startup warning with
the external settings-file path and an initialization command. The default
settings file is outside the repository; inspect its resolved location with:

```sh
agentx config path
```

## Provider Usage

AgentX can inspect provider availability, explain a route, and run the live
provider workflows exposed by the current CLI. Provider selection is policy-
filtered before a provider receives context.

### Codex

Initialize a Codex profile, then run a read-only plan against a scoped
workspace:

```sh
agentx init --profile codex --codex-command codex --force
agentx providers list
agentx plan --provider codex --context README.md \
  "review the README and propose documentation improvements"
```

The Codex command must be installed and authenticated separately. AgentX stores
the plan transcript and policy artifacts under the configured local state root;
it does not apply source changes in plan mode.

### Claude

Claude Code can be selected for a read-only plan when its CLI is installed and
authenticated:

```sh
agentx providers list
agentx plan --provider claude --context README.md \
  "review the authentication module for design risks"
```

AgentX invokes Claude Code in print/plan mode from a policy-scoped workspace;
it does not enable file edits through this plan path.

### Kiro CLI

Kiro CLI can be selected when `kiro-cli` is installed and logged in:

```sh
agentx providers list
agentx plan --provider kiro --context README.md \
  "review the current implementation and identify risks"
```

AgentX invokes Kiro's non-interactive chat mode with read-only filesystem
access for the plan workflow.

### Self-hosted models

For the AMD Halo deployment, use the [Halo deployment guide](deploy/halo/README.md).
It configures the local Qwen provider automatically and keeps the Web UI and
AgentX on the same machine. No ngrok tunnel is needed for that workflow.

For another OpenAI-compatible endpoint, configure the endpoint and model in
the external AgentX settings file:

```sh
agentx init \
  --profile private-openai-compatible \
  --endpoint https://model.example/v1 \
  --model <model-id> \
  --timeout 900 \
  --force
```

For a remote Halo endpoint, use the active ngrok URL in place of
`https://model.example/v1`. Install and authenticate ngrok from its [official
setup page](https://ngrok.com/), then run `ngrok http 8000` on Halo. This
exposes the entire gateway, including the Web UI, session APIs, machine-chat
endpoint, and `/v1` API—so do not expose it to the public internet with real
private data until authentication is added. Free ngrok URLs can change after a
restart; keep the updated endpoint only in external settings.

The settings file is outside the repository. Inspect its location with
`agentx config path`, then verify the provider and route:

```sh
agentx providers list
agentx plan --provider private-openai-compatible \
  "summarize the private planner implementation"
agentx --provider private-openai-compatible
```

The endpoint must already be running; AgentX does not provision arbitrary model
weights, start arbitrary model servers, or manage cloud compute. If an
OpenAI-compatible endpoint requires authentication, configure only the name of
an environment variable with `--api-key-env`; the secret itself is read at
runtime and is not written to settings or run artifacts:

```sh
agentx init \
  --profile private-openai-compatible \
  --endpoint https://<current-tunnel>.ngrok-free.app/v1 \
  --model Qwen/Qwen3-14B \
  --api-key-env AGENTX_QWEN_API_KEY \
  --timeout 900 \
  --force
```

### Automatic routing across providers

Use `auto` to inspect the providers that pass availability and privacy policy
filters:

```sh
agentx route --provider auto --mode review --explain \
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
- `plan --provider claude`: run a live Claude Code print/plan request against a
  scoped read-only workspace.
- `plan --provider kiro`: run a live Kiro CLI non-interactive plan with
  read-only filesystem access.
- `plan --provider private-openai-compatible`: run a live plan through a
  configured local or remote OpenAI-compatible endpoint.
- `execute --fake`: run the deterministic offline execute workflow, validate any
  adapter patch output, and write local artifacts without applying source
  mutations.
- `run --fake`: run the lower-level deterministic fake adapter path.
- `config path`: show the resolved AgentX state paths.
- `config show`: show resolved settings.

Live execution is exposed for Codex, Claude Code, Kiro CLI, and the configured
OpenAI-compatible plan adapters. Execute/apply mode remains fake-only and never
applies source mutations.

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
`agentx config path` to inspect the resolved
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
  "public_providers": ["codex", "claude", "kiro"],
  "private_provider": "private-openai-compatible",
  "external_max_classification": "internal",
  "providers": {
    "private-openai-compatible": {
      "endpoint": "https://example.invalid",
      "model": "Qwen/Qwen3-14B",
      "timeout": 900,
      "enabled": true
    }
  }
}
```

Provider IDs are configuration data. Do not assume one public provider is the
only viable execution path.

The CLI can write common profiles:

```sh
agentx init
agentx init --profile codex --codex-command codex --force
agentx init --profile private-openai-compatible \
  --endpoint http://127.0.0.1:8000/v1 --model Qwen/Qwen3-14B --force
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
