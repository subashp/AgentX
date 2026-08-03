# AgentX

AgentX turns an AMD Ryzen AI Max+ Halo machine into a private Qwen service for
chatting, coding assistance, and OpenAI-compatible inference. It includes a
local browser UI, the `agentx` coding CLI, and a gateway for other machines.

The default deployment keeps every service on your Halo machine:

- **Web UI:** a private, persistent chat interface at `http://127.0.0.1:8000/`.
- **AgentX CLI:** a coding assistant that can inspect a selected workspace.
- **OpenAI-compatible API:** `http://127.0.0.1:8000/v1` for local programs.

## Get started on a Halo machine

This is the shortest supported path for a Halo image that already includes its
ROCm stack. It downloads the vLLM container and Qwen3-14B on the first start,
then keeps both cached locally.

You need:

- A Halo Linux image where `rocminfo` sees `gfx1151`.
- Docker or Podman, and a user in the `render` group.
- At least 70 GiB of free disk space.

Clone the repository and use a virtual environment. The virtual environment is
intentional: Ubuntu prevents `pip` from changing its system Python installation.

```bash
git clone --recurse-submodules https://github.com/subashp/AgentX.git
cd AgentX
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --editable .
python -m pip install --editable third_party/AgentMemory
```

For an existing checkout, initialize third-party modules before installing:

```bash
git submodule update --init --recursive
```

Prepare the host, then start the model and local gateway:

```bash
./deploy/halo/setup.sh
./deploy/halo/start.sh
```

`start.sh` waits until the model is ready. On its first run it can take a while
to pull the container and download the model. It prints the log locations if
startup needs investigation.

Once ready, use the service locally:

```bash
# Open http://127.0.0.1:8000/ in a browser for chat.
agentx
```

At the `agentx` prompt, choose the private provider when asked, then give it a
coding task. You can also select it explicitly:

```bash
agentx --provider private-openai-compatible
```

Useful everyday commands:

```bash
./deploy/halo/status.sh  # check the model and gateway
./deploy/halo/stop.sh    # stop them
./deploy/halo/start.sh   # start them again
```

After a reboot, return to the checkout, activate the virtual environment, and
run `./deploy/halo/start.sh` again.

### What to use for what

| Need | Use |
| --- | --- |
| Private ChatGPT-style chat | Open `http://127.0.0.1:8000/` |
| Coding help in a repository | `agentx` from that repository |
| A local application or script | `http://127.0.0.1:8000/v1` |

The detailed host, model, and troubleshooting reference is in
[the Halo deployment guide](deploy/halo/README.md).

## For advanced users

### Use AgentX as a coding agent

AgentX is provider-neutral and privacy-first: it limits what a model can see
before a request leaves the CLI. The private Qwen provider has bounded,
read-only workspace tools for listing files, reading files, literal search,
Git status, scoped diffs, memory, web research, browser automation, and
bounded sub-agents. Slash commands complete in the installed CLI.

Common interactive commands:

```text
/providers                  list provider availability
/provider codex|claude|kiro|private-openai-compatible|auto
/context README.md src/agentx
/memory search qwen
/memory remember --class private "prefer local Qwen for confidential code"
/tools                      show model-callable tools for the selected provider
/execute <task>             allow bounded patch/test/shell tools for this task
/quit
```

Normal private-provider chat is read-only. `/execute <task>` adds approved
`workspace_patch`, `test_run`, and `shell_exec`, shows progress, and records
limits plus stop reason. Git staging/commit tools are explicit and never push.
The model can request up to ten read-only sub-agents; children cannot edit,
shell out, commit, or create grandchildren.

AgentX supports both standard OpenAI `tool_calls` and Qwen/vLLM raw
`<tool_call>{...}</tool_call>` blocks. Malformed or mixed prose/tool output is
rejected. Interactive private-model sessions can ask for `web_search`,
`web_fetch`, `web_fetch_document`, and optional Playwright browser tools
(`browser_open`, `browser_text`, `browser_click`, `browser_fill`,
`browser_screenshot`). AgentX asks before every internet request or browser
side effect, bounds results, blocks private-network fetches, records citations,
and keeps non-interactive plans network-free.

Install optional capabilities when needed:

```bash
python -m pip install -e ".[web,browser]"
python -m playwright install chromium
```

Press `Esc` to cancel an active private-model request or an internet-approval
prompt. Run `agentx providers list` to inspect optional local providers outside
the interactive session.

### Use the API from another local program

The gateway exposes an unauthenticated OpenAI-compatible API on loopback:

```bash
curl http://127.0.0.1:8000/v1/models
```

For example, configure an OpenAI-compatible client with:

```text
base URL: http://127.0.0.1:8000/v1
model:    Qwen/Qwen3-14B
```

The `/v1` API is stateless: the calling program sends the conversation history
it needs. The Web UI keeps named chats, summaries, and local cross-session
memory in its own local SQLite database. A machine client that needs one shared
gateway-managed conversation can use `/api/machine-chat`; see the
[deployment guide](deploy/halo/README.md#clients).

### Connect from another machine

The default service is deliberately bound to `127.0.0.1`. For a remote client,
you may create a tunnel such as `ngrok http 8000`, then configure AgentX with
the active tunnel URL:

```bash
agentx init \
  --profile private-openai-compatible \
  --endpoint https://<current-tunnel>.ngrok-free.app/v1 \
  --model Qwen/Qwen3-14B \
  --timeout 900 \
  --force
```

The current gateway has **no user authentication**. A public tunnel exposes
the Web UI, chats, machine-chat endpoint, and API. Do not expose private data
or an always-on public service until an authentication boundary is added. See
[remote access in the deployment guide](deploy/halo/README.md#optional-remote-agentx-through-ngrok).

### Configure a different OpenAI-compatible endpoint

AgentX stores settings and secrets outside the repository. To connect to a
different local or private endpoint:

```bash
agentx init \
  --profile private-openai-compatible \
  --endpoint https://model.example/v1 \
  --model <model-id> \
  --timeout 900 \
  --force
```

If the endpoint requires a key, provide the *name* of an environment variable,
never the secret itself:

```bash
agentx init \
  --profile private-openai-compatible \
  --endpoint https://model.example/v1 \
  --model <model-id> \
  --api-key-env AGENTX_MODEL_API_KEY \
  --force
```

Inspect the external settings location with `agentx config path`. The command
`agentx config show` displays the resolved non-secret configuration.

### Other providers and command-line use

When installed and authenticated, Codex, Claude Code, and Kiro CLI remain
available alongside local Qwen. Provider selection is policy-filtered before
context is sent. Common non-interactive commands are:

```bash
agentx providers list
agentx route "summarize the routing module"
agentx plan --provider private-openai-compatible --context README.md \
  "plan a documentation cleanup"
agentx --json providers list
```

`fake-local` is a deterministic offline provider for testing. Non-interactive
live execute/apply mode is not enabled; use interactive `/execute <task>` for
approved private-provider patch, test and shell tools.

### Models and operations

Qwen3-14B is the default tested Halo model. A smaller 4B profile is available
for smoke tests:

```bash
./deploy/halo/start-qwen3-4b-vllm.sh
```

The 14B launcher enables Qwen tool calling with vLLM's `qwen3_xml` parser by
default. Set `ENABLE_TOOL_CALLING=0` only for a chat-only server—AgentX's
workspace and sub-agent tools need tool calling enabled. See the
[deployment guide](deploy/halo/README.md) for environment overrides, logs,
model cache locations, and GPU prerequisites.

### Privacy and local state

AgentX records local audit artifacts for each run, including the scoped prompt,
context map, transcript, provider metadata, and outcome. Its state root holds
settings, sessions, memories, and authentication references. Use
`agentx config path` to find the exact location or set `AGENTX_HOME`,
`AGENTX_SETTINGS`, `AGENTX_SESSIONS`, `AGENTX_MEMORIES`, or `AGENTX_AUTH` to
override individual locations.

Do not commit runtime settings, model caches, chat databases, tunnel URLs, or
credentials. See the implementation for the full policy and path-redaction
contracts.

### Shared memory module

AgentX includes AgentMemory as a Git submodule under
`third_party/AgentMemory`. AgentMemory is the reusable local-first memory
library for explicit long-term memories, persona/preferences, privacy classes
(`generic`, `team`, `private`), correction/deletion, prompt assembly, and a
JSON process/API boundary for non-Python clients.

Install it from a full checkout with:

```bash
python -m pip install -e third_party/AgentMemory
```

The CLI exposes memory through `/memory ...` commands and model-callable memory
tools. `generic` memory may be sent to public providers, `team` depends on user
policy, and `private` is local-only. The Halo Web UI currently keeps a separate
gateway-local chat memory; shared AgentMemory-backed Web UI memory is not yet
implemented.

### Develop AgentX

AgentX is a Python 3.11+ package. Interactive slash commands such as
`/provider`, `/memory`, `/tools`, `/execute`, and `/quit` support completion in
the installed CLI. From a checkout with the virtual environment active:

```bash
git submodule update --init --recursive
python -m pip install -e .
python -m pip install -e third_party/AgentMemory
python -m unittest discover -s tests
```

Tests use local fixtures and do not require a live model, provider subscription,
or network connection.
