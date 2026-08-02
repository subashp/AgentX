# Halo Qwen/vLLM deployment

This directory deploys a local Qwen model on an AMD Ryzen AI Max+ Halo host
with the ROCm, Python, and vLLM runtime preconfigured by the Halo image. It
exposes three client surfaces through one loopback-only gateway. ngrok is not
required for the local Web UI or local AgentX workflow.

| Client | Endpoint | Conversation state |
| --- | --- | --- |
| Web UI | `/` | Persistent named chats, summaries, and user memory |
| External OpenAI-compatible client | `/v1/*` | Stateless; the client sends its own messages |
| Machine client | `/api/machine-chat` | One persistent shared machine conversation per user |
| AgentX CLI | `http://127.0.0.1:8000/v1` | Stateless OpenAI-compatible plan adapter |

`vLLM` itself is intentionally stateless. The gateway persists chat state in a
local SQLite database and reconstructs a bounded model prompt from a session
summary, recent messages, and cross-session user memory.

## Host prerequisites

- An AMD Halo image with its supported ROCm, Python, and vLLM prerequisites
  installed.
- AMD Halo GPU usable through ROCm (`rocminfo` sees `gfx1151`).
- `/dev/kfd` and `/dev/dri` available to the launching user; that user must be
  in the `render` group.
- Podman or Docker. The scripts default to `docker`; set
  `CONTAINER_ENGINE=podman` on hosts that expose Podman directly.
- Roughly 70 GiB free disk for the vLLM container and Qwen3-14B cache.

Run the local host check once after cloning:

```bash
cd AgentX/deploy/halo
./bootstrap.sh
```

No model weights, Hugging Face token, chat history, container storage, or ngrok
credentials are committed to this repository.

## Halo-local quickstart

From a fresh checkout, run the idempotent setup command. It validates the
preconfigured Halo runtime, installs this checkout as the `agentx` command in
the selected Python environment when needed, and merges the local Qwen
provider into the external AgentX settings file without removing existing
Codex, Claude, or Kiro configuration.

```bash
cd AgentX
./deploy/halo/setup.sh
```

Start the local model and Web UI with one command:

```bash
./deploy/halo/start.sh
```

The first launch pulls `docker.io/vllm/vllm-openai-rocm:latest` and downloads
`Qwen/Qwen3-14B` into `~/models/vllm-huggingface`. Set `HF_TOKEN` in your own
shell if Hugging Face authentication is required; never put it in this repo.

The launcher enables vLLM automatic tool calling by default with the Qwen XML
tool parser (`qwen3_xml` in vLLM 0.26). AgentX uses this for read-only workspace
inspection and private-model subagents. It normally receives OpenAI structured
tool calls and also safely handles complete raw Qwen `<tool_call>` blocks if a
vLLM response leaves them in assistant content. If you intentionally need a
plain chat-only server, set `ENABLE_TOOL_CALLING=0` before starting it; AgentX
tool calls will not work in that mode.

`start.sh` starts the existing vLLM and gateway scripts, waits for
`/v1/models`, and updates the external AgentX settings with the model ID that
the gateway advertises. It records logs and process state outside the
repository. Inspect or stop the services with:

```bash
./deploy/halo/status.sh
./deploy/halo/stop.sh
```

The model is bound to `127.0.0.1:8001`; the gateway is bound to
`127.0.0.1:8000`. Open the local Web UI at
`http://127.0.0.1:8000/`; no tunnel is needed.

Then use both surfaces locally:

```bash
agentx providers list
agentx
```

The interactive startup lists whichever of Codex, Claude, Kiro, and Qwen are
available. A missing subscription or CLI does not prevent local Qwen use, and
the setup command does not replace existing provider configuration. Use
`/provider` or `--provider` to choose explicitly; use `auto` to let AgentX
route among the providers that are available and policy-eligible.

The 4B smoke-test profile is also available:

```bash
./deploy/halo/start-qwen3-4b-vllm.sh
```

## Clients

The Web UI is available at `http://127.0.0.1:8000/`. It saves sessions in
`~/.agentx/vllm-chat.sqlite3`, retains the original transcript for display, compresses
older context, and keeps a small cross-session memory. Runtime database files
must remain local.

When a browser chat needs current public information, Qwen can request the
same bounded `web_search` and `web_fetch` tools available to AgentX. The UI
permits these requests automatically; a generic request searches DuckDuckGo
(with a Brave fallback), while a named site is fetched directly. The bounded
result is then supplied to Qwen for its answer. AgentX CLI continues to ask
for confirmation before every web request, and external/machine API clients
do not receive these browser web tools. The UI is still limited to public
HTTPS pages and bounded output, but an unauthenticated public tunnel lets any
visitor induce those requests, so do not expose it publicly until authentication
is configured.

External OpenAI-compatible clients use the standard endpoint:

```bash
curl http://127.0.0.1:8000/v1/models
```

Machine clients that want one shared persistent conversation use:

```bash
curl -N http://127.0.0.1:8000/api/machine-chat \
  -H 'Content-Type: application/json' \
  -d '{"content":"Continue the task.","model":"Qwen/Qwen3-14B","mode":"auto"}'
```

Before authentication is added, every caller is user `0`; that means public
callers share the same chats and memory. Do not expose this unauthenticated
gateway on the internet with real user data. Google authentication should
replace the request-supplied `user_id` before public use.

## AgentX integration

AgentX uses the gateway's stateless `/v1` endpoint directly. For the
Halo-local workflow, use `http://127.0.0.1:8000/v1` and keep the Web UI and
AgentX on the Halo host. The persistent `/api/machine-chat` endpoint remains a
separate chat surface; AgentX uses `/v1/chat/completions` so each coding run
has its own stateless request and local audit artifacts.

The checked-in example keeps the settings outside the repository and includes
all three optional CLI providers alongside Qwen:

```bash
mkdir -p "$HOME/.config/agentx"
cp deploy/halo/example-agentx-settings.json \
  "$HOME/.config/agentx/halo-settings.json"
export AGENTX_SETTINGS="$HOME/.config/agentx/halo-settings.json"
agentx providers list
agentx --provider private-openai-compatible
```

Provider availability is independent: a user may have Codex, Claude, Kiro,
any combination of them, or none. AgentX lists unavailable integrations and
continues to offer the configured local Qwen provider.

### Optional remote AgentX through ngrok

For a remote AgentX client, install and authenticate ngrok using the [official
ngrok download and setup instructions](https://ngrok.com/), then run this on
the Halo host:

```bash
ngrok http 8000
```

This exposes the entire gateway on port `8000`, including the Web UI, session
APIs, machine-chat endpoint, and `/v1` API. It is not an API-only tunnel and
the current gateway has no user authentication. Do not expose it to the public
internet with real conversations or private data until an authentication
boundary is configured. The local Halo workflow does not require ngrok.

Use the active tunnel URL, including `/v1`, when configuring AgentX on the
remote machine:

```bash
agentx init \
  --profile private-openai-compatible \
  --endpoint https://<current-tunnel>.ngrok-free.app/v1 \
  --model Qwen/Qwen3-14B \
  --timeout 900 \
  --force
```

The free ngrok URL can change after a restart, so update the external AgentX
settings with the current URL. Do not commit the URL, credentials, or runtime
settings to this repository.

After updating the launcher or upgrading the vLLM image, restart the model
service so the tool-calling flags take effect:

```bash
./deploy/halo/stop.sh
./deploy/halo/start.sh
```

## Operational notes

- Qwen3-14B is the tested larger dense profile. The previously attempted
  Qwen3-Coder-30B-A3B profile did not fit reliably in this ROCm/vLLM setup.
- The gateway can handle independent chat sessions concurrently. It serializes
  turns in the same session to keep their context ordered.
- The Web UI's `Auto`, `Think`, and `Fast answer` modes append Qwen's per-turn
  thinking directive; they do not require another vLLM instance.
