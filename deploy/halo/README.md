# Halo Qwen/vLLM deployment

This directory deploys a local Qwen model on an AMD Ryzen AI Max+ Halo host
with ROCm and exposes three client surfaces through one loopback-only gateway.

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

- AMD Halo GPU usable through ROCm (`rocminfo` sees `gfx1151`).
- `/dev/kfd` and `/dev/dri` available to the launching user; that user must be
  in the `render` group.
- Podman or Docker. The scripts default to `docker`; set
  `CONTAINER_ENGINE=podman` on hosts that expose Podman directly.
- Python 3.11+ for the web gateway.
- Roughly 70 GiB free disk for the vLLM container and Qwen3-14B cache.

Run the local host check once after cloning:

```bash
cd AgentX/deploy/halo
./bootstrap.sh
```

No model weights, Hugging Face token, chat history, container storage, or ngrok
credentials are committed to this repository.

## Launch

Start the model in one terminal:

```bash
cd AgentX/deploy/halo
./start-qwen3-14b-vllm.sh
```

The first launch pulls `docker.io/vllm/vllm-openai-rocm:latest` and downloads
`Qwen/Qwen3-14B` into `~/models/vllm-huggingface`. Set `HF_TOKEN` in your own
shell if Hugging Face authentication is required; never put it in this repo.

Start the gateway in a second terminal:

```bash
cd AgentX/deploy/halo
./start-vllm-web-gateway.sh
```

The model is bound to `127.0.0.1:8001`; the gateway is bound to
`127.0.0.1:8000`. If ngrok is desired, tunnel only the gateway:

```bash
ngrok http 8000
```

The 4B smoke-test profile is also available:

```bash
./start-qwen3-4b-vllm.sh
```

## Clients

The Web UI is available at `http://127.0.0.1:8000/`. It saves sessions in
`~/vllm-chat.sqlite3`, retains the original transcript for display, compresses
older context, and keeps a small cross-session memory. Runtime database files
must remain local.

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

AgentX can use the gateway's stateless `/v1` endpoint directly. Configure the
provider profile from the checkout or point it at the active ngrok tunnel:

```bash
cp example-agentx-settings.json /path/outside/the/repository/agentx-settings.json
export AGENTX_SETTINGS=/path/outside/the/repository/agentx-settings.json
agentx providers list
agentx plan --provider private-openai-compatible "review the current task"
```

For a tunnel that changes after restart, configure `--endpoint-env
AGENTX_QWEN_ENDPOINT` once and update that environment variable instead of
rewriting the settings file.

The persistent `/api/machine-chat` endpoint remains a separate chat surface;
AgentX uses `/v1/chat/completions` so each coding run has its own stateless
request and local audit artifacts.

## Operational notes

- Qwen3-14B is the tested larger dense profile. The previously attempted
  Qwen3-Coder-30B-A3B profile did not fit reliably in this ROCm/vLLM setup.
- The gateway can handle independent chat sessions concurrently. It serializes
  turns in the same session to keep their context ordered.
- The Web UI's `Auto`, `Think`, and `Fast answer` modes append Qwen's per-turn
  thinking directive; they do not require another vLLM instance.
