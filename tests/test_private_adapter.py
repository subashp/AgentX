import json
import shutil
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agentx.adapters import AdapterError, AdapterRequest, execute_adapter_run
from agentx.config import AgentXPaths
from agentx.cli import _PrivateSubagentRunner
from agentx.memory import AgentMemoryTools, call_memory_tool
from agentx.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleChatClient,
    OpenAICompatibleClientError,
)
from agentx.routing import AgentRun
from agentx.store import RUN_ARTIFACT_FILES, SessionStore
from agentx.subagents import SubagentManager, SubagentTask, SubagentTools
from agentx.tools import CompositeToolExecutor, ReadOnlyWorkspaceTools


class PrivateAdapterFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.fixture_root = Path("tests") / ".tmp_private_adapter"
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)
        self.fixture_root.mkdir(parents=True)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingOpenAIHandler)
        self.server.requests = []
        self.server.response_status = 200
        self.server.stream_events = None
        self.server.response_bodies = None
        self.server.response_body = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Private adapter completed.",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        }
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)

    @property
    def base_url(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def make_paths(self):
        root = self.fixture_root / "state"
        return AgentXPaths(
            root=root,
            settings=root / "settings.json",
            sessions=root / "sessions",
            memories=root / "memories",
            auth=root / "auth",
        )


class OpenAICompatibleAdapterTests(PrivateAdapterFixtureTestCase):
    def test_cancel_active_request_interrupts_a_blocking_read(self):
        class BlockingResponse:
            def __init__(self):
                self.read_started = threading.Event()
                self.released = threading.Event()
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                self.read_started.set()
                self.released.wait(timeout=2)
                if self.closed:
                    raise OSError("response closed")
                return b'{"choices":[{"message":{"role":"assistant","content":"late"}}]}'

            def close(self):
                self.closed = True
                self.released.set()

        response = BlockingResponse()
        client = OpenAICompatibleChatClient(
            base_url="http://127.0.0.1:8000",
            model="local-coder",
            opener=lambda request, timeout=None: response,
        )
        cancelled = threading.Event()
        errors = []

        def invoke():
            try:
                client.create_chat_completion(
                    [{"role": "user", "content": "Wait"}],
                    cancel_event=cancelled,
                )
            except OpenAICompatibleClientError as exc:
                errors.append(exc)

        worker = threading.Thread(target=invoke)
        worker.start()
        self.assertTrue(response.read_started.wait(timeout=1))
        cancelled.set()
        client.cancel_active_request()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(1, len(errors))
        self.assertEqual("cancelled", errors[0].error_type)

    def test_success_posts_chat_completion_payload_and_parses_usage(self):
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            api_key="test-secret-key",
            timeout=5.0,
        )
        run = AgentRun(
            prompt="Plan AX-009",
            mode="plan",
            provider="private-openai-compatible",
            model_tier="standard",
            context_paths=("src/agentx/openai_compatible.py",),
            task_hints=("do not edit docs",),
            required_tools=("unittest",),
        )

        result = adapter.execute(AdapterRequest(run=run))

        self.assertEqual("success", result.status)
        self.assertEqual("private-openai-compatible", result.provider_id)
        self.assertEqual("local-coder", result.model_id)
        self.assertEqual("Private adapter completed.", result.outcome["summary"])
        self.assertEqual("", result.patch)
        self.assertEqual(
            {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            result.cost["usage"],
        )
        self.assertEqual(11, result.cost["input_tokens"])
        self.assertEqual(7, result.cost["output_tokens"])
        self.assertEqual(18, result.cost["total_tokens"])

        self.assertEqual(1, len(self.server.requests))
        request = self.server.requests[0]
        self.assertEqual("/v1/chat/completions", request["path"])
        self.assertEqual("Bearer test-secret-key", request["headers"].get("Authorization"))
        self.assertEqual("true", request["headers"].get("Ngrok-Skip-Browser-Warning"))
        payload = request["json"]
        self.assertEqual("local-coder", payload["model"])
        self.assertFalse(payload["stream"])
        self.assertEqual(["system", "user"], [message["role"] for message in payload["messages"]])
        self.assertIn("use web_fetch for a user-named website", payload["messages"][0]["content"])
        user_content = payload["messages"][1]["content"]
        self.assertIn("Mode: plan", user_content)
        self.assertIn("Context paths:\n- src/agentx/openai_compatible.py", user_content)
        self.assertIn("Task hints:\n- do not edit docs", user_content)
        self.assertIn("Required tools:\n- unittest", user_content)
        self.assertIn("User prompt:\nPlan AX-009", user_content)
        self.assertNotIn("test-secret-key", json.dumps(result.as_dict()))
        self.assertNotIn("test-secret-key", json.dumps(payload))

    def test_auth_header_is_omitted_when_api_key_is_not_supplied(self):
        adapter = OpenAICompatibleAdapter(base_url=self.base_url, model="local-coder")

        result = adapter.execute(AdapterRequest(run=AgentRun(prompt="No auth", provider="private-openai-compatible")))

        self.assertEqual("success", result.status)
        self.assertNotIn("Authorization", self.server.requests[0]["headers"])

    def test_success_preserves_reasoning_content_separately_from_response(self):
        self.server.response_body["choices"][0]["message"] = {
            "role": "assistant",
            "reasoning_content": "The greeting is simple and needs a direct answer.",
            "content": "Hello from Qwen.",
        }
        adapter = OpenAICompatibleAdapter(base_url=self.base_url, model="local-coder")

        result = adapter.execute(
            AdapterRequest(
                run=AgentRun(prompt="Hello", provider="private-openai-compatible")
            )
        )

        self.assertEqual("Hello from Qwen.", result.outcome["summary"])
        self.assertEqual("Hello from Qwen.", result.outcome["response"])
        self.assertEqual(
            "The greeting is simple and needs a direct answer.",
            result.outcome["thinking"],
        )

    def test_streaming_posts_stream_payload_and_emits_reasoning_and_content(self):
        self.server.stream_events = [
            {"choices": [{"delta": {"reasoning_content": "Think first. "}}]},
            {"choices": [{"delta": {"reasoning_content": "Then answer."}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " from Qwen."}}]},
            {"usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}},
        ]
        events: list[tuple[str, str]] = []
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            stream=True,
            stream_callback=lambda kind, value: events.append((kind, value)),
        )

        result = adapter.execute(
            AdapterRequest(
                run=AgentRun(prompt="Hello", provider="private-openai-compatible")
            )
        )

        self.assertEqual("success", result.status)
        self.assertTrue(result.outcome["streamed"])
        self.assertEqual("Hello from Qwen.", result.outcome["response"])
        self.assertEqual("Think first. Then answer.", result.outcome["thinking"])
        self.assertEqual(
            ["thinking", "thinking", "content", "content", "complete"],
            [kind for kind, _ in events],
        )
        self.assertEqual("Hello from Qwen.", "".join(value for kind, value in events if kind == "content"))
        self.assertTrue(self.server.requests[0]["json"]["stream"])

    def test_streaming_splits_inline_think_tags_across_chunks(self):
        self.server.stream_events = [
            {"choices": [{"delta": {"content": "<thi"}}]},
            {"choices": [{"delta": {"content": "nk>reasoning "}}]},
            {"choices": [{"delta": {"content": "continues</think>"}}]},
            {"choices": [{"delta": {"content": "Final answer."}}]},
        ]
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            stream=True,
        )

        result = adapter.execute(
            AdapterRequest(
                run=AgentRun(prompt="Hello", provider="private-openai-compatible")
            )
        )

        self.assertEqual("reasoning continues", result.outcome["thinking"])
        self.assertEqual("Final answer.", result.outcome["response"])

    def test_tool_loop_executes_read_only_tool_and_returns_follow_up(self):
        (self.fixture_root / "README.md").write_text("tool-visible fixture", encoding="utf-8")
        self.server.response_bodies = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-tree-1",
                                    "type": "function",
                                    "function": {
                                        "name": "workspace.tree",
                                        "arguments": '{"path":"","max_entries":10}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The workspace contains the requested fixture.",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            },
        ]
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            tool_executor=ReadOnlyWorkspaceTools(self.fixture_root),
        )

        result = adapter.execute(
            AdapterRequest(
                run=AgentRun(
                    prompt="Inspect the workspace",
                    provider="private-openai-compatible",
                    required_tools=("workspace.tree",),
                )
            )
        )

        self.assertEqual("success", result.status)
        self.assertEqual("The workspace contains the requested fixture.", result.outcome["summary"])
        self.assertEqual(["workspace.tree"], result.outcome["tools_used"])
        self.assertEqual(2, len(self.server.requests))
        first_payload = self.server.requests[0]["json"]
        self.assertEqual("auto", first_payload["tool_choice"])
        self.assertEqual(
            {"workspace_tree", "workspace_read", "workspace_search", "git_status", "git_diff"},
            {tool["function"]["name"] for tool in first_payload["tools"]},
        )
        second_messages = self.server.requests[1]["json"]["messages"]
        self.assertEqual("assistant", second_messages[2]["role"])
        self.assertEqual("tool", second_messages[3]["role"])
        self.assertEqual("call-tree-1", second_messages[3]["tool_call_id"])
        self.assertIn('"path": "README.md"', second_messages[3]["content"])
        self.assertEqual(40, result.cost["usage"]["total_tokens"])

    def test_tool_loop_executes_raw_qwen_tool_call_and_returns_follow_up(self):
        (self.fixture_root / "README.md").write_text("tool-visible fixture", encoding="utf-8")
        self.server.response_bodies = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<tool_call>\n"
                                '{"name":"workspace_tree","arguments":{"path":"","max_entries":10}}\n'
                                "</tool_call>"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The workspace contains the requested fixture.",
                        }
                    }
                ]
            },
        ]
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            tool_executor=ReadOnlyWorkspaceTools(self.fixture_root),
        )

        result = adapter.execute(
            AdapterRequest(
                run=AgentRun(
                    prompt="Inspect the workspace",
                    provider="private-openai-compatible",
                )
            )
        )

        self.assertEqual("success", result.status)
        self.assertEqual(["workspace_tree"], result.outcome["tools_used"])
        self.assertEqual("The workspace contains the requested fixture.", result.outcome["summary"])
        self.assertEqual(2, len(self.server.requests))
        second_messages = self.server.requests[1]["json"]["messages"]
        self.assertEqual("assistant", second_messages[2]["role"])
        self.assertIsNone(second_messages[2]["content"])
        raw_call = second_messages[2]["tool_calls"][0]
        self.assertEqual("workspace_tree", raw_call["function"]["name"])
        self.assertEqual(
            {"path": "", "max_entries": 10},
            json.loads(raw_call["function"]["arguments"]),
        )
        self.assertEqual("tool", second_messages[3]["role"])
        self.assertEqual("agentx-qwen-tool-call-1", second_messages[3]["tool_call_id"])

    def test_tool_loop_executes_memory_search_and_returns_follow_up(self):
        paths = self.make_paths()
        remembered = call_memory_tool(
            paths,
            "memory_remember",
            {"content": "User prefers compact engineering updates.", "privacy_class": "generic"},
        )
        self.server.response_bodies = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-memory-1",
                                    "type": "function",
                                    "function": {
                                        "name": "memory_search",
                                        "arguments": '{"query":"engineering updates"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Memory says the user prefers compact engineering updates.",
                        }
                    }
                ]
            },
        ]
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            tool_executor=AgentMemoryTools(paths, user_prompt="What information do you have about me in memory?"),
        )

        result = adapter.execute(AdapterRequest(run=AgentRun(prompt="What do you remember?", provider="private-openai-compatible")))

        self.assertEqual("success", result.status)
        self.assertEqual(["memory_search"], result.outcome["tools_used"])
        second_messages = self.server.requests[1]["json"]["messages"]
        self.assertIn(remembered["memory_id"], second_messages[3]["content"])

    def test_memory_tool_rejects_mutation_without_user_intent(self):
        paths = self.make_paths()
        tools = AgentMemoryTools(paths, user_prompt="What do you know about me?")

        result = tools.call(
            "memory_remember",
            {"content": "secret inferred preference", "privacy_class": "private"},
        )

        self.assertFalse(result.ok)
        self.assertIn("requires explicit user intent", result.error)

    def test_tool_loop_rejects_malformed_raw_qwen_tool_call(self):
        self.server.response_body["choices"][0]["message"] = {
            "role": "assistant",
            "content": "<tool_call>{not-json}</tool_call>",
        }
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            tool_executor=ReadOnlyWorkspaceTools(self.fixture_root),
        )

        result = adapter.execute(
            AdapterRequest(
                run=AgentRun(
                    prompt="Inspect the workspace",
                    provider="private-openai-compatible",
                )
            )
        )

        self.assertEqual("failure", result.status)
        self.assertIn("valid JSON", result.outcome["summary"])
        self.assertEqual(1, len(self.server.requests))

    def test_tool_loop_creates_child_and_returns_child_summary(self):
        class FixtureRunner:
            def __init__(self):
                self.calls = []

            def run(self, task, *, session_id, depth):
                self.calls.append((task, session_id, depth))
                return {
                    "status": "success",
                    "summary": f"Child completed: {task.prompt}",
                    "artifact_root": f"tests/.tmp/{session_id}",
                }

        runner = FixtureRunner()
        manager = SubagentManager(parent_session_id="parent", runner=runner)
        tools = CompositeToolExecutor(
            ReadOnlyWorkspaceTools(self.fixture_root),
            SubagentTools(manager),
        )
        self.server.response_bodies = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-child-1",
                                    "type": "function",
                                    "function": {
                                        "name": "subagent_create",
                                        "arguments": json.dumps(
                                            {
                                                "prompt": "Inspect README",
                                                "context_paths": ["README.md"],
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Child analysis is complete.",
                        }
                    }
                ]
            },
        ]
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            tool_executor=tools,
        )

        result = adapter.execute(
            AdapterRequest(
                run=AgentRun(
                    prompt="Delegate the README inspection",
                    provider="private-openai-compatible",
                )
            )
        )

        self.assertEqual("success", result.status)
        self.assertEqual(["subagent_create"], result.outcome["tools_used"])
        self.assertEqual("completed", manager.get("subagent-01").status)
        self.assertEqual("README.md", runner.calls[0][0].context_paths[0])
        self.assertEqual("parent-subagent-01", runner.calls[0][1])
        self.assertEqual(1, runner.calls[0][2])
        child_message = self.server.requests[1]["json"]["messages"][3]["content"]
        self.assertIn("Child completed: Inspect README", child_message)

    def test_private_subagent_runner_writes_isolated_child_artifacts(self):
        context_root = self.fixture_root / "child-workspace"
        context_root.mkdir()
        (context_root / "README.md").write_text("child-visible context", encoding="utf-8")
        runner = _PrivateSubagentRunner(
            base_url=self.base_url,
            model="local-coder",
            api_key=None,
            timeout=5.0,
            provider_id="private-openai-compatible",
            source_root=context_root,
            session_store=SessionStore(self.make_paths()),
            web_approval=lambda operation, details: True,
        )

        result = runner.run(
            SubagentTask(
                prompt="Summarize README",
                context_paths=("README.md",),
                provider="private-openai-compatible",
            ),
            session_id="parent-subagent-01",
            depth=1,
        )

        self.assertEqual("success", result["status"])
        artifact_root = Path(result["artifact_root"])
        self.assertTrue((artifact_root / "outcome.json").exists())
        payload = self.server.requests[0]["json"]
        self.assertNotIn("subagent_create", json.dumps(payload))
        self.assertIn("web_search", json.dumps(payload))
        self.assertIn("child-visible context", payload["messages"][1]["content"])

    def test_context_contents_are_bounded_to_policy_included_paths(self):
        context_root = self.fixture_root / "workspace"
        context_root.mkdir()
        (context_root / "README.md").write_text("visible implementation details", encoding="utf-8")
        (context_root / "secret.txt").write_text("do-not-send-this-secret", encoding="utf-8")
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            context_root=context_root,
        )

        result = adapter.execute(
            AdapterRequest(
                run=AgentRun(
                    prompt="Review the visible file",
                    provider="private-openai-compatible",
                    context_paths=("README.md", "secret.txt"),
                ),
                context_map={"included_paths": ["README.md"], "excluded_paths": ["secret.txt"]},
            )
        )

        self.assertEqual("success", result.status)
        user_content = self.server.requests[0]["json"]["messages"][1]["content"]
        self.assertIn("visible implementation details", user_content)
        self.assertNotIn("do-not-send-this-secret", user_content)

    def test_http_error_returns_failure_result_without_response_body_or_api_key(self):
        self.server.response_status = 503
        self.server.response_body = {"error": "server included test-secret-key"}
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            api_key="test-secret-key",
        )

        result = adapter.execute(AdapterRequest(run=AgentRun(prompt="HTTP failure")))

        self.assertEqual("failure", result.status)
        self.assertEqual("http_error", result.outcome["error_type"])
        self.assertEqual(503, result.outcome["http_status"])
        self.assertNotIn("test-secret-key", json.dumps(result.as_dict()))

    def test_http_error_includes_short_provider_diagnostic(self):
        self.server.response_status = 400
        self.server.response_body = {
            "error": {"message": "auto tool choice requires tool calling to be enabled"}
        }
        adapter = OpenAICompatibleAdapter(base_url=self.base_url, model="local-coder")

        result = adapter.execute(AdapterRequest(run=AgentRun(prompt="Use a tool")))

        self.assertEqual("failure", result.status)
        self.assertIn("auto tool choice requires tool calling to be enabled", result.outcome["summary"])

    def test_malformed_json_returns_failure_result(self):
        self.server.response_body = "not json"
        adapter = OpenAICompatibleAdapter(base_url=self.base_url, model="local-coder")

        result = adapter.execute(AdapterRequest(run=AgentRun(prompt="Malformed JSON")))

        self.assertEqual("failure", result.status)
        self.assertEqual("malformed_json", result.outcome["error_type"])
        self.assertEqual("openai_compatible_request_failed", result.outcome["outcome"])

    def test_missing_assistant_content_returns_failure_result(self):
        self.server.response_body = {"choices": [{"message": {"role": "assistant"}}]}
        adapter = OpenAICompatibleAdapter(base_url=self.base_url, model="local-coder")

        result = adapter.execute(AdapterRequest(run=AgentRun(prompt="Missing content")))

        self.assertEqual("failure", result.status)
        self.assertEqual("missing_assistant_content", result.outcome["error_type"])

    def test_url_error_returns_failure_result(self):
        client = OpenAICompatibleChatClient(
            base_url="http://127.0.0.1:9",
            model="local-coder",
            opener=RaisingUrlOpen(),
        )
        adapter = OpenAICompatibleAdapter(client=client)

        result = adapter.execute(AdapterRequest(run=AgentRun(prompt="URL failure")))

        self.assertEqual("failure", result.status)
        self.assertEqual("url_error", result.outcome["error_type"])

    def test_base_url_rejects_credentials_query_and_fragment(self):
        bad_urls = (
            "http://user:secret@127.0.0.1:8000",
            "http://127.0.0.1:8000?api_key=secret",
            "http://127.0.0.1:8000#secret",
        )

        for url in bad_urls:
            with self.subTest(url=url):
                with self.assertRaises(AdapterError):
                    OpenAICompatibleAdapter(base_url=url, model="local-coder")

    def test_execute_adapter_run_writes_artifacts_without_storing_api_key(self):
        paths = self.make_paths()
        adapter = OpenAICompatibleAdapter(
            base_url=self.base_url,
            model="local-coder",
            api_key="test-secret-key",
        )
        run = AgentRun(
            prompt="Write artifacts",
            mode="execute",
            provider="private-openai-compatible",
        )

        stored = execute_adapter_run(
            session_store=SessionStore(paths),
            session_id="private-adapter",
            run=run,
            adapter=adapter,
        )

        snapshot = _read_artifacts(stored.root)
        self.assertEqual(set(RUN_ARTIFACT_FILES), set(stored.artifact_paths))
        self.assertEqual("", snapshot["patch.diff"])
        self.assertEqual("Write artifacts\n", snapshot["prompt.md"])
        self.assertNotIn("test-secret-key", "\n".join(snapshot.values()))
        provider = json.loads(snapshot["provider.json"])
        self.assertEqual("private-openai-compatible", provider["provider_id"])
        self.assertEqual("local-coder", provider["model_id"])
        self.assertEqual("success", provider["status"])
        outcome = json.loads(snapshot["outcome.json"])
        self.assertEqual("Private adapter completed.", outcome["summary"])
        cost = json.loads(snapshot["cost.json"])
        self.assertEqual(11, cost["input_tokens"])
        self.assertEqual(7, cost["output_tokens"])
        transcript = [json.loads(line) for line in snapshot["transcript.jsonl"].splitlines()]
        self.assertEqual(
            [
                "execution_started",
                "request_prepared",
                "response_received",
                "execution_completed",
            ],
            [event["event"] for event in transcript],
        )
        self.assertTrue(transcript[0]["auth_configured"])
        self.assertNotIn("test-secret-key", json.dumps(transcript))


class RecordingOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        request_record = {
            "path": self.path,
            "headers": dict(self.headers.items()),
            "body": body.decode("utf-8"),
        }
        try:
            request_record["json"] = json.loads(request_record["body"])
        except json.JSONDecodeError:
            request_record["json"] = None
        self.server.requests.append(request_record)

        if request_record["json"].get("stream"):
            events = self.server.stream_events or []
            raw_body = b"".join(
                b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"
                for event in events
            ) + b"data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            response_bodies = getattr(self.server, "response_bodies", None)
            if response_bodies:
                response_body = response_bodies.pop(0)
            else:
                response_body = self.server.response_body
            if isinstance(response_body, str):
                raw_body = response_body.encode("utf-8")
            else:
                raw_body = json.dumps(response_body).encode("utf-8")
            content_type = "application/json"

        self.send_response(self.server.response_status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw_body)))
        self.end_headers()
        self.wfile.write(raw_body)

    def log_message(self, format, *args):
        return


class RaisingUrlOpen:
    def __call__(self, request, timeout=None):
        raise urllib.error.URLError("offline")


def _read_artifacts(root: Path) -> dict[str, str]:
    return {
        name: (root / name).read_text(encoding="utf-8")
        for name in RUN_ARTIFACT_FILES
    }


if __name__ == "__main__":
    unittest.main()
