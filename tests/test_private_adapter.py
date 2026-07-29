import json
import shutil
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agentx.adapters import AdapterError, AdapterRequest, execute_adapter_run
from agentx.config import AgentXPaths
from agentx.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleChatClient,
)
from agentx.routing import AgentRun
from agentx.store import RUN_ARTIFACT_FILES, SessionStore


class PrivateAdapterFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.fixture_root = Path("tests") / ".tmp_private_adapter"
        if self.fixture_root.exists():
            shutil.rmtree(self.fixture_root)
        self.fixture_root.mkdir(parents=True)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingOpenAIHandler)
        self.server.requests = []
        self.server.response_status = 200
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

        response_body = self.server.response_body
        if isinstance(response_body, str):
            raw_body = response_body.encode("utf-8")
        else:
            raw_body = json.dumps(response_body).encode("utf-8")

        self.send_response(self.server.response_status)
        self.send_header("Content-Type", "application/json")
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
