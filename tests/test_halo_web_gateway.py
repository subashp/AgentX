import importlib.util
import http.client
import json
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _load_gateway_module():
    path = Path(__file__).resolve().parents[1] / "deploy" / "halo" / "vllm-web-gateway.py"
    spec = importlib.util.spec_from_file_location("agentx_halo_web_gateway_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Halo web gateway")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gateway = _load_gateway_module()


class HaloWebGatewayTests(unittest.TestCase):
    def test_chat_textarea_enter_submits_and_shift_enter_keeps_newline(self):
        page = gateway.PAGE.decode("utf-8")

        self.assertIn("Enter to send, Shift+Enter for a newline", page)
        self.assertIn("prompt.addEventListener('keydown'", page)
        self.assertIn("e.key==='Enter'&&!e.shiftKey&&!e.isComposing", page)
        self.assertIn("e.preventDefault();submitPrompt();", page)
        self.assertIn("form.requestSubmit", page)

    def test_web_tools_are_advertised_to_vllm(self):
        tools = {item["function"]["name"]: item for item in gateway.WEB_TOOL_SPECS}
        self.assertTrue({"web_search", "web_fetch"}.issubset(tools))
        self.assertIn("browser_open", tools)
        for name in ("web_search", "web_fetch"):
            spec = tools[name]
            description = spec["function"]["description"].lower()
            self.assertIn("already permitted", description)
            self.assertNotIn("approval", description)

    def test_streamed_standard_tool_call_is_reassembled(self):
        collected = {}
        gateway.append_stream_tool_calls(
            collected,
            {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "web_", "arguments": '{"url":"https://'}}]},
        )
        gateway.append_stream_tool_calls(
            collected,
            {"tool_calls": [{"index": 0, "function": {"name": "fetch", "arguments": 'finance.yahoo.com/quote/AMD/"}'}}]},
        )

        self.assertEqual(
            [
                {
                    "id": "call-1",
                    "name": "web_fetch",
                    "arguments": {"url": "https://finance.yahoo.com/quote/AMD/"},
                }
            ],
            gateway.standard_stream_tool_calls(collected),
        )

    def test_raw_qwen_tool_call_is_normalized_and_hidden_from_visible_reply(self):
        raw = '<tool_call>{"name":"web_fetch","arguments":{"url":"https://finance.yahoo.com/quote/AMD/"}}</tool_call>'

        calls = gateway.raw_qwen_tool_calls(raw)

        self.assertEqual("web_fetch", calls[0]["name"])
        self.assertEqual("https://finance.yahoo.com/quote/AMD/", calls[0]["arguments"]["url"])
        self.assertEqual("(No visible response)", gateway.visible_content(raw))

    def test_browser_research_preflight_uses_yahoo_quote_api_for_named_stock(self):
        request = gateway.browser_research_request(
            "Can you check finance.yahoo.com and get me the current price of AMD?"
        )

        self.assertEqual(
            (
                "web_fetch",
                {
                    "url": "https://query1.finance.yahoo.com/v8/finance/chart/AMD",
                    "max_chars": 6_000,
                },
            ),
            request,
        )

    def test_legacy_home_database_is_moved_to_agentx_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "vllm-chat.sqlite3"
            destination = root / ".agentx" / "vllm-chat.sqlite3"
            with sqlite3.connect(legacy) as connection:
                connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
                connection.execute("INSERT INTO marker(value) VALUES ('preserved')")

            migrated = gateway.migrate_legacy_chat_database(
                legacy_path=legacy,
                destination_path=destination,
            )

            self.assertTrue(migrated)
            self.assertFalse(legacy.exists())
            with sqlite3.connect(destination) as connection:
                self.assertEqual("preserved", connection.execute("SELECT value FROM marker").fetchone()[0])

    def test_browser_chat_executes_web_tool_but_machine_chat_does_not_receive_tools(self):
        class FakeVllmHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                self.server.requests.append(payload)
                if len(self.server.requests) == 1:
                    content = '<tool_call>{"name":"web_fetch","arguments":{"url":"https://finance.yahoo.com/quote/AMD/"}}</tool_call>'
                elif len(self.server.requests) == 2:
                    # Reproduce the empty post-tool completion observed with
                    # the Qwen/vLLM browser path.
                    content = ""
                else:
                    content = "AMD is the stock symbol for Advanced Micro Devices."
                events = [
                    {"choices": [{"delta": {"content": content}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
                raw = b"".join(
                    b"data: " + json.dumps(event).encode() + b"\n\n"
                    for event in events
                ) + b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):
                return

        class FakeWebResearchTools:
            def __init__(self, *, approval_callback):
                self.approval_callback = approval_callback

            def call(self, name, arguments):
                if not self.approval_callback("web.fetch", {"url": arguments["url"], "max_chars": 4000}):
                    return gateway.ToolResult(name=name, ok=False, error="web access was unexpectedly disabled")
                return gateway.ToolResult(
                    name=name,
                    ok=True,
                    output={"url": arguments["url"], "content": "AMD test quote"},
                )

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllmHandler)
        upstream.requests = []
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()

        original_host = gateway.UPSTREAM_HOST
        original_port = gateway.UPSTREAM_PORT
        original_db_path = gateway.DB_PATH
        original_tools = gateway.WebResearchTools
        with tempfile.TemporaryDirectory() as temporary:
            gateway.UPSTREAM_HOST, gateway.UPSTREAM_PORT = upstream.server_address
            gateway.DB_PATH = str(Path(temporary) / "chat.sqlite3")
            gateway.WebResearchTools = FakeWebResearchTools
            gateway.init_db()
            service = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
            service_thread = threading.Thread(target=service.serve_forever, daemon=True)
            service_thread.start()
            host, port = service.server_address
            try:
                session_connection, session_response = _post_json(host, port, "/api/sessions", {"user_id": "0"})
                session_id = json.loads(session_response.read())["id"]
                session_response.close()
                session_connection.close()

                connection = http.client.HTTPConnection(host, port, timeout=3)
                connection.request(
                    "POST",
                    f"/api/sessions/{session_id}/chat",
                    body=json.dumps({"user_id": "0", "model": "test-model", "content": "Fetch AMD"}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                while True:
                    line = response.readline()
                    if not line:
                        break
                response.close()
                connection.close()

                machine_connection, machine_response = _post_json(
                    host,
                    port,
                    "/api/machine-chat",
                    {"user_id": "0", "model": "test-model", "content": "Hello from a machine client"},
                )
                machine_response.read()
                machine_response.close()
                machine_connection.close()

                self.assertEqual(4, len(upstream.requests))
                self.assertNotIn("tools", upstream.requests[2])
                self.assertEqual("user", upstream.requests[2]["messages"][-1]["role"])
                self.assertIn("AMD test quote", upstream.requests[1]["messages"][-1]["content"])
                self.assertNotIn("tool", {message["role"] for message in upstream.requests[1]["messages"]})
                self.assertNotIn("tools", upstream.requests[3])
                messages = gateway.get_messages(session_id)
                self.assertEqual("AMD is the stock symbol for Advanced Micro Devices.", messages[-1]["content"])
            finally:
                service.shutdown()
                service.server_close()
                service_thread.join(timeout=2)
                gateway.WebResearchTools = original_tools
                gateway.UPSTREAM_HOST = original_host
                gateway.UPSTREAM_PORT = original_port
                gateway.DB_PATH = original_db_path
                upstream.shutdown()
                upstream.server_close()
                upstream_thread.join(timeout=2)


def _post_json(host, port, path, payload):
    connection = http.client.HTTPConnection(host, port, timeout=3)
    connection.request(
        "POST",
        path,
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    return connection, response


if __name__ == "__main__":
    unittest.main()
