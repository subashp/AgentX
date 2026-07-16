import json
import unittest
from dataclasses import dataclass

from agentx.lifecycle import (
    HealthCheckResult,
    LifecycleError,
    RUNTIME_MODE_CONNECT,
    RUNTIME_MODE_LAUNCH,
    RuntimeConfig,
    RuntimeLifecycleManager,
    SHUTDOWN_ALWAYS,
    SHUTDOWN_NEVER,
)


class LifecycleTests(unittest.TestCase):
    def test_existing_endpoint_mode_waits_for_health_without_launcher(self):
        launcher = RecordingLauncher()
        manager = RuntimeLifecycleManager(
            launcher=launcher,
            health_check=SequenceHealthCheck([HealthCheckResult(True, "ready")]),
        )
        config = RuntimeConfig(
            runtime_id="private-local",
            endpoint="http://127.0.0.1:8000/v1",
            mode=RUNTIME_MODE_CONNECT,
        )

        result = manager.acquire(config)

        self.assertTrue(result.ready)
        self.assertFalse(result.launched)
        self.assertIsNone(result.handle)
        self.assertEqual(0, launcher.launch_count)
        self.assertEqual(
            [
                "runtime_lifecycle_started",
                "runtime_connect_selected",
                "runtime_health_check_passed",
                "runtime_lifecycle_ready",
            ],
            [event["event"] for event in result.events],
        )

    def test_fake_launch_waits_until_health_passes(self):
        clock = FakeClock()
        launcher = RecordingLauncher()
        health = SequenceHealthCheck(
            [
                HealthCheckResult(False, "starting"),
                HealthCheckResult(True, "ready"),
            ]
        )
        manager = RuntimeLifecycleManager(
            launcher=launcher,
            health_check=health,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        config = RuntimeConfig(
            runtime_id="private-local",
            endpoint="http://127.0.0.1:8000",
            mode=RUNTIME_MODE_LAUNCH,
            command="model-server",
            args=("--port", "8000"),
            env={"MODEL_PATH": "/models/local"},
            health_interval_seconds=0.25,
        )

        result = manager.acquire(config)

        self.assertEqual("ready", result.status)
        self.assertTrue(result.launched)
        self.assertEqual(("model-server", "--port", "8000"), launcher.configs[0].argv)
        self.assertEqual([0.25], clock.sleeps)
        self.assertEqual(2, health.calls)
        self.assertEqual(
            [
                "runtime_lifecycle_started",
                "runtime_launch_started",
                "runtime_launch_completed",
                "runtime_health_check_failed",
                "runtime_health_check_passed",
                "runtime_lifecycle_ready",
            ],
            [event["event"] for event in result.events],
        )
        self.assertEqual(["MODEL_PATH"], result.events[0]["env_keys"])

    def test_health_timeout_is_deterministic(self):
        clock = FakeClock()
        manager = RuntimeLifecycleManager(
            launcher=RecordingLauncher(),
            health_check=SequenceHealthCheck(
                [
                    HealthCheckResult(False, "starting"),
                    HealthCheckResult(False, "starting"),
                    HealthCheckResult(False, "starting"),
                ]
            ),
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        config = RuntimeConfig(
            runtime_id="private-local",
            endpoint="http://127.0.0.1:8000",
            mode=RUNTIME_MODE_LAUNCH,
            command="model-server",
            health_timeout_seconds=2.0,
            health_interval_seconds=1.0,
        )

        result = manager.acquire(config)

        self.assertEqual("health_timeout", result.status)
        self.assertFalse(result.ready)
        self.assertEqual([1.0, 1.0], clock.sleeps)
        self.assertEqual("runtime_health_timeout", result.events[-1]["event"])
        self.assertEqual(3, result.events[-1]["attempts"])

    def test_launcher_failure_returns_failure_status(self):
        manager = RuntimeLifecycleManager(
            launcher=FailingLauncher(),
            health_check=SequenceHealthCheck([HealthCheckResult(True, "ready")]),
        )
        config = RuntimeConfig(
            runtime_id="private-local",
            endpoint="http://127.0.0.1:8000",
            mode=RUNTIME_MODE_LAUNCH,
            command="model-server",
        )

        result = manager.acquire(config)

        self.assertEqual("launch_failed", result.status)
        self.assertFalse(result.launched)
        self.assertIsNone(result.handle)
        self.assertEqual("runtime_launch_failed", result.events[-1]["event"])
        self.assertEqual("RuntimeError", result.events[-1]["error_type"])

    def test_shutdown_policy_controls_launched_handle(self):
        manager = RuntimeLifecycleManager(
            launcher=RecordingLauncher(),
            health_check=SequenceHealthCheck(
                [
                    HealthCheckResult(True, "ready"),
                    HealthCheckResult(True, "ready"),
                ]
            ),
        )
        config = RuntimeConfig(
            runtime_id="private-local",
            endpoint="http://127.0.0.1:8000",
            mode=RUNTIME_MODE_LAUNCH,
            command="model-server",
            shutdown_policy=SHUTDOWN_ALWAYS,
            shutdown_timeout_seconds=3.0,
        )
        result = manager.acquire(config)

        shutdown = manager.shutdown(result)

        self.assertEqual("shutdown_complete", shutdown.status)
        self.assertTrue(shutdown.attempted)
        self.assertEqual([3.0], result.handle.shutdown_timeouts)
        self.assertEqual(
            ["runtime_shutdown_started", "runtime_shutdown_completed"],
            [event["event"] for event in shutdown.events],
        )

        never_config = RuntimeConfig(
            runtime_id="private-never",
            endpoint="http://127.0.0.1:8001",
            mode=RUNTIME_MODE_LAUNCH,
            command="model-server",
            shutdown_policy=SHUTDOWN_NEVER,
        )
        never_result = manager.acquire(never_config)

        skipped = manager.shutdown(never_result)

        self.assertEqual("skipped", skipped.status)
        self.assertFalse(skipped.attempted)
        self.assertEqual([], never_result.handle.shutdown_timeouts)
        self.assertEqual("runtime_shutdown_skipped", skipped.events[0]["event"])

    def test_events_do_not_leak_env_or_auth_secret_values(self):
        secret = "super-secret-token"
        manager = RuntimeLifecycleManager(
            launcher=FailingLauncher(secret),
            health_check=SequenceHealthCheck([HealthCheckResult(True, "ready")]),
        )
        config = RuntimeConfig(
            runtime_id="private-local",
            endpoint="http://127.0.0.1:8000",
            mode=RUNTIME_MODE_LAUNCH,
            command="model-server",
            args=("--api-key", secret),
            env={"API_KEY": secret, "TOKEN_PATH": "safe-key-name-only"},
            auth_fields={"authorization": "Bearer " + secret},
            shutdown_policy=SHUTDOWN_ALWAYS,
        )

        result = manager.acquire(config)
        encoded = json.dumps(result.as_dict(), sort_keys=True)

        self.assertNotIn(secret, encoded)
        self.assertNotIn("Bearer " + secret, encoded)
        self.assertIn("API_KEY", encoded)
        self.assertIn("authorization", json.dumps(config.auth_fields))
        self.assertNotIn("authorization", encoded)
        self.assertEqual(2, result.events[0]["arg_count"])
        self.assertTrue(result.events[0]["auth_configured"])
        self.assertEqual(1, result.events[0]["auth_field_count"])

    def test_launch_mode_requires_command(self):
        with self.assertRaises(LifecycleError):
            RuntimeConfig(
                runtime_id="private-local",
                endpoint="http://127.0.0.1:8000",
                mode=RUNTIME_MODE_LAUNCH,
            )


@dataclass
class FakeHandle:
    runtime_id: str
    endpoint: str
    pid: int | None = 4321

    def __post_init__(self):
        self.shutdown_timeouts = []

    def shutdown(self, timeout=None):
        self.shutdown_timeouts.append(timeout)


class RecordingLauncher:
    def __init__(self):
        self.configs = []
        self.handles = []

    @property
    def launch_count(self):
        return len(self.configs)

    def launch(self, config):
        self.configs.append(config)
        handle = FakeHandle(config.runtime_id, config.endpoint)
        self.handles.append(handle)
        return handle


class FailingLauncher:
    def __init__(self, message="launch failed"):
        self.message = message

    def launch(self, config):
        raise RuntimeError(self.message)


class SequenceHealthCheck:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def __call__(self, config):
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return HealthCheckResult(False, "not_ready")


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


if __name__ == "__main__":
    unittest.main()
