import unittest

from agentx.context import (
    MemoryRecord,
    compile_external_context_manifest,
    compile_private_context_manifest,
)
from agentx.policy import Policy


def policy_fixture():
    return Policy(
        private_provider="private-local",
        classification_rules={
            "docs/public/**": "public",
            "tests/**": "internal",
            "src/customer/**": "confidential",
            "src/core/**": "proprietary",
            "secrets/**": "secret",
        },
        routing={
            "confidential": ("private-local",),
            "proprietary": ("private-local",),
            "secret": ("private-local",),
        },
    )


def memory_fixtures():
    return (
        MemoryRecord(
            id="public-task",
            classification="public",
            content="User asked for a routing explanation.",
        ),
        MemoryRecord(
            id="customer-note",
            classification="confidential",
            content="Customer escalation details stay local.",
            summary="Customer escalation exists; use generic account terminology.",
        ),
        MemoryRecord(
            id="planner-note",
            classification="proprietary",
            content="Planner internals mention private heuristics.",
        ),
        MemoryRecord(
            id="secret-token",
            classification="secret",
            content="credential=abc123",
        ),
    )


class ContextManifestTests(unittest.TestCase):
    maxDiff = None

    def test_external_context_manifest_matches_golden(self):
        manifest = compile_external_context_manifest(
            policy_fixture(),
            requested_paths=[
                "docs/public/readme.md",
                "tests/test_policy.py",
                "src/customer/account.py",
            ],
            inferred_paths=[
                "src/core/planner.py",
                "secrets/api.key",
            ],
            memories=memory_fixtures(),
        )

        self.assertEqual(
            {
                "requested_paths": [
                    "docs/public/readme.md",
                    "tests/test_policy.py",
                    "src/customer/account.py",
                ],
                "inferred_paths": [
                    "src/core/planner.py",
                    "secrets/api.key",
                ],
                "included_paths": [
                    "docs/public/readme.md",
                    "tests/test_policy.py",
                ],
                "excluded_paths": [
                    "src/customer/account.py",
                    "src/core/planner.py",
                    "secrets/api.key",
                ],
                "classification_by_path": {
                    "docs/public/readme.md": "public",
                    "tests/test_policy.py": "internal",
                    "src/customer/account.py": "confidential",
                    "src/core/planner.py": "proprietary",
                    "secrets/api.key": "secret",
                },
                "provider_visible_context": {
                    "provider_class": "external",
                    "visible_paths": [
                        "docs/public/readme.md",
                        "tests/test_policy.py",
                    ],
                    "visible_memories": [
                        {
                            "memory_id": "public-task",
                            "classification": "public",
                            "source": "content",
                            "text": "User asked for a routing explanation.",
                        },
                        {
                            "memory_id": "customer-note",
                            "classification": "confidential",
                            "source": "summary",
                            "text": "Customer escalation exists; use generic account terminology.",
                        },
                    ],
                },
                "redactions": [
                    {
                        "target_type": "path",
                        "target_id": "src/customer/account.py",
                        "classification": "confidential",
                        "action": "exclude",
                        "reason": "path_classification_not_visible_to_external_provider",
                    },
                    {
                        "target_type": "path",
                        "target_id": "src/core/planner.py",
                        "classification": "proprietary",
                        "action": "exclude",
                        "reason": "path_classification_not_visible_to_external_provider",
                    },
                    {
                        "target_type": "path",
                        "target_id": "secrets/api.key",
                        "classification": "secret",
                        "action": "exclude",
                        "reason": "path_classification_not_visible_to_external_provider",
                    },
                    {
                        "target_type": "memory",
                        "target_id": "planner-note",
                        "classification": "proprietary",
                        "action": "redact",
                        "reason": "memory_redacted_from_external_provider",
                    },
                    {
                        "target_type": "memory",
                        "target_id": "secret-token",
                        "classification": "secret",
                        "action": "exclude",
                        "reason": "secret_memory_excluded_from_external_provider",
                    },
                ],
                "summary_substitutions": [
                    {
                        "target_type": "path",
                        "target_id": "src/customer/account.py",
                        "classification": "confidential",
                        "summary": (
                            "Withheld confidential path 'src/customer/account.py'. "
                            "Provider received a summary placeholder only."
                        ),
                        "reason": "summary_substitution_for_withheld_path",
                    },
                    {
                        "target_type": "path",
                        "target_id": "src/core/planner.py",
                        "classification": "proprietary",
                        "summary": (
                            "Withheld proprietary path 'src/core/planner.py'. "
                            "Provider received a summary placeholder only."
                        ),
                        "reason": "summary_substitution_for_withheld_path",
                    },
                    {
                        "target_type": "memory",
                        "target_id": "customer-note",
                        "classification": "confidential",
                        "summary": "Customer escalation exists; use generic account terminology.",
                        "reason": "summary_substitution_for_memory",
                    },
                ],
                "policy_decision": {
                    "provider_class": "external",
                    "eligible": False,
                    "reason": "requested_paths_blocked_by_external_context_policy",
                    "external_max_classification": "internal",
                    "effective_max_classification": "internal",
                    "highest_requested_classification": "secret",
                    "highest_included_classification": "internal",
                    "has_unclassified_paths": False,
                    "secret_explicitly_allowed": False,
                },
                "memory_exposure": [
                    {
                        "memory_id": "public-task",
                        "classification": "public",
                        "action": "include",
                        "visible_text": "User asked for a routing explanation.",
                        "source": "content",
                        "reason": "memory_classification_visible_to_external_provider",
                    },
                    {
                        "memory_id": "customer-note",
                        "classification": "confidential",
                        "action": "summarize",
                        "visible_text": "Customer escalation exists; use generic account terminology.",
                        "source": "summary",
                        "reason": "memory_summary_used_for_external_provider",
                    },
                    {
                        "memory_id": "planner-note",
                        "classification": "proprietary",
                        "action": "redact",
                        "visible_text": None,
                        "source": "metadata",
                        "reason": "memory_redacted_from_external_provider",
                    },
                    {
                        "memory_id": "secret-token",
                        "classification": "secret",
                        "action": "exclude",
                        "visible_text": None,
                        "source": "metadata",
                        "reason": "secret_memory_excluded_from_external_provider",
                    },
                ],
            },
            manifest.as_dict(),
        )

    def test_external_context_excludes_confidential_and_proprietary_paths(self):
        manifest = compile_external_context_manifest(
            policy_fixture(),
            requested_paths=[
                "src/customer/account.py",
                "src/core/planner.py",
                "tests/test_routing.py",
            ],
            memories=memory_fixtures(),
        )

        self.assertEqual(("tests/test_routing.py",), manifest.included_paths)
        self.assertEqual(
            ("src/customer/account.py", "src/core/planner.py"),
            manifest.excluded_paths,
        )
        self.assertFalse(manifest.policy_decision.eligible)

    def test_private_context_includes_allowed_higher_tier_paths(self):
        manifest = compile_private_context_manifest(
            policy_fixture(),
            requested_paths=[
                "docs/public/readme.md",
                "src/customer/account.py",
                "src/core/planner.py",
            ],
            inferred_paths=["secrets/api.key"],
            memories=memory_fixtures(),
        )

        self.assertEqual(
            (
                "docs/public/readme.md",
                "src/customer/account.py",
                "src/core/planner.py",
            ),
            manifest.included_paths,
        )
        self.assertEqual(("secrets/api.key",), manifest.excluded_paths)
        self.assertEqual(
            ["public-task", "customer-note", "planner-note"],
            [
                entry["memory_id"]
                for entry in manifest.provider_visible_context.as_dict()["visible_memories"]
            ],
        )
        self.assertTrue(manifest.policy_decision.eligible)

    def test_external_context_summarizes_or_redacts_disallowed_memories(self):
        manifest = compile_external_context_manifest(
            policy_fixture(),
            requested_paths=["tests/test_context.py"],
            memories=memory_fixtures(),
        )

        decisions = {
            decision.memory_id: decision for decision in manifest.memory_exposure
        }

        self.assertEqual("include", decisions["public-task"].action)
        self.assertEqual("summarize", decisions["customer-note"].action)
        self.assertEqual("redact", decisions["planner-note"].action)
        self.assertEqual("exclude", decisions["secret-token"].action)

    def test_every_manifest_has_memory_exposure_decision_for_each_memory(self):
        memories = memory_fixtures()
        external_manifest = compile_external_context_manifest(
            policy_fixture(),
            requested_paths=["tests/test_context.py"],
            memories=memories,
        )
        private_manifest = compile_private_context_manifest(
            policy_fixture(),
            requested_paths=["tests/test_context.py"],
            memories=memories,
        )

        self.assertEqual(
            {memory.id for memory in memories},
            {decision.memory_id for decision in external_manifest.memory_exposure},
        )
        self.assertEqual(
            {memory.id for memory in memories},
            {decision.memory_id for decision in private_manifest.memory_exposure},
        )

    def test_private_context_can_include_secret_with_explicit_opt_in(self):
        manifest = compile_private_context_manifest(
            policy_fixture(),
            requested_paths=["secrets/api.key"],
            memories=memory_fixtures(),
            allow_secret=True,
        )

        self.assertEqual(("secrets/api.key",), manifest.included_paths)
        self.assertTrue(manifest.policy_decision.eligible)
        self.assertEqual(
            "secret-token",
            manifest.provider_visible_context.visible_memories[-1].memory_id,
        )


if __name__ == "__main__":
    unittest.main()
