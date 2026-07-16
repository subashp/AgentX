import unittest

from agentx.policy import Policy


class PolicyTests(unittest.TestCase):
    def test_policy_from_text_classifies_fixture_paths(self):
        policy = Policy.from_text(
            """
            [defaults]
            external_max_classification = "internal"
            private_provider = "private-local"

            [classification]
            "docs/public/**" = "public"
            "tests/**" = "internal"
            "src/customer/**" = "confidential"
            "src/core/**" = "proprietary"
            """
        )

        classifications = {
            entry.normalized_path: entry.classification
            for entry in policy.classify_paths(
                [
                    "docs/public/readme.md",
                    "tests/test_policy.py",
                    "src/customer/account.py",
                    "src/core/planner.py",
                ]
            )
        }

        self.assertEqual("public", classifications["docs/public/readme.md"])
        self.assertEqual("internal", classifications["tests/test_policy.py"])
        self.assertEqual("confidential", classifications["src/customer/account.py"])
        self.assertEqual("proprietary", classifications["src/core/planner.py"])

    def test_unknown_paths_use_default_classification_when_configured(self):
        policy = Policy(default_classification="confidential")

        result = policy.classify_path("notes/todo.txt")

        self.assertEqual("confidential", result.classification)
        self.assertEqual("default", result.source)

    def test_unknown_paths_can_remain_unclassified_and_require_private_provider(self):
        policy = Policy(
            private_provider="private-local",
            default_classification=None,
            require_private_for_unclassified=True,
        )

        result = policy.classify_path("notes/todo.txt")
        eligibility = policy.evaluate_provider_eligibility(
            ["codex", "private-local"],
            ["notes/todo.txt"],
            public_provider_ids={"codex"},
        )

        self.assertIsNone(result.classification)
        self.assertEqual("unclassified", result.source)
        self.assertEqual(("private-local",), eligibility.eligible_provider_ids)
        self.assertEqual(
            "unclassified_requires_private",
            eligibility.rejected_providers["codex"],
        )

    def test_secret_is_deny_by_default_without_explicit_routing(self):
        policy = Policy(
            private_provider="private-local",
            classification_rules={"secrets/**": "secret"},
        )

        eligibility = policy.evaluate_provider_eligibility(
            ["codex", "private-local"],
            ["secrets/api.key"],
            public_provider_ids={"codex"},
        )

        self.assertEqual((), eligibility.eligible_provider_ids)
        self.assertEqual(
            "classification_exceeds_external_max",
            eligibility.rejected_providers["codex"],
        )
        self.assertEqual(
            "secret_requires_explicit_routing",
            eligibility.rejected_providers["private-local"],
        )

    def test_confidential_and_proprietary_paths_remove_external_providers(self):
        policy = Policy(
            private_provider="private-local",
            classification_rules={
                "src/customer/**": "confidential",
                "src/core/**": "proprietary",
            },
            routing={
                "confidential": ("private-local",),
                "proprietary": ("private-local",),
            },
        )

        confidential = policy.evaluate_provider_eligibility(
            ["codex", "claude", "private-local"],
            ["src/customer/account.py"],
            public_provider_ids={"codex", "claude"},
        )
        proprietary = policy.evaluate_provider_eligibility(
            ["codex", "claude", "private-local"],
            ["src/core/planner.py"],
            public_provider_ids={"codex", "claude"},
        )

        self.assertEqual(("private-local",), confidential.eligible_provider_ids)
        self.assertEqual(("private-local",), proprietary.eligible_provider_ids)
        self.assertEqual(
            "classification_exceeds_external_max",
            confidential.rejected_providers["codex"],
        )
        self.assertEqual(
            "classification_exceeds_external_max",
            proprietary.rejected_providers["claude"],
        )

    def test_portable_separators_are_normalized_for_patterns_and_paths(self):
        policy = Policy(
            classification_rules={
                ".\\src\\customer\\**": "confidential",
            }
        )

        result = policy.classify_path(".\\src\\customer\\account.py")

        self.assertEqual("src/customer/account.py", result.normalized_path)
        self.assertEqual("confidential", result.classification)
        self.assertEqual("src/customer/**", result.matched_pattern)


if __name__ == "__main__":
    unittest.main()
