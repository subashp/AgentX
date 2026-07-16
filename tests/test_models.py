import unittest
from datetime import date

from agentx.models import ModelCatalog, ModelCatalogError, classify_task_complexity


def profile(
    provider_id="codex",
    model_id="gpt-5-mini",
    tier="economy",
    capability_score=60,
    cost_profile="economy",
    latency_profile="fast",
    context_limit=128000,
    tool_support=True,
    structured_output_support=True,
    privacy_clearance="internal",
    best_for=("tests", "docs"),
    not_for=("risky refactors",),
    metadata_source="manual",
    metadata_updated_at="2026-07-01",
):
    return {
        "provider_id": provider_id,
        "model_id": model_id,
        "tier": tier,
        "capability_score": capability_score,
        "cost_profile": cost_profile,
        "latency_profile": latency_profile,
        "context_limit": context_limit,
        "tool_support": tool_support,
        "structured_output_support": structured_output_support,
        "privacy_clearance": privacy_clearance,
        "best_for": list(best_for),
        "not_for": list(not_for),
        "metadata_source": metadata_source,
        "metadata_updated_at": metadata_updated_at,
    }


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_parses_profiles_from_dicts(self):
        catalog = ModelCatalog.from_dicts(
            [
                profile(),
                profile(
                    provider_id="claude",
                    model_id="sonnet-code",
                    tier="standard",
                    capability_score=80,
                    cost_profile="standard",
                    latency_profile="standard",
                ),
            ]
        )

        self.assertEqual(2, len(catalog.profiles))
        self.assertEqual("codex", catalog.profiles[0].provider_id)
        self.assertEqual(date(2026, 7, 1), catalog.profiles[0].metadata_updated_at)
        self.assertEqual(["tests", "docs"], catalog.as_dict()["profiles"][0]["best_for"])

    def test_invalid_catalog_rejects_bad_tier(self):
        with self.assertRaisesRegex(ModelCatalogError, "invalid tier 'ultra'"):
            ModelCatalog.from_dicts([profile(tier="ultra")])

    def test_invalid_catalog_rejects_duplicate_provider_model_pairs(self):
        with self.assertRaisesRegex(
            ModelCatalogError,
            "duplicate model profile 'codex/gpt-5-mini'",
        ):
            ModelCatalog.from_dicts([profile(), profile()])

    def test_stale_metadata_warnings_report_old_profiles(self):
        catalog = ModelCatalog.from_dicts(
            [profile(metadata_updated_at="2026-01-01")]
        )

        warnings = catalog.stale_warnings(
            as_of=date(2026, 7, 16),
            max_age_days=30,
        )

        self.assertEqual(1, len(warnings))
        self.assertIn("196 days old", warnings[0])
        self.assertIn("2026-01-01", warnings[0])


class TaskComplexityClassifierTests(unittest.TestCase):
    def test_classifier_matches_mode_and_hints(self):
        fixtures = (
            ("plan", (), "high"),
            ("execute", (), "standard"),
            ("review", ("risky_refactor",), "high"),
            ("tests", (), "economy"),
            ("explain", ("log triage",), "economy"),
        )

        for mode, hints, expected_tier in fixtures:
            with self.subTest(mode=mode, hints=hints):
                assessment = classify_task_complexity(mode, hints)
                self.assertEqual(expected_tier, assessment.tier)

    def test_classifier_explains_hint_escalation(self):
        assessment = classify_task_complexity(
            "execute",
            ["architecture planning", "summarization"],
        )

        self.assertEqual("high", assessment.tier)
        self.assertIn("mode 'execute' maps to tier 'standard'", assessment.explanation)
        self.assertIn("architecture_planning", assessment.explanation)


if __name__ == "__main__":
    unittest.main()
