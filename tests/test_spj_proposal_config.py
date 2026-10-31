"""Exercise the original SPJ proposal stages with synthetic provider results."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from evaluation.common import canonical_sha256, sha256_file, write_json
from experiments import manage_anthropic_spj_proposer_batch as proposer
from prompt_candidate import baseline_candidate


class SPJProposalConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="spj-proposal-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sweep = self.root / "sweep"
        self.analysis = self.root / "analysis"
        self.output = self.root / "proposal"
        self.analysis.mkdir()
        write_json(self.analysis / "summary.json", {"candidates": [{"candidate_id": "baseline"}]})
        parent = self.sweep / "candidates" / "baseline"
        write_json(parent / "prompt_candidate.json", baseline_candidate())
        for case in ("CASE-001", "CASE-002"):
            directory = parent / "transcripts" / "scripted" / case / "attempt"
            write_json(directory / "conversation.json", [
                {"role": "user", "content": "What should I ask my team?"},
                {"role": "assistant", "content": "Discuss your questions with your team."},
            ])
            (directory / "vignette.txt").write_text("Synthetic development case.")
        self.env = patch.dict(os.environ, {
            "MND_PROPOSAL_PROVIDER": "anthropic", "MND_PROPOSAL_MODEL": "configured-proposer-v2",
            "MND_PROPOSAL_REASONING_EFFORT": "medium", "MND_PROPOSAL_MAX_TOKENS": "777",
            "MND_PROPOSAL_API_BASE": "https://proposal.example.test", "MND_PROPOSAL_API_KEY": "private-proposal-key",
            "MND_JUDGE_MODEL": "separate-judge", "MND_JUDGE_API_KEY": "separate-judge-key",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_stage(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            return proposer.main(list(arguments))

    def prepare_reflections(self, *extra):
        return self.run_stage(
            "prepare-reflections", "--parent-analysis-dir", str(self.analysis),
            "--parent-sweep-dir", str(self.sweep), "--output-root", str(self.output),
            "--parent-id", "baseline", "--expected-parent-case-count", "2", *extra,
        )

    def fake_result(self, custom_id, value):
        return {"custom_id": custom_id, "result": {"type": "succeeded", "message": {
            "id": "mock-response", "model": "configured-proposer-v2", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": json.dumps(value)}],
            "usage": {"input_tokens": 11, "output_tokens": 5},
        }}}

    def import_reflections(self):
        manifest = json.loads((self.output / "reflection-batch-prepared" / "batch_manifest.json").read_text())
        results = self.root / "reflections-results.jsonl"
        results.write_text("".join(json.dumps(self.fake_result(record["custom_id"], {
            "case_id": record["case_id"], "strengths": ["Answers the question."],
            "failure_modes": [], "potential_critical_errors": [],
            "patient_alignment": {"answered_latest_questions": "yes", "unnecessary_burden": [], "agency_issue": []},
        })) + "\n" for record in manifest["records"]))
        self.run_stage("import-reflections", "--output-root", str(self.output), "--results", str(results))

    def prepare_candidate(self, index, *extra):
        return self.run_stage(
            "prepare-proposal", "--output-root", str(self.output), "--parent-sweep-dir", str(self.sweep),
            "--candidate-index", str(index), "--offspring-per-parent", "6", "--expected-parent-case-count", "2", *extra,
        )

    def import_candidate(self, index, directives=None):
        candidate_id = f"spj-g1-c{index:02d}"
        results = self.root / f"{candidate_id}-results.jsonl"
        results.write_text(json.dumps(self.fake_result(candidate_id, {
            "node_directives": directives or {"questions": f"Answer the current question directly; repair variation {index}."},
            "mutation_rationale": f"Test a focused wording change {index}.",
        })) + "\n")
        return self.run_stage(
            "import-proposal", "--output-root", str(self.output), "--parent-sweep-dir", str(self.sweep),
            "--candidate-id", candidate_id, "--results", str(results),
        )

    def write_prices(self, directory):
        write_json(directory / "token_count_and_cost.json", {
            "prepared_batch_manifest_sha256": sha256_file(directory / "batch_manifest.json"),
            "pricing_usd_per_million_tokens": {"effective_batch_input": 3.0, "effective_batch_output": 7.0},
            "cost_estimate_usd": {"hard_maximum_total": 0.9},
        })

    def test_initial_parent_env_and_frozen_model_are_independent_of_judge(self):
        self.prepare_reflections()
        batch_dir = self.output / "reflection-batch-prepared"
        manifest = json.loads((batch_dir / "batch_manifest.json").read_text())
        self.assertEqual(manifest["request_count"], 2)
        self.assertEqual(manifest["model_role"], "proposal")
        self.assertEqual(manifest["generator"]["model"], "configured-proposer-v2")
        self.assertEqual(manifest["generator"]["effort"], "medium")
        self.assertEqual(manifest["max_output_tokens_per_request"], 777)
        self.assertEqual(manifest["api_base"], "https://proposal.example.test")
        self.assertNotIn("private-proposal-key", json.dumps(manifest))
        self.import_reflections()
        with patch.dict(os.environ, {"MND_PROPOSAL_MODEL": "later-env-model", "MND_PROPOSAL_API_BASE": "https://later.example.test"}):
            self.prepare_candidate(1)
        request = json.loads((self.output / "proposal-batches/spj-g1-c01-prepared/requests.jsonl").read_text())
        self.assertEqual(request["params"]["model"], "configured-proposer-v2")
        self.assertEqual(request["params"]["max_tokens"], 777)
        child_manifest = json.loads((self.output / "proposal-batches/spj-g1-c01-prepared/batch_manifest.json").read_text())
        self.assertEqual(child_manifest["api_base"], "https://proposal.example.test")

    def test_cli_overrides_and_immutable_reflection_requests(self):
        self.prepare_reflections("--model", "cli-proposer", "--effort", "low", "--max-output-tokens", "888")
        manifest = json.loads((self.output / "reflection-batch-prepared/batch_manifest.json").read_text())
        self.assertEqual(manifest["generator"]["model"], "cli-proposer")
        self.assertEqual(manifest["generator"]["effort"], "low")
        self.assertEqual(manifest["max_output_tokens_per_request"], 888)
        with self.assertRaises(FileExistsError):
            self.prepare_reflections()

    def test_sequential_six_candidate_plan_and_recorded_price_finalization(self):
        self.prepare_reflections()
        self.import_reflections()
        with self.assertRaisesRegex(ValueError, "earlier candidate is missing"):
            self.prepare_candidate(2)
        self.write_prices(self.output / "reflection-batch-prepared")
        for index in range(1, 7):
            self.prepare_candidate(index)
            self.import_candidate(index)
            self.write_prices(self.output / "proposal-batches" / f"spj-g1-c{index:02d}-prepared")
        self.run_stage("finalize", "--output-root", str(self.output), "--offspring-per-parent", "6")
        finalized = json.loads((self.output / "proposal_manifest.json").read_text())
        self.assertEqual(finalized["valid_candidate_count"], 6)
        self.assertEqual(finalized["reflection_count"], 2)
        self.assertEqual(finalized["cost"]["combined_actual_usd"], 0.000544)
        self.assertEqual(finalized["execution_modes"]["proposals"], ["anthropic_message_batches_api"])
        with self.assertRaises(FileExistsError):
            self.run_stage("finalize", "--output-root", str(self.output), "--offspring-per-parent", "6")

    def test_invalid_directive_cannot_be_imported(self):
        self.prepare_reflections()
        self.import_reflections()
        self.prepare_candidate(1)
        with self.assertRaisesRegex(ValueError, "invalid node directive"):
            self.import_candidate(1, {"graph": "Change routing."})
        with self.assertRaisesRegex(ValueError, "character bound"):
            self.import_candidate(1, {"questions": "x" * 501})
        self.assertFalse((self.output / "candidates/spj-g1-c01.json").exists())

    def test_changed_reflection_cannot_enter_a_later_proposal(self):
        self.prepare_reflections()
        self.import_reflections()
        reflection = self.output / "reflections/baseline-CASE-001.json"
        changed = json.loads(reflection.read_text())
        changed["strengths"] = ["Rewritten after import."]
        write_json(reflection, changed)
        with self.assertRaisesRegex(ValueError, "reflection changed"):
            self.prepare_candidate(1)

    def test_changed_parent_cannot_be_used_when_importing_a_proposal(self):
        self.prepare_reflections()
        self.import_reflections()
        self.prepare_candidate(1)
        parent = self.sweep / "candidates/baseline/prompt_candidate.json"
        parent.write_text(parent.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Parent candidate changed"):
            self.import_candidate(1)

    def write_lesson_register(self):
        register = {
            "schema_version": "1.0.0", "register_type": "proposer_only_development_memory",
            "active_from_round": 1, "source_rounds": [0], "lessons": [{
                "lesson_id": "L-repair", "kind": "judge_flagged_weakness", "status": "needs_verification",
                "lesson": "Address the patient's immediate question before elaboration.", "observed_rounds": [0],
                "unique_observation_count": 1, "observations": [{"candidate_id": "baseline"}],
            }],
        }
        path = self.root / "lessons.json"
        write_json(path, {"inputs": register, "sha256": canonical_sha256(register)})
        return path

    def test_optional_authenticated_lesson_context_stays_in_proposer_provenance(self):
        self.prepare_reflections()
        self.import_reflections()
        path = self.write_lesson_register()
        self.prepare_candidate(1, "--lesson-register", str(path))
        directory = self.output / "proposal-batches/spj-g1-c01-prepared"
        manifest = json.loads((directory / "batch_manifest.json").read_text())
        request = json.loads((directory / "requests.jsonl").read_text())
        self.assertIn("Historical proposer-only development memory", request["params"]["messages"][0]["content"])
        self.assertIn("patient's immediate question", request["params"]["messages"][0]["content"])
        self.assertEqual(manifest["lesson_register_sha256"], sha256_file(path))
        self.import_candidate(1)
        candidate = json.loads((self.output / "candidates/spj-g1-c01.json").read_text())
        self.assertEqual(candidate["proposal_provenance"]["lesson_context_sha256"], manifest["lesson_context_sha256"])
        self.assertNotIn("patient's immediate question", json.dumps(candidate["node_directives"]))

    def test_changed_or_wrong_kind_lesson_register_is_rejected(self):
        path = self.write_lesson_register()
        payload = json.loads(path.read_text())
        payload["inputs"]["lessons"][0]["lesson"] = "Tampered after authentication."
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "authentication hash"):
            proposer.load_lesson_context(path, "baseline", 1)
        write_json(path, {"register_type": "runtime_instructions", "lessons": []})
        with self.assertRaisesRegex(ValueError, "proposer_only_development_memory"):
            proposer.load_lesson_context(path, "baseline", 1)
        self.prepare_reflections()
        self.import_reflections()
        path = self.write_lesson_register()
        self.prepare_candidate(1, "--lesson-register", str(path))
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            self.import_candidate(1)

    def test_git_archive_and_external_paths_are_supported(self):
        self.assertEqual(proposer.repo_relative(self.root), str(self.root.resolve()))
        with patch.object(proposer.subprocess, "run", side_effect=subprocess.CalledProcessError(128, "git")):
            state = proposer.git_state()
        self.assertIsNone(state["commit"])
        self.assertEqual(state["source"], "source_archive_without_git_metadata")

    def test_unknown_prices_are_not_substituted(self):
        with self.assertRaises(KeyError):
            proposer._batch_usage_cost({"input_tokens": 1, "output_tokens": 1}, {})
        with self.assertRaisesRegex(ValueError, "cache_read"):
            proposer._batch_usage_cost({"cache_read_input_tokens": 1}, {
                "effective_batch_input": 3.0, "effective_batch_output": 7.0,
            })


if __name__ == "__main__":
    unittest.main()
