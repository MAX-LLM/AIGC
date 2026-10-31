"""Offline checks for canonical SPJ candidate delivery and session isolation."""

import json
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import agent_api
from agent_base import CounsellingState, build_system_message
from evaluation.build_spj_generation_pairs import build_generation_dataset
from mnd_langgraph_agent import LangGraphMNDAgent
from prompt_candidate import baseline_candidate, canonical_sha256, validate_prompt_candidate
from spj_evolve import build_parser, run_sweep
from tests.test_langgraph_agent import FakeBackend


class RuntimeFakeBackend(FakeBackend):
    def generation_options(self, max_new_tokens=None):
        return nullcontext()


def candidate(**directives):
    return validate_prompt_candidate({
        "schema_version": "1.0.0", "candidate_id": "spj-runtime-test",
        "parent_id": "baseline", "generation": 1,
        "node_directives": directives,
        "mutation_rationale": "Check that proposed directives reach the configured node.",
    })


@patch.dict(os.environ, {"MND_RETRIEVAL_MODE": "lexical"})
class CandidateRuntimeTests(unittest.TestCase):
    def test_portable_sweep_authenticates_directives_and_builds_mirrored_pairs(self):
        # A temporary, hash-checked case keeps this test independent of study data.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir, candidate_dir = root / 'cases', root / 'candidates'
            case_dir.mkdir()
            candidate_dir.mkdir()
            patient_turn = 'What should I ask my genetics team?'
            case = {
                'immutable_blueprint': {'case_id': 'TEST-CASE', 'phase': 'pre_test',
                    'scenario': 'Synthetic discussion', 'fixed_facts': {}},
                'generated_surface': {'case_id': 'TEST-CASE', 'opening_message': patient_turn,
                    'persona_surface': 'Synthetic participant',
                    'turn_plan': [{'patient_utterance': patient_turn, 'reveal_only_after': 0}]},
            }
            case['content_sha256'] = canonical_sha256(case)
            (case_dir / 'example.json').write_text(json.dumps(case), encoding='utf-8')
            selected = candidate(questions='Answer the direct question first.')
            (candidate_dir / 'candidate.json').write_text(json.dumps(selected), encoding='utf-8')
            args = build_parser().parse_args([
                '--case-dir', str(case_dir), '--candidate-dir', str(candidate_dir),
                '--output-dir', str(root / 'sweep'),
            ])
            active, audit = None, []

            def request(method, url, **kwargs):
                nonlocal active, audit
                if url.endswith('/health'):
                    return {'prompt_candidates_enabled': True, 'effective_prompts_recorded': True,
                        'agent_model': 'fake-agent', 'backend': 'api', 'max_new_tokens': 512}
                if url.endswith('/reset'):
                    active, audit = kwargs['json']['prompt_candidate'], []
                    return {'candidate_sha256': active['candidate_sha256']}
                if url.endswith('/chat'):
                    prompt = 'Question prompt. ' + active['node_directives'].get('questions', '')
                    messages = [{'role': 'system', 'content': 'System prompt'}, {'role': 'user', 'content': prompt}]
                    audit.append({'component': 'questions', 'candidate_sha256': active['candidate_sha256'],
                        'messages': messages, 'messages_sha256': canonical_sha256(messages),
                        'system_sha256': canonical_sha256(messages[0]['content']),
                        'applied_directives': {key: canonical_sha256(value) for key, value in active['node_directives'].items()}})
                    return {'response': 'Please discuss the available options with your genetics team.'
                        if active['candidate_id'] == 'baseline' else 'A genetics team can explain the available options.'}
                if url.endswith('/state'):
                    return {'candidate_sha256': active['candidate_sha256'], 'prompt_audit': audit, 'notes': {}}
                raise AssertionError(url)

            with patch('spj_evolve.request_json', side_effect=request):
                result = run_sweep(args)
            self.assertEqual(len(result['records']), 2)
            self.assertFalse((root / 'sweep/failure.json').exists())
            manifest = build_generation_dataset(root / 'sweep', root / 'pairs', seed=123)
            self.assertEqual(manifest['presentation_count'], 2)
            presentations = [json.loads(line) for line in (root / 'pairs/presentations.jsonl').read_text().splitlines()]
            self.assertNotEqual(presentations[0]['transcript_A'], presentations[0]['transcript_B'])
            self.assertEqual(presentations[0]['transcript_A'], presentations[1]['transcript_B'])
            self.assertEqual(presentations[0]['transcript_B'], presentations[1]['transcript_A'])
            inventory = json.loads((root / 'sweep/parent_inventory/summary.json').read_text())
            self.assertEqual(inventory['status'], 'UNSCORED_CANDIDATE_INVENTORY')

    def test_baseline_keeps_original_system_prompt_and_node_behavior(self):
        state = CounsellingState()
        self.assertEqual(build_system_message(state), build_system_message(state, baseline_candidate()))
        left, right = FakeBackend(), FakeBackend()
        baseline = LangGraphMNDAgent(left)
        explicit = LangGraphMNDAgent(right, baseline_candidate())
        self.assertEqual(baseline.ask("What does a negative result mean?"),
                         explicit.ask("What does a negative result mean?"))
        self.assertEqual(left.calls, right.calls)

    def test_real_graph_delivers_system_and_question_directives_and_audits_hashes(self):
        selected = candidate(system="SYSTEM_SENTINEL", questions="QUESTION_SENTINEL")
        backend = FakeBackend()
        agent = LangGraphMNDAgent(backend, selected)
        agent.ask("What does a negative result mean?")
        delivered = [messages for messages in backend.calls if "QUESTION_SENTINEL" in messages[-1]["content"]]
        self.assertTrue(delivered)
        self.assertIn("SYSTEM_SENTINEL", delivered[-1][0]["content"])
        self.assertIn("Result interpretation guardrails", delivered[-1][0]["content"])
        audit = agent.prompt_audit[-1]
        self.assertEqual(audit["candidate_sha256"], selected["candidate_sha256"])
        self.assertEqual(audit["messages_sha256"], canonical_sha256(delivered[-1]))
        self.assertEqual(audit["applied_directives"], {
            "system": canonical_sha256("SYSTEM_SENTINEL"),
            "questions": canonical_sha256("QUESTION_SENTINEL"),
        })
        self.assertNotIn("messages", audit)
        restored = LangGraphMNDAgent(FakeBackend())
        restored.load_dict(agent.to_dict())
        self.assertEqual(restored.prompt_candidate, selected)
        self.assertEqual(restored.prompt_audit, agent.prompt_audit)

    def test_stage_directives_reach_each_mutable_stage_and_preserve_retrieval(self):
        methods = {"background": "_background_node", "education": "_education_node",
                   "questions": "_questions_node", "wrap_up": "_wrap_up_node"}
        for component, method in methods.items():
            with self.subTest(component=component):
                backend = FakeBackend()
                agent = LangGraphMNDAgent(backend, candidate(**{component: "NODE_SENTINEL"}))
                graph_state = agent._initial_graph_state("What does a negative result mean?")
                getattr(agent, method)(graph_state)
                self.assertIn("NODE_SENTINEL", backend.calls[-1][-1]["content"])
                self.assertEqual(agent.prompt_audit[-1]["component"], component)
                if component in {"education", "questions"}:
                    self.assertTrue(any("curated references" in message["content"] for message in backend.calls[-1]))

    def test_effective_prompt_recording_requires_both_offline_flags(self):
        for enabled in (False, True):
            with patch.dict(os.environ, {"MND_RECORD_EFFECTIVE_PROMPTS": "1",
                                        "MND_ALLOW_PROMPT_CANDIDATES": "1" if enabled else "0"}):
                agent = LangGraphMNDAgent(FakeBackend(), candidate(questions="NODE_SENTINEL"))
                agent.ask("What does a negative result mean?")
                self.assertEqual("messages" in agent.prompt_audit[-1], enabled)

    def test_original_spj_runtime_contract_authenticates_delivered_messages(self):
        from experiments.spj_runtime_contract import validate_run_delivery

        selected = candidate(system="SYSTEM_SENTINEL", questions="QUESTION_SENTINEL")
        patient_turn = "What does a negative result mean?"
        with patch.dict(os.environ, {"MND_RECORD_EFFECTIVE_PROMPTS": "1", "MND_ALLOW_PROMPT_CANDIDATES": "1"}):
            agent = LangGraphMNDAgent(FakeBackend(), selected)
            reply = agent.ask(patient_turn)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            output = run / "transcripts" / "scripted" / "Example" / "001"
            output.mkdir(parents=True)
            (output / "conversation.json").write_text(json.dumps([
                {"role": "user", "content": patient_turn},
                {"role": "assistant", "content": reply},
            ]), encoding="utf-8")
            (output / "state.json").write_text(json.dumps({
                "candidate_sha256": selected["candidate_sha256"],
                "prompt_audit": agent.prompt_audit, "notes": agent.state.notes,
            }), encoding="utf-8")
            result = validate_run_delivery(run, selected, {"Example": {"turns": [patient_turn]}})
            self.assertEqual(result["status"], "VALIDATED")
            self.assertEqual(result["applied_directives"], {"system": 1, "questions": 1})

    def test_invalid_hash_or_immutable_node_is_rejected(self):
        selected = candidate(questions="NODE_SENTINEL")
        selected["node_directives"]["questions"] = "Changed after hashing"
        with self.assertRaisesRegex(ValueError, "sha256"):
            LangGraphMNDAgent(FakeBackend(), selected)
        with self.assertRaisesRegex(ValueError, "immutable or unknown"):
            candidate(extract_information="Do something else")
        with self.assertRaises(ValueError):
            LangGraphMNDAgent(FakeBackend(), {})


class CandidateAPITests(unittest.TestCase):
    def start_context(self, directory, prompt_path=None):
        args = agent_api.build_args(['--backend', 'api', '--model', 'test-model'])
        args.prompt_candidate = str(prompt_path) if prompt_path else None
        return patch.multiple(agent_api, args=args, SESSION_STATE_DIR=Path(directory))

    def test_reset_candidate_is_opt_in_validated_and_isolated_per_session(self):
        selected = candidate(system="SYSTEM_SENTINEL", questions="QUESTION_SENTINEL")
        with tempfile.TemporaryDirectory() as directory, self.start_context(directory), \
                patch.object(agent_api, 'warm_knowledge_retrieval'), \
                patch.dict(os.environ, {'MND_ALLOW_PROMPT_CANDIDATES': '0'}):
            with TestClient(agent_api.app) as client:
                request = {'session_id': 'candidate-session', 'prompt_candidate': selected}
                self.assertEqual(client.post('/reset', json=request).status_code, 403)
                with patch.dict(os.environ, {'MND_ALLOW_PROMPT_CANDIDATES': '1'}):
                    response = client.post('/reset', json=request)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()['candidate_sha256'], selected['candidate_sha256'])
                    bad = dict(selected, candidate_sha256='invalid')
                    self.assertEqual(client.post('/reset', json={'prompt_candidate': bad}).status_code, 422)
                self.assertEqual(client.get('/state', params={'session_id': 'candidate-session'}).json()['candidate_id'], selected['candidate_id'])
                self.assertEqual(client.get('/state', params={'session_id': 'other-session'}).json()['candidate_id'], 'baseline')
                with patch.object(agent_api, 'backend', RuntimeFakeBackend()), patch.dict(os.environ, {'MND_RETRIEVAL_MODE': 'lexical'}):
                    agent_api.agents['candidate-session'].backend = agent_api.backend
                    self.assertEqual(client.post('/chat', json={'session_id': 'candidate-session', 'message': 'What does a negative result mean?'}).status_code, 200)
                state = client.get('/state', params={'session_id': 'candidate-session'}).json()
                self.assertTrue(state['prompt_audit'])
                self.assertEqual(state['prompt_audit'][-1]['candidate_sha256'], selected['candidate_sha256'])
                # Persisted candidate cannot silently become active after offline mode ends.
                agent_api.cleanup_idle_sessions(force=True)
                self.assertEqual(client.get('/state', params={'session_id': 'candidate-session'}).status_code, 409)
                reset = client.post('/reset', json={'session_id': 'candidate-session'})
                self.assertEqual(reset.status_code, 200)
                self.assertEqual(reset.json()['candidate_id'], 'baseline')

    def test_startup_selected_candidate_applies_without_allowing_browser_mutation(self):
        selected = candidate(questions="QUESTION_SENTINEL")
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / 'selected.json'
            prompt_path.write_text(json.dumps(selected), encoding='utf-8')
            with self.start_context(directory, prompt_path), patch.object(agent_api, 'warm_knowledge_retrieval'), \
                    patch.dict(os.environ, {'MND_ALLOW_PROMPT_CANDIDATES': '0'}):
                with TestClient(agent_api.app) as client:
                    self.assertEqual(client.get('/health').json()['candidate_sha256'], selected['candidate_sha256'])
                    reset = client.post('/reset', json={'session_id': 'web-user'})
                    self.assertEqual(reset.json()['candidate_id'], selected['candidate_id'])
                    agent_api.cleanup_idle_sessions(force=True)
                    state = client.get('/state', params={'session_id': 'web-user'})
                    self.assertEqual(state.status_code, 200)
                    self.assertEqual(state.json()['candidate_sha256'], selected['candidate_sha256'])
                    self.assertEqual(client.post('/reset', json={'session_id': 'web-user', 'prompt_candidate': baseline_candidate()}).status_code, 403)

    def test_invalid_startup_candidate_fails_before_backend_or_download(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / 'invalid.json'
            prompt_path.write_text('{}', encoding='utf-8')
            with self.start_context(directory, prompt_path), patch.object(agent_api, 'build_backend') as backend, \
                    patch.object(agent_api, 'warm_knowledge_retrieval') as retrieval:
                with self.assertRaises(ValueError), TestClient(agent_api.app):
                    pass
                backend.assert_not_called()
                retrieval.assert_not_called()


if __name__ == '__main__':
    unittest.main()
