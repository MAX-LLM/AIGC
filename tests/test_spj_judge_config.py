"""Native SPJ request/configuration and result validation tests, without API calls."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evaluation.common import DOMAIN_NAMES
from evaluation.frontier_pairwise_batch import build_provider_request, import_batch_results
from evaluation.frontier_pointwise_batch import import_batch_results as import_pointwise
from evaluation.provider_config import make_client, model_role, native_config
from experiments import manage_frontier_pairwise_batch as pairwise
from experiments import manage_frontier_pointwise_batch as pointwise


def domains():
    return {name: {'score': 1, 'rationale': 'Evidence from the example transcript.'} for name in DOMAIN_NAMES}


def presentation():
    transcript = [{'role': 'user', 'content': 'What does testing involve?'}, {'role': 'assistant', 'content': 'Discuss the options with your team.'}]
    return {'presentation_id': 'P-example', 'vignette': 'Synthetic case.', 'transcript_A': transcript, 'transcript_B': transcript}


class SPJJudgeConfigTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.loader = patch('evaluation.provider_config.load_environment')
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_independent_roles_and_primary_reasoning_effort(self):
        with patch.dict(os.environ, {
            'MND_JUDGE_MODEL': 'chosen-judge', 'MND_JUDGE_API_KEY': 'judge-key',
            'MND_JUDGE_REASONING_EFFORT': 'medium', 'MND_JUDGE_EFFORT': 'low',
            'MND_PROPOSAL_MODEL': 'chosen-proposer', 'MND_PROPOSAL_API_KEY': 'proposal-key',
            'MND_PROPOSAL_API_BASE': 'http://localhost:9001',
        }):
            judge = native_config('judge').validate()
            proposal = native_config('proposal').validate()
        self.assertEqual((judge.provider, judge.model, judge.effort), ('openai', 'chosen-judge', 'medium'))
        self.assertEqual((proposal.provider, proposal.model, proposal.api_key), ('anthropic', 'chosen-proposer', 'proposal-key'))
        self.assertNotIn('proposal-key', repr(proposal))
        self.assertNotIn('api_key', proposal.public_dict())
        self.assertEqual(model_role({'generator': {'model': 'chosen-proposer'}}), 'proposal')
        self.assertEqual(model_role({'model_role': 'proposal'}), 'proposal')

    def test_sdk_clients_receive_independent_endpoints_and_no_write_retries(self):
        openai = Mock()
        anthropic = Mock()
        with patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=openai), 'anthropic': SimpleNamespace(Anthropic=anthropic)}), patch.dict(os.environ, {
            'MND_JUDGE_API_BASE': 'http://localhost:9100/v1', 'MND_JUDGE_API_KEY': 'judge-key',
            'MND_PROPOSAL_API_BASE': 'http://localhost:9101/v1', 'MND_PROPOSAL_API_KEY': 'proposal-key',
        }):
            make_client('openai', 'judge')
            make_client('anthropic', 'proposal')
        self.assertEqual(openai.call_args.kwargs['base_url'], 'http://localhost:9100/v1')
        self.assertEqual(openai.call_args.kwargs['api_key'], 'judge-key')
        self.assertEqual(anthropic.call_args.kwargs['base_url'], 'http://localhost:9101')
        self.assertEqual(anthropic.call_args.kwargs['api_key'], 'proposal-key')
        self.assertEqual(anthropic.call_args.kwargs['max_retries'], 0)

    def test_native_payloads_preserve_selected_model_and_effort(self):
        kwargs = dict(custom_id='request1', prompt='Evaluate this.', schema={'type': 'object'}, model='arbitrary-model', effort='medium', max_output_tokens=1234)
        response_request = build_provider_request(provider='openai', **kwargs)
        messages_request = build_provider_request(provider='anthropic', **kwargs)
        self.assertEqual(response_request['url'], '/v1/responses')
        self.assertEqual(response_request['body']['reasoning']['effort'], 'medium')
        self.assertEqual(response_request['body']['model'], 'arbitrary-model')
        self.assertEqual(messages_request['params']['output_config']['effort'], 'medium')
        self.assertEqual(messages_request['params']['max_tokens'], 1234)

    def test_provider_specific_reasoning_effort_is_checked_before_requests(self):
        kwargs = dict(custom_id='request1', prompt='Evaluate.', schema={'type': 'object'}, model='chosen-model', max_output_tokens=1024)
        for effort in ('none', 'minimal', 'xhigh'):
            with self.subTest(effort=effort):
                self.assertEqual(build_provider_request(provider='openai', effort=effort, **kwargs)['body']['reasoning']['effort'], effort)
                with self.assertRaisesRegex(ValueError, 'Anthropic reasoning effort'):
                    native_config('proposal', provider='anthropic', effort=effort)
                with self.assertRaisesRegex(ValueError, 'Anthropic reasoning effort'):
                    build_provider_request(provider='anthropic', effort=effort, **kwargs)

    def test_pairwise_prepare_and_original_score_import_accept_custom_judge(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'MND_JUDGE_MODEL': 'custom-judge', 'MND_JUDGE_API_KEY': 'never-written'}):
            root = Path(directory)
            presentations = root / 'presentations.jsonl'
            presentations.write_text(json.dumps(presentation()) + '\n')
            output = root / 'batch'
            pairwise.main(['prepare', '--presentations', str(presentations), '--output-dir', str(output), '--judge-id', 'spj-custom'])
            manifest_text = (output / 'batch_manifest.json').read_text()
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest['judge']['model'], 'custom-judge')
            self.assertEqual(manifest['model_role'], 'judge')
            self.assertNotIn('never-written', manifest_text)
            payload = json.loads((output / 'requests.jsonl').read_text())
            self.assertEqual(payload['body']['model'], 'custom-judge')
            score = {'schema_version': '1.0.0', 'presentation_id': 'P-example', 'preference': {'winner': 'tie', 'strength': 0, 'rationale': 'Equivalent.'}, 'direct_question_resolution': {'winner': 'tie', 'rationale': 'Equivalent.'}, 'scores': {'A': domains(), 'B': domains()}, 'critical_error_candidates': [], 'uncertainties': []}
            result = root / 'result.jsonl'
            result.write_text(json.dumps({'custom_id': 'P-example', 'response': {'status_code': 200, 'body': {'id': 'resp-test', 'model': 'custom-judge', 'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(score)}]}]}}}) + '\n')
            imported = import_batch_results(provider='openai', results_path=result, batch_dir=output, output_dir=root / 'scores')
            self.assertEqual(imported['failed_evaluations'], 0)
            stored = json.loads((root / 'scores' / 'P-example.score.json').read_text())
            self.assertEqual(stored['score']['computed_totals'], {'A': 8, 'B': 8})
            self.assertFalse(stored['score']['critical_errors_adjudicated'])

    def test_pointwise_anthropic_prepare_and_nested_score_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = {'item_id': 'I-example', 'vignette': 'Synthetic case.', 'transcript': presentation()['transcript_A']}
            source = root / 'items.jsonl'
            source.write_text(json.dumps(item) + '\n')
            output = root / 'batch'
            pointwise.main(['prepare', '--items', str(source), '--output-dir', str(output), '--provider', 'anthropic', '--model', 'custom-native-judge', '--effort', 'low'])
            request = json.loads((output / 'requests.jsonl').read_text())
            self.assertEqual(request['params']['model'], 'custom-native-judge')
            score = {'schema_version': '1.0.0', 'item_id': 'I-example', 'scores': domains(), 'direct_question_resolution': {'score': 1, 'rationale': 'Partially answered.'}, 'critical_error_candidates': [], 'uncertainties': []}
            result = root / 'result.jsonl'
            result.write_text(json.dumps({'custom_id': 'I-example', 'result': {'type': 'succeeded', 'message': {'id': 'msg-test', 'model': 'custom-native-judge', 'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': json.dumps(score)}]}}}) + '\n')
            imported = import_pointwise(provider='anthropic', results_path=result, batch_dir=output, output_dir=root / 'scores')
            self.assertEqual(imported['failed_evaluations'], 0)
            stored = json.loads((root / 'scores' / 'I-example.score.json').read_text())
            self.assertEqual(stored['score']['computed_total'], 8)

    def test_prepared_input_tampering_is_rejected_before_any_client(self):
        with tempfile.TemporaryDirectory() as directory, patch('experiments.manage_frontier_pairwise_batch.make_client') as client:
            root = Path(directory)
            (root / 'requests.jsonl').write_text('{}\n')
            (root / 'batch_manifest.json').write_text(json.dumps({'batch_input_path': 'requests.jsonl', 'batch_input_sha256': 'incorrect', 'request_count': 1}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                pairwise.load_prepared_batch(root)
            client.assert_not_called()

    def test_submit_requires_explicit_gate_before_creating_provider_client(self):
        with patch('experiments.manage_frontier_pairwise_batch.make_client') as client:
            with self.assertRaisesRegex(ValueError, 'Paid submission blocked'):
                pairwise.command_submit(argparse.Namespace(confirm_paid_submission='not-approved'))
            client.assert_not_called()

    def test_invalid_embedded_credentials_cannot_enter_manifest(self):
        with self.assertRaisesRegex(ValueError, 'without credentials'):
            native_config('judge', api_base='https://user:secret@example.com/v1')


if __name__ == '__main__':
    unittest.main()
