import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_config import ModelConfig, load_environment


class ModelConfigTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_roles_have_independent_models_and_credentials(self):
        with patch('model_config.load_environment'), patch.dict(os.environ, {
            'MND_AGENT_MODEL': 'counsellor', 'MND_AGENT_API_KEY': 'agent-secret',
            'MND_JUDGE_MODEL': 'judge', 'MND_JUDGE_API_KEY': 'judge-secret',
            'MND_PROPOSAL_MODEL': 'proposer', 'MND_PROPOSAL_API_BASE': 'http://localhost:9000/v1',
        }):
            agent = ModelConfig.from_env('agent').validate()
            judge = ModelConfig.from_env('judge').validate()
            proposal = ModelConfig.from_env('proposal').validate()
        self.assertEqual((agent.model, judge.model, proposal.model), ('counsellor', 'judge', 'proposer'))
        self.assertEqual((agent.api_key, judge.api_key), ('agent-secret', 'judge-secret'))
        self.assertEqual(proposal.api_key, '')
        self.assertEqual(proposal.api_base, 'http://localhost:9000/v1')
        self.assertEqual((agent.max_tokens, judge.max_tokens), (512, 4096))
        self.assertNotIn('judge-secret', repr(judge))
        self.assertNotIn('api_key', judge.public_dict())

    @patch.dict(os.environ, {}, clear=True)
    def test_primary_alias_fallback_and_cli_precedence(self):
        with patch('model_config.load_environment'), patch.dict(os.environ, {
            'MND_JUDGE_MODEL': 'primary', 'JUDGE_MODEL': 'alias',
            'JUDGE_API_BASE': 'http://localhost:9000/v1', 'OPENAI_API_KEY': 'fallback',
        }):
            config = ModelConfig.from_env('judge')
            self.assertEqual(config.model, 'primary')
            self.assertEqual(config.api_key, 'fallback')
            self.assertEqual(config.api_base, 'http://localhost:9000/v1')
            self.assertEqual(ModelConfig.from_env('judge', model='cli').model, 'cli')
            self.assertEqual(ModelConfig.from_env('judge', model=None).model, 'primary')
            self.assertEqual(ModelConfig.from_env('judge', api_key='').api_key, '')

    def test_env_keeps_exported_values_and_never_evaluates_shell(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'EXISTING': 'exported'}, clear=True):
            source = Path(directory) / '.env'
            source.write_text('EXISTING=file\nexport MODEL="vendor/model" # comment\nSECRET=\'a#b\'\nLITERAL=$(touch /tmp/no-command)\nOTHER=$HOME # comment\nEMPTY= # comment\n', encoding='utf-8')
            load_environment(source)
            self.assertEqual(os.environ['EXISTING'], 'exported')
            self.assertEqual(os.environ['MODEL'], 'vendor/model')
            self.assertEqual(os.environ['SECRET'], 'a#b')
            self.assertEqual(os.environ['LITERAL'], '$(touch /tmp/no-command)')
            self.assertEqual(os.environ['OTHER'], '$HOME')
            self.assertEqual(os.environ['EMPTY'], '')

    def test_env_error_does_not_reveal_secret_or_partially_apply(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            source = Path(directory) / '.env'
            source.write_text('KEY=secret\nBAD="unterminated-secret\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'line 2') as caught:
                load_environment(source)
            self.assertNotIn('secret', str(caught.exception))
            self.assertNotIn('KEY', os.environ)

    @patch.dict(os.environ, {}, clear=True)
    def test_served_alias_does_not_choose_upstream_agent_model(self):
        with patch('model_config.load_environment'), patch.dict(os.environ, {'MND_MODEL': 'public-alias'}):
            self.assertEqual(ModelConfig.from_env('agent').model, '')
        with patch('model_config.load_environment'), patch.dict(os.environ, {
            'LLM': 'legacy-local-path', 'AGENT_MODEL': 'role-alias', 'MND_AGENT_MODEL': 'primary-model',
            'MND_MAX_NEW_TOKENS': '900', 'MND_AGENT_MAX_TOKENS': '1200',
        }):
            config = ModelConfig.from_env('agent')
            self.assertEqual(config.model, 'primary-model')
            self.assertEqual(config.max_tokens, 1200)

    def test_public_metadata_rejects_credentials_embedded_in_url(self):
        config = ModelConfig(role='proposal', api_base='https://user:secret@example.com/v1')
        with self.assertRaises(ValueError) as caught:
            config.public_dict()
        self.assertNotIn('secret', str(caught.exception))

    def test_invalid_settings_fail_before_request(self):
        for settings in ({'model': ''}, {'api_base': 'file:///tmp/x'},
                         {'api_base': 'https://user:secret@example.com/v1'},
                         {'timeout': float('nan')}, {'max_tokens': 0},
                         {'token_parameter': 'unsupported'}, {'temperature': -0.1}):
            values = {'role': 'judge', 'model': 'judge-model'}
            values.update(settings)
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ModelConfig(**values).validate()


if __name__ == '__main__':
    unittest.main()
