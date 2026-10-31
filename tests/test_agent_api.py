import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import requests

import agent_api
from api_backend import APIBackend
from model_config import ModelConfig


class AgentAPITests(unittest.TestCase):
    @patch('api_backend.requests.post')
    def test_selected_model_endpoint_and_token_parameter_reach_provider(self, post):
        post.return_value.json.return_value = {'choices': [{'message': {'content': ' A reply. '}}]}
        backend = APIBackend(ModelConfig(
            role='agent', model='vendor/chosen-model', api_base='http://localhost:9000/v1/',
            api_key='test-private-key', timeout=12, max_tokens=800,
            token_parameter='max_completion_tokens',
        ))
        self.assertEqual(backend.chat([{'role': 'user', 'content': 'hello'}]), 'A reply.')
        self.assertEqual(post.call_args.args[0], 'http://localhost:9000/v1/chat/completions')
        request = post.call_args.kwargs
        self.assertEqual(request['headers']['Authorization'], 'Bearer test-private-key')
        self.assertEqual(request['json']['model'], 'vendor/chosen-model')
        self.assertEqual(request['json']['max_completion_tokens'], 800)
        self.assertNotIn('max_tokens', request['json'])
        self.assertNotIn('temperature', request['json'])
        self.assertEqual(request['timeout'], 12)

    @patch('api_backend.requests.post')
    def test_provider_failures_do_not_expose_credentials(self, post):
        backend = APIBackend(ModelConfig(role='agent', model='chosen-model', api_key='test-private-key'))
        post.side_effect = requests.HTTPError('test-private-key provider response', response=Mock(status_code=401))
        with self.assertRaisesRegex(RuntimeError, 'HTTP 401') as caught:
            backend.chat([])
        self.assertNotIn('test-private-key', str(caught.exception))
        post.side_effect = requests.Timeout('test-private-key URL')
        with self.assertRaisesRegex(RuntimeError, 'could not be reached') as caught:
            backend.chat([])
        self.assertNotIn('test-private-key', str(caught.exception))

    def test_import_never_loads_local_inference_or_parses_external_arguments(self):
        script = '''import sys
sys.argv = ['external-tool', '--unrelated-option']
import agent_api
assert agent_api.backend is None
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert not any(t.name == 'mnd-session-cleanup' for t in __import__('threading').enumerate())
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_clinical_follow_up_questions_stay_with_counselling_agent(self):
        for text in ("What follow-up questions should I ask my genetics team?",
                     "Can you help me generate a title for my family conversation?",
                     "Would a web search help me find a genetics clinic?"):
            message = agent_api.OpenAIMessage(role="user", content=text)
            self.assertEqual(agent_api.detect_openwebui_task([message]), "chat")

    def test_explicit_openwebui_internal_task_templates_are_detected(self):
        tasks = {
            "Suggest 3-5 relevant follow-up questions based on the chat history.": "followups",
            "Generate a concise title summarizing the chat history.": "title",
            "Generate 1-3 broad tags and return JSON.": "tags",
            "Analyze the chat history to generate web search queries.": "search",
        }
        for instruction, expected in tasks.items():
            message = agent_api.OpenAIMessage(
                role="user", content=f"### Task:\n{instruction}\n### Chat History:\nI need a follow-up appointment.",
            )
            self.assertEqual(agent_api.detect_openwebui_task([message]), expected)
        message = agent_api.OpenAIMessage(role="user", content=(
            "### Task:\nAnswer this question.\n### Chat History:\nGenerate follow-up questions."
        ))
        self.assertEqual(agent_api.detect_openwebui_task([message]), "chat")

    def test_backend_selection_and_argument_overrides(self):
        args = agent_api.build_args(['--backend', 'api', '--model', 'custom-agent',
                                     '--api-base', 'http://localhost:9000/v1',
                                     '--max_new_tokens', '700'])
        backend = agent_api.build_backend(args)
        self.assertIsInstance(backend, APIBackend)
        self.assertEqual(backend.config.model, 'custom-agent')
        self.assertEqual(backend.config.max_tokens, 700)
        self.assertEqual(backend.config.api_base, 'http://localhost:9000/v1')
        args.model = ''
        with self.assertRaisesRegex(ValueError, 'MND_AGENT_MODEL'):
            agent_api.build_backend(args)

    def test_http_chat_health_state_and_reset_with_configured_api(self):
        args = agent_api.build_args(['--backend', 'api', '--model', 'custom-agent', '--api-base', 'http://localhost:9000/v1'])
        with tempfile.TemporaryDirectory() as directory, patch.object(agent_api, 'args', args), \
                patch.object(agent_api, 'SESSION_STATE_DIR', Path(directory)), \
                patch.object(agent_api, 'warm_knowledge_retrieval'), \
                patch.object(agent_api.LangGraphMNDAgent, 'ask', return_value='Configured model reply.'):
            with TestClient(agent_api.app) as client:
                self.assertEqual(client.get('/health').json()['agent_model'], 'custom-agent')
                response = client.post('/chat', json={'session_id': 'test-session', 'message': 'hello'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['response'], 'Configured model reply.')
                self.assertEqual(client.get('/v1/models').json()['data'][0]['id'], agent_api.SERVED_MODEL_NAME)
                response = client.post('/v1/chat/completions', json={
                    'model': 'mnd-local-agent', 'user': 'test-session',
                    'messages': [{'role': 'user', 'content': 'hello'}],
                })
                self.assertEqual(response.json()['choices'][0]['message']['content'], 'Configured model reply.')
                self.assertEqual(client.post('/reset', json={'session_id': 'test-session'}).status_code, 200)
                self.assertTrue((Path(directory) / 'test-session.json').is_file())
            self.assertIsNone(agent_api.backend)
            self.assertEqual(agent_api.agents, {})


if __name__ == '__main__':
    unittest.main()
