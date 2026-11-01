import pytest
from fastapi.testclient import TestClient
from ai import ProviderError, REFUSAL
from main import app, Assistant, get_assistant


class StubLLM:
    """Test double only: never used by the running application."""
    def generate(self, question, chunks):
        return {'answer': 'Test double response', 'source': f'Page {chunks[0].page}'}


@pytest.fixture
def client(knowledge):
    app.dependency_overrides.clear()
    # Routes call this helper directly, so patch the cached helper in its module.
    import main as module
    original = module.get_assistant
    module.get_assistant = lambda: Assistant(knowledge, StubLLM())
    with TestClient(app) as test_client:
        yield test_client
    module.get_assistant = original


def test_supported_question_contract(client):
    result = client.post('/ask', json={'question': 'What are the total bootcamp fees?'})
    assert result.status_code == 200
    assert result.json() == {'answer': 'Test double response', 'source': 'Page 16'}


def test_no_context_refuses_without_llm(knowledge):
    class MustNotRun:
        def generate(self, *args):
            raise AssertionError('LLM should not run without context')
    assert Assistant(knowledge, MustNotRun()).ask('qzxvbnmasdfghjkl') == {
        'answer': REFUSAL, 'source': None}


@pytest.mark.parametrize('body', [{}, {'question': ''}, {'question': '   '}, {'question': 12},
    {'question': None}, {'question': ['hello']}, {'question': 'x' * 1001},
    {'question': 'fees?', 'extra': True}])
def test_invalid_input_is_json_422(client, body):
    result = client.post('/ask', json=body)
    assert result.status_code == 422
    assert 'detail' in result.json()


def test_malformed_json(client):
    result = client.post('/ask', content='{bad', headers={'Content-Type': 'application/json'})
    assert result.status_code == 422
    assert result.headers['content-type'].startswith('application/json')


def test_provider_failure_is_503(client, knowledge, monkeypatch):
    class BrokenLLM:
        def generate(self, *args):
            raise ProviderError('LLM unavailable')
    monkeypatch.setattr('main.get_assistant', lambda: Assistant(knowledge, BrokenLLM()))
    result = client.post('/ask', json={'question': 'What are the total bootcamp fees?'})
    assert result.status_code == 503
    assert result.json() == {'detail': 'LLM unavailable'}


def test_missing_index_is_503(monkeypatch, tmp_path):
    monkeypatch.setenv('INDEX_DIR', str(tmp_path / 'missing'))
    get_assistant.cache_clear()
    with TestClient(app) as client:
        result = client.post('/ask', json={'question': 'Fees?'})
    get_assistant.cache_clear()
    assert result.status_code == 503


def test_health_and_readiness(client):
    assert client.get('/health').json()['status'] == 'ok'
    assert client.get('/ready').json()['llm_verified'] is False
