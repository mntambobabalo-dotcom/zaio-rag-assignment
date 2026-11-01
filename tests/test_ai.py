import json
import httpx
import pytest
from ai import LLM, ProviderError, REFUSAL, validate_answer
from read_pdf import Chunk

CHUNK = Chunk('p16-c0', 16, 'Total Fees: R 29,950. Payment deadline: 30 March 2026.')
VALID = {'found': True, 'answer': 'R 29,950',
         'evidence': [{'chunk_id': 'p16-c0', 'quote': 'Total Fees: R 29,950.'}]}


def test_validated_source():
    assert validate_answer(VALID, [CHUNK]) == {
        'answer': 'R 29,950', 'source': 'Page 16'}


@pytest.mark.parametrize('evidence', [[], [{'chunk_id': 'p99-c0', 'quote': 'Total Fees: R 29,950.'}],
    [{'chunk_id': 'p16-c0', 'quote': 'The total fees are R 1.'}],
    [{'chunk_id': 'p16-c0', 'quote': ''}], ['invalid']])
def test_invented_or_missing_evidence_refuses(evidence):
    result = validate_answer({**VALID, 'evidence': evidence}, [CHUNK])
    assert result == {'answer': REFUSAL, 'source': None}


def test_model_refusal():
    assert validate_answer({'found': False}, [CHUNK]) == {'answer': REFUSAL, 'source': None}


def test_real_quote_does_not_authorize_an_invented_answer():
    result = validate_answer({**VALID, 'answer': 'R 1,000'}, [CHUNK])
    assert result == {'answer': REFUSAL, 'source': None}


def test_direct_extractive_answer():
    result = validate_answer({'found': True, 'answer': 'R 29,950', 'chunk_id': CHUNK.id}, [CHUNK])
    assert result == {'answer': 'R 29,950', 'source': 'Page 16'}


def test_extract_retains_schedule_conditions():
    chunk = Chunk('schedule', 11, 'We have live classes on Monday mornings every week for the first twelve weeks, then once a month.')
    result = validate_answer({'found': True,
        'answer': 'We have live classes on Monday mornings every week', 'chunk_id': 'schedule'}, [chunk])
    assert result['answer'] == chunk.text


@pytest.mark.parametrize('answer,chunk_id', [('R 1,000', CHUNK.id), ('R 29,950', 'p99-c0'),
                                          ('R 29,950', None)])
def test_direct_extract_rejects_wrong_fact_or_source(answer, chunk_id):
    assert validate_answer({'found': True, 'answer': answer, 'chunk_id': chunk_id}, [CHUNK]) == {
        'answer': REFUSAL, 'source': None}


def test_multiple_sources_only_used_pages():
    other = Chunk('p17-c0', 17, 'The financing partner will contact you within 24 hours.')
    result = validate_answer({**VALID, 'evidence': VALID['evidence'] + [
        {'chunk_id': other.id, 'quote': other.text}]}, [CHUNK, other, Chunk('unused', 22, 'Unused')])
    assert result['source'] == 'Page 16; Page 17'


def mock_client(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr('ai.httpx.Client',
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_ollama_transport_contract(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == '/api/chat'
        assert body['stream'] is False and body['format'] == 'json'
        assert 'R 29,950' in body['messages'][1]['content']
        assert 'untrusted' in body['messages'][0]['content']
        return httpx.Response(200, json={'message': {'content': json.dumps(VALID)}})
    mock_client(monkeypatch, handler)
    assert LLM(provider='ollama', base_url='http://localhost:11434').generate(
        'What are the fees?', [CHUNK])['source'] == 'Page 16'


def test_compatible_transport_contract(monkeypatch):
    def handler(request):
        assert request.url.path == '/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer test-key'
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(VALID)}}]})
    mock_client(monkeypatch, handler)
    result = LLM(provider='compatible', base_url='https://example.test/v1',
                 model='test', api_key='test-key').generate('Fees?', [CHUNK])
    assert result['source'] == 'Page 16'


def test_provider_error_is_not_a_fake_answer(monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(503))
    with pytest.raises(ProviderError):
        LLM(provider='ollama').generate('Fees?', [CHUNK])


def test_invalid_model_output(monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(200, json={'message': {'content': 'not JSON'}}))
    with pytest.raises(ProviderError):
        LLM(provider='ollama').generate('Fees?', [CHUNK])


def test_missing_key():
    with pytest.raises(ProviderError):
        LLM(provider='compatible', api_key='', base_url='https://example.test/v1').generate('Fees?', [CHUNK])
