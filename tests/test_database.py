from pathlib import Path
import pytest
from database import KnowledgeBase
from read_pdf import split_text, clean_text


@pytest.mark.parametrize('question,page', [
    ('What are the total bootcamp fees?', 16),
    ('When must fees be paid in full?', 16),
    ('What minimum internet speeds are recommended?', 5),
    ('When are live classes held?', 11),
    ('Where must assignments be submitted?', 13),
    ('How much does the final project contribute to the final grade?', 14),
    ('Who is the lead instructor?', 15),
    ('Who is my dedicated tutor?', 20),
    ('Which channel is used for communication after orientation?', 18),
    ('What do green calendar tiles mean?', 9),
    ('What is the minimum SSD storage?', 5),
    ('When are tutor support hours?', 20),
    ('Which financing partners are available?', 16),
    ('What does a red calendar tile mean?', 9),
])
def test_real_handbook_retrieval(knowledge, question, page):
    assert page in [c.page for c, score in knowledge.search(question)]


def test_round_trip(knowledge, tmp_path):
    original = KnowledgeBase.build(Path('data/handbook.pdf'), tmp_path, backend=knowledge.manifest['embedding_backend'])
    restored = KnowledgeBase.load(tmp_path)
    before = original.search('What are the total bootcamp fees?')
    after = restored.search('What are the total bootcamp fees?')
    assert [(c.id, s) for c, s in before] == [(c.id, s) for c, s in after]
    assert restored.index.ntotal == len(restored.chunks)


def test_unknown_question(knowledge):
    assert knowledge.search('qzxvbnmasdfghjkl') == []


def test_tutor_schedule_excludes_class_schedule(knowledge):
    if knowledge.manifest['embedding_backend'] != 'semantic':
        pytest.skip('Requires neural embeddings.')
    hits = knowledge.search('When are tutor support hours?')
    assert hits and all(chunk.page == 20 for chunk, _ in hits)


def test_unrelated_question(knowledge):
    assert knowledge.search('What is the capital of Japan?') == []


def test_chunks_preserve_overlap_and_last_words():
    text = ' '.join(f'word{i}' for i in range(325))
    chunks = split_text(text)
    assert chunks[0].split()[-35:] == chunks[1].split()[:35]
    assert chunks[-1].endswith('word324')
    assert all(len(c.split()) <= 150 for c in chunks)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text('hello', size=10, overlap=10)


def test_team_roles_preserved(knowledge):
    team = '\n'.join(c.text for c in knowledge.chunks if c.page == 15)
    assert 'Kago Kola — Lead Instructor' in team
    assert 'Suhana Patel — Student Success' in team


def test_cleaning():
    assert clean_text('  Some\n\n text\u00ad\x02  ') == 'Some\ntext'
