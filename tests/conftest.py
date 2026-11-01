from pathlib import Path
import os
import pytest


@pytest.fixture(scope='session')
def knowledge(tmp_path_factory):
    from database import KnowledgeBase
    backend = os.getenv('TEST_EMBEDDINGS', 'semantic')
    return KnowledgeBase.build(Path('data/handbook.pdf'), tmp_path_factory.mktemp('index'), backend=backend)
