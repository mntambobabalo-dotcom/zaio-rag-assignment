from pathlib import Path
import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai import LLM, ProviderError, REFUSAL
from database import KnowledgeBase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question: str = Field(min_length=1, max_length=1000)

    @field_validator('question')
    @classmethod
    def strip_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('Enter a non-empty question.')
        return value


class Answer(BaseModel):
    answer: str
    source: str | None


class Assistant:
    def __init__(self, knowledge, llm, min_score=None):
        self.knowledge, self.llm, self.min_score = knowledge, llm, min_score

    def ask(self, question):
        hits = self.knowledge.search(question, min_score=self.min_score)
        if not hits:
            return {'answer': REFUSAL, 'source': None}
        return self.llm.generate(question, [chunk for chunk, score in hits])


@lru_cache(maxsize=1)
def get_assistant():
    directory = Path(os.getenv('INDEX_DIR', str(ROOT / 'data/index')))
    try:
        kb = KnowledgeBase.load(directory)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(503, 'Knowledge base unavailable. Run python load_data.py.') from exc
    score = os.getenv('MIN_SCORE')
    return Assistant(kb, LLM(), float(score) if score else None)


app = FastAPI(title='ZAIO Student Handbook Assistant', version='1.0.0',
              description='Page-cited answers from the March 2026 student handbook.')


@app.get('/health')
def health():
    return {'status': 'ok', 'note': 'API process is running; use /ready for index readiness.'}


@app.get('/ready')
def ready():
    assistant = get_assistant()
    return {'status': 'ready', 'chunks': len(assistant.knowledge.chunks),
            'llm_verified': False,
            'note': 'Index loaded. LLM connectivity is verified only by a successful supported /ask request.'}


@app.post('/ask', response_model=Answer)
def ask(request: Question):
    try:
        return get_assistant().ask(request.question)
    except ProviderError as exc:
        raise HTTPException(503, str(exc)) from exc
